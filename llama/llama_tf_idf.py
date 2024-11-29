from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from preprocess import *
from merge import merge_data
from config import Config
from transformers import BitsAndBytesConfig
from tqdm import tqdm
from dataset_creation import split_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def filter_top_k_claims(post, claims, k=10):
 
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([post] + claims)
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    top_k_indices = cosine_sim.argsort()[-k:][::-1]
    return [claims[i] for i in top_k_indices]


def retrieve_claim(post, claims, tokenizer, model, device):
 
    claims_list = "\n".join([f"{i + 1}. \"{claim}\"" for i, claim in enumerate(claims)])
    prompt = f"Post: \"{post}\"\nHere are some claims:\n{claims_list}\nWhich claim best matches the post? Please return the most relevant claim."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.cuda.amp.autocast():  # Mixed precision
        outputs = model.generate(**inputs, max_new_tokens=50)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def success_at_k_batch(posts, claims, ground_truths, k, tokenizer, model, device, batch_size=16):
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    success_count = 0
    total = len(posts)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating Success@K"):
        batch_posts = posts[i:i + batch_size]
        batch_ground_truths = ground_truths[i:i + batch_size]

  
        batch_filtered_claims = [filter_top_k_claims(post, claims, k=k) for post in batch_posts]


        batch_prompts = [
            f"Post: \"{post}\"\nHere are some claims:\n" +
            "\n".join([f"{j + 1}. \"{claim}\"" for j, claim in enumerate(filtered_claims)]) +
            "\nWhich claim best matches the post? Please return the most relevant claim."
            for post, filtered_claims in zip(batch_posts, batch_filtered_claims)
        ]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model.generate(**inputs, max_new_tokens=50)

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    
        for response, ground_truth in zip(responses, batch_ground_truths):
            if ground_truth in response:
                success_count += 1

    return success_count / total


def evaluate_success_at_k(dataset, test_df, post_column, claim_column, k, tokenizer, model, device, batch_size=16):
 
    posts = test_df[post_column].tolist()
    claims = dataset[claim_column].tolist()
    ground_truths = test_df[claim_column].tolist()

    return success_at_k_batch(posts, claims, ground_truths, k, tokenizer, model, device, batch_size=batch_size)


def main():
    fact_checks = pd.read_csv("../dataset/fact_checks.csv")
    posts = pd.read_csv("../dataset/posts.csv")
    pairs = pd.read_csv("../dataset/pairs.csv")

    model_name = "meta-llama/Llama-3.1-8B"
    hf_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.float16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preprocessor = TextPreprocessor()

    if 'text' in posts.columns and 'ocr' in posts.columns:
        posts = replace_text_with_ocr(posts, text_column='text', ocr_column='ocr')

    if 'ocr' in posts.columns:
        posts[['ocr_original', 'ocr_translated', 'ocr_language', 'ocr_confidence']] = posts.apply(
            lambda row: split_text_column(row, 'ocr'), axis=1
        )

    if 'text' in posts.columns:
        posts[['text_original', 'text_translated', 'text_language', 'text_confidence']] = posts.apply(
            lambda row: split_text_column(row, 'text'), axis=1
        )

    fact_checks[['original_claim', 'translated_claim', 'language', 'confidence']] = fact_checks.apply(
        lambda row: split_text_column(row, 'claim'), axis=1
    )

    columns_to_preprocess = [
        ('translated_claim', fact_checks),
        ('text_translated', posts),
        ('ocr_translated', posts),
        ('original_claim', fact_checks),
        ('text_original', posts),
        ('ocr_original', posts)
    ]

    for col, df in columns_to_preprocess:
        if col in df.columns:
            df[col] = df[col].apply(preprocessor.preprocess)
    
    for col, df in columns_to_preprocess:
        if col in df.columns:
            df[col] = df[col].apply(preprocessor.preprocess)

    merged_data = merge_data(posts, fact_checks, pairs)
   #merged_data = merged_data.sample(frac=0.1)  # Sample for faster evaluation

    train_df, val_df, test_df = split_data(posts, merged_data)

    success_at_k_score = evaluate_success_at_k(
        merged_data,
        test_df,
        post_column='text_translated',
        claim_column='translated_claim',
        k=10,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=16
    )
    print(f"Overall Success@10 for translated with tfidf: {success_at_k_score:.4f}")

    success_at_k_score_orig = evaluate_success_at_k(
        merged_data,
        test_df,
        post_column='text_original',
        claim_column='original_claim',
        k=10,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=16
    )

    print(f"Overall Success@10 for original with tfidf: {success_at_k_score_orig:.4f}")


if __name__ == "__main__":
    main()
