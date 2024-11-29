from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from tqdm import tqdm
from preprocess import *
from merge import merge_data
from config import Config
from transformers import BitsAndBytesConfig
from dataset_creation import split_data


import numpy as np

def compute_claim_embeddings(claims, tokenizer, model, device, save_path="claim_embeddings.npy"):
    """
    Compute and save embeddings for all claims using LLaMA.
    """
    print("Computing claim embeddings...")
    model.eval()  
    embeddings = []

    with torch.no_grad():
        for claim in tqdm(claims, desc="Encoding Claims"):
            inputs = tokenizer(claim, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            claim_embedding = outputs.hidden_states[-1].mean(dim=1)  # Mean pooling of the last hidden state
            embeddings.append(claim_embedding.cpu().numpy())

    # Stack all embeddings into a single array
    embeddings = np.vstack(embeddings)

    # Save embeddings to a file
    np.save(save_path, embeddings)
    print(f"Claim embeddings saved to {save_path}")

    return embeddings


def retrieve_top_k_claims(post, claim_embeddings, claims, tokenizer, model, device, k=10):
    """
    Retrieve top-k claims for a post using precomputed claim embeddings and LLaMA post embeddings.
    """
    # Compute embedding for the post
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(post, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        post_embedding = outputs.hidden_states[-1].mean(dim=1)  # Mean pooling of the last hidden state
        post_embedding = post_embedding.cpu().numpy()  # Convert to numpy array

    # Reshape post_embedding to ensure it's 2D (1, embedding_dim)
    post_embedding = post_embedding.reshape(1, -1)

    # Reshape claim_embeddings to ensure it's 2D (num_claims, embedding_dim)
    claim_embeddings = claim_embeddings.reshape(claim_embeddings.shape[0], -1)

    # Compute cosine similarity between post embedding and claim embeddings
    similarities = cosine_similarity(post_embedding, claim_embeddings).flatten()

    # Get top-k indices
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [claims[i] for i in top_k_indices], similarities[top_k_indices]



def success_at_k_batch(posts, claim_embeddings, claims, ground_truths, tokenizer, model, device, k=10, batch_size=16):
    """
    Compute Success@K using precomputed claim embeddings.
    """
    success_count = 0
    total = len(posts)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating Success@K"):
        batch_posts = posts[i:i + batch_size]
        batch_ground_truths = ground_truths[i:i + batch_size]

        for post, ground_truth in zip(batch_posts, batch_ground_truths):
            retrieved_claims, _ = retrieve_top_k_claims(post, claim_embeddings, claims, tokenizer, model, device, k)
            if ground_truth in retrieved_claims:
                success_count += 1

    return success_count / total


def evaluate_success_at_k(dataset, test_df, post_column, claim_column, claim_embeddings, tokenizer, model, device, k=10, batch_size=16):
    """
    Evaluate Success@K using precomputed claim embeddings.
    """
    posts = test_df[post_column].tolist()
    claims = dataset[claim_column].tolist()
    ground_truths = test_df[claim_column].tolist()

    return success_at_k_batch(posts, claim_embeddings, claims, ground_truths, tokenizer, model, device, k, batch_size)


def main():
    fact_checks = pd.read_csv("/home/stud/haroonm0/localdisk/FactCheck/Dataset/fact_checks.csv")
    posts = pd.read_csv("/home/stud/haroonm0/localdisk/FactCheck/Dataset/posts.csv")
    pairs = pd.read_csv("/home/stud/haroonm0/localdisk/FactCheck/Dataset/pairs.csv")

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

    mergedata = merge_data(posts, fact_checks, pairs)
    #mergedata = mergeddata.sample(frac=0.0001)
    train_df, val_df, test_df = split_data(posts, mergedata)

    
    claims = mergedata['original_claim'].tolist()
    claim_embeddings = compute_claim_embeddings(claims, tokenizer, model, device)


    success_at_k_score_original = evaluate_success_at_k(
        mergedata,
        test_df,
        post_column='text_original',
        claim_column='original_claim',
        claim_embeddings=claim_embeddings,
        tokenizer=tokenizer,
        model=model,
        device=device,
        k=10,
        batch_size=16
    )

    claims_trans = mergedata['translated_claim'].tolist()
    claim_embeddings_trans = compute_claim_embeddings(claims_trans, tokenizer, model, device)
    success_at_k_score_translated = evaluate_success_at_k(
        mergedata,
        test_df,
        post_column='text_translated',
        claim_column='translated_claim',
        claim_embeddings=claim_embeddings_trans,
        tokenizer=tokenizer,
        model=model,
        device=device,
        k=10,
        batch_size=16
    )

    print(f"Overall Success@10: {success_at_k_score_translated:.4f}")
    print(f"Overall Success@10: {success_at_k_score_original:.4f}")

    with open("output.txt", "w") as file:
        file.write("success for original success@k {success_at_k_score_original}.\n")
        file.write("success for translated {success_at_k_score_translated:.4f} .\n")


if __name__ == "__main__":
    main()
