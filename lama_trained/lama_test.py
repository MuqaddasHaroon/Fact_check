
from lama_train import *
from lama_utils import *
from real_work.pre_process.utilities import *
from real_work.pre_process.pre_process_orig import *
import json
import numpy as np
import pandas as pd




from transformers import AutoModelForCausalLM, AutoTokenizer


def rank_claims_docArray(post, claims, model, tokenizer, max_length=1024, batch_size=4):
    """
    Ranks claims for a given post by relevance using the fine-tuned model.
    """
    scores = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i : i + batch_size]
        
        # Create inputs using the Alpaca-style prompt
        inputs = [alpaca_prompt.format(post, claim["original_claim"], "") for claim in batch_claims]

        # Tokenize the batch
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        ).to("cuda")

        # Model inference
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

        # Use the logits for the token corresponding to "Supported" as relevance scores
        logits = logits.to(torch.float32)
        supported_token_id = tokenizer.convert_tokens_to_ids("Supported")
        batch_scores = logits[:, -1, supported_token_id].cpu().numpy()
        scores.extend(batch_scores)

    # Rank claims by scores
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)  # Descending order
    ranked_claims = [
        {
            "original_claim": claims[idx]["original_claim"],
            "fact_check_id": claims[idx]["fact_check_id"],
            "score": scores[idx]
        }
        for idx in ranked_indices
    ]

    return ranked_claims[:10]  # Return top 10 ranked claims


def load_model():
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/stud/haroonm0/localdisk/Fact_check/lora_model5",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto",  # Automatically map layers to GPU/CPU  # Enable CPU offload
    )
    return model, tokenizer




def convert_to_native_types(data):
    if isinstance(data, dict):
        return {str(key): convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    else:
        return data
    
def replace_short_text(row):
        if isinstance(row[text_type], str) and len(row[text_type].split()) < 5:
            return row[ocr_type] if isinstance(row[ocr_type], str) else row[text_type]
        return row[text_type]

if __name__ == "__main__":
    text_type = 'text_original' 
    claim_type = 'original_claim'
    ocr_type = 'ocr_original'
    
    with open('/home/stud/haroonm0/localdisk/Fact_check/dataset/crosslingual_predictions1.json', 'r') as f:
        crosslingual_predictions = json.load(f)

    model, tokenizer = load_model()
    FastLanguageModel.for_inference(model)
    
    fact_checks = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/dataset/fact_checks1.csv')
    #fact_checks = fact_checks.sample(frac=0.01)
    posts = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/dataset/posts1.csv')
    posts['text'] = posts['text'].apply(lambda x: x if pd.notna(x) and x.strip() != "" else None)
    posts['text'] = posts['text'].fillna(posts['ocr'])
    
    
    
    post_ids = list(map(int, crosslingual_predictions.keys()))
    
    matched_posts = posts[posts['post_id'].isin(post_ids)]
    print(len(matched_posts))

    matched_posts, fact_checks = pre_process_test(matched_posts, fact_checks, text_type)
    matched_posts.loc[:, text_type] = matched_posts.apply(replace_short_text, axis=1)
    
    post_texts = matched_posts.set_index('post_id')[text_type]
    claims = [{"original_claim": row[claim_type], "fact_check_id": row["fact_check_id"]} for _, row in fact_checks.iterrows()]
    

    print("started embeddings")
    #using normal sentence embedder
    simple = False
    if simple:
        post_embeddings, claim_embeddings = embed_posts_and_claims(post_texts.tolist(), [claim[claim_type] for claim in claims])
        print("started retriveing top 50")
        top_50_claims_per_post = retrieve_top_claims(post_embeddings, claim_embeddings, claims, top_k=50)
    else:
        top_50_claims_per_post = docArray(post_texts, claims)


    """
    crosslingual_predictions = {}
    
    csv_data = []

    for post_entry in top_50_claims_per_post:
        post_id = post_ids[post_entry["post_id"]]
        post_text = post_texts.loc[post_id]
        top_claims = post_entry["top_claims"]

        ranked_claims = rank_claims(post_text, top_claims, model, tokenizer)
        ranked_fact_check_ids = [claim["claim_text"]["fact_check_id"] for claim in ranked_claims]
        crosslingual_predictions[str(post_id)] = ranked_fact_check_ids

        for rank, claim in enumerate(ranked_claims, start=1):
            csv_data.append({
                "post_id": post_id,
                "post_text": post_text,
                "rank": rank,
                "fact_check_id": claim["claim_text"]["fact_check_id"],
                "claim_text": claim["claim_text"][claim_type],
                "score": claim["score"]
            })

    crosslingual_predictions_native = convert_to_native_types(crosslingual_predictions)

    csv_file = '/home/stud/haroonm0/localdisk/Fact_check/mistraal_finetune/retrieved_claims.csv'
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_file, index=False)
    print(f"Retrieved claims saved to CSV at {csv_file}")

    output_file = '/home/stud/haroonm0/localdisk/Fact_check/mistraal_finetune/crosslingual_predictions5.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(crosslingual_predictions_native, f, indent=4)
        print(f"Updated crosslingual_predictions saved to {output_file}")
    except Exception as e:
        print("Error saving JSON:", e) 
        """
        # Initialize the cross-lingual predictions and CSV data storage
    print("started with lama ")
    crosslingual_predictions = {}
    csv_data = []

    # Iterate over the top 50 claims for each post
    for post_entry in top_50_claims_per_post:
        # Get the post ID and its corresponding text
        post_id = post_ids[post_entry["post_id"]]
        if post_id not in post_texts.index:
            print(f"Warning: post_id {post_id} not found in post_texts index. Skipping.")
            continue
        post_text = post_texts.loc[post_id] 

        # Retrieve the top claims for the current post
        top_claims = post_entry["top_claims"]  # This already includes fact_check_id, similarity, and original_claim
        ranked_claims = rank_claims_docArray(post_text, top_claims, model, tokenizer)
        ranked_fact_check_ids = [claim["fact_check_id"] for claim in ranked_claims]     
        crosslingual_predictions[str(post_id)] = ranked_fact_check_ids

        # Collect data for CSV generation
        for rank, claim in enumerate(ranked_claims, start=1):
            csv_data.append({
                "post_id": post_id,
                "post_text": post_text,
                "rank": rank,
                "fact_check_id": claim["fact_check_id"],
                "claim_text": claim["original_claim"],
                "score": float(claim["score"])  # Convert np.float32 to float for JSON compatibility
            })

    # Convert predictions to native Python types for JSON compatibility
    crosslingual_predictions_native = convert_to_native_types(crosslingual_predictions)

    # Save predictions as a JSON file
    output_file = 'crosslingual_predictions5.json'
    with open(output_file, 'w') as f:
        json.dump(crosslingual_predictions_native, f, indent=4)
    print(f"Updated crosslingual_predictions saved to {output_file}")

    # Save the data as a CSV file
    csv_file = 'retrieved_claims.csv'
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_file, index=False)
    print(f"Retrieved claims saved to CSV at {csv_file}")
 