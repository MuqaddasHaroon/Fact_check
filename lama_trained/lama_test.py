from config.config import *
from lama_train import *
from lama_utils import *
from real_work.pre_process.utilities import *
from pre_process.pre_process_orig import *
import json
import numpy as np
import pandas as pd



def load_model():
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    if True:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "/home/stud/haroonm0/localdisk/Fact_check/mistraal_finetune/lora_model4", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        



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

if __name__ == "__main__":
    text_type = 'text_translated'
    claim_type = 'translated_claim'
    ocr_type = 'ocr_translated'
    
    with open('/home/stud/haroonm0/localdisk/Fact_check/mistraal_finetune/crosslingual_predictions.json', 'r') as f:
        crosslingual_predictions = json.load(f)

    FastLanguageModel.for_inference(model)
    
    fact_checks = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/lama_trained/real_work/dataset/fact_checks.csv')
    posts = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/lama_trained/real_work/dataset/posts.csv')
    pairs = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/lama_trained/real_work/dataset/pairs.csv')


    posts, fact_checks = pre_process_test(posts, fact_checks, pairs)
    post_ids = list(map(int, crosslingual_predictions.keys()))

    matched_posts = posts[posts['post_id'].isin(post_ids)]

    def replace_short_text(row):
        if isinstance(row[text_type], str) and len(row[text_type].split()) < 5:
            return row[ocr_type] if isinstance(row[ocr_type], str) else row[text_type]
        return row[text_type]

    matched_posts[text_type] = matched_posts.apply(replace_short_text, axis=1)
    post_texts = matched_posts.set_index('post_id')['text_translated']

    claims = [{"translated_claim": row[text_type], "fact_check_id": row["fact_check_id"]} for _, row in fact_checks.iterrows()]
    post_embeddings, claim_embeddings = embed_posts_and_claims(post_texts.tolist(), [claim[claim_type] for claim in claims])
    top_50_claims_per_post = retrieve_top_claims(post_embeddings, claim_embeddings, claims, top_k=100)

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

    output_file = '/home/stud/haroonm0/localdisk/Fact_check/mistraal_finetune/crosslingual_predictions.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(crosslingual_predictions_native, f, indent=4)
        print(f"Updated crosslingual_predictions saved to {output_file}")
    except Exception as e:
        print("Error saving JSON:", e)
