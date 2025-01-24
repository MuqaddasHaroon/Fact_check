
from lama_train import *
from lama_utils import *
from real_work.pre_process.utilities import *
from real_work.pre_process.pre_process_orig import *
import json
import numpy as np
import pandas as pd




from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model():
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/stud/haroonm0/localdisk/lora_model5",
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
    text_type = 'text_original' 
    claim_type = 'original_claim'
    ocr_type = 'ocr_original'
    if isinstance(row[text_type], str) and len(row[text_type].split()) < 5:
            return row[ocr_type] if isinstance(row[ocr_type], str) else row[text_type]
    return row[text_type]

def start():
    text_type = 'text_original' 
    claim_type = 'original_claim'
    ocr_type = 'ocr_original'
    
    with open('/home/stud/haroonm0/localdisk/dataset/crosslingual_predictions1.json', 'r') as f:
        crosslingual_predictions = json.load(f)

    
    
    fact_checks = pd.read_csv('/home/stud/haroonm0/localdisk/dataset/fact_checks1.csv')
    fact_checks = fact_checks.sample(frac=0.1)
    posts = pd.read_csv('/home/stud/haroonm0/localdisk/dataset/posts1.csv')
    #posts = posts.sample(frac=0.01)
    
    
    post_ids = list(map(int, crosslingual_predictions.keys()))
    
    matched_posts = posts[posts['post_id'].isin(post_ids)]


    matched_posts, fact_checks = pre_process_test(matched_posts, fact_checks, text_type)
    matched_posts.loc[:, text_type] = matched_posts.apply(replace_short_text, axis=1)
    
    post_texts = matched_posts.set_index('post_id')[text_type]
    #post_texts = post_texts[:2]
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
        print("top 50 claims are")
        print(top_50_claims_per_post)
        
    return top_50_claims_per_post, post_ids, posts, claims, post_texts 


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



alpaca_prompt = """
  
 ### Instruction:
Your task is to determine if the following claim is supported by the given post.

### Definitions:
- **Claim**: A claim is a statement that asserts something to be true, factual, or believable. It may be a fact, an opinion, or a prediction that can be verified or debated. Claims often require evidence or reasoning to determine their validity.
- **Post**: A post is a piece of text, often from social media, that is user-generated and may or may not be verified. Posts can include opinions, observations, rumors, or misinformation.

### Task:
- Compare the claim with the information provided in the post.
- Determine whether the claim is:
    1. **Supported**: The post provides sufficient evidence or information to confirm the claim.
    2. **Refuted**: The post provides sufficient evidence or information to contradict the claim.
    3. **Not Enough Information**: The post does not provide sufficient evidence to determine whether the claim is true or false.
- If the post contains contradictory or ambiguous information, explain why the claim cannot be verified.

### Guidelines:
- Carefully evaluate the relationship between the post and the claim.
- Avoid making assumptions beyond the content of the post.
- Focus on the specific evidence provided in the post.
- Provide a clear and concise response: either **Supported**, **Refuted**, or **Not Enough Information**.

### Post:
{}

### Claim:
{}

### Response:
{}.  """



from torch.nn.functional import cosine_similarity

def rank_claims_docArray(post, claims, model, tokenizer, max_length=1024):
    """
    Ranks claims for a given post by semantic similarity using embeddings.
    """
    scores = []

    for claim in claims:
        # Create input using the Alpaca-style prompt
        input_text = alpaca_prompt.format(post, claim["original_claim"], "")
        
        # Tokenize the input
        tokenized_input = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        ).to("cuda")

        # Model inference with hidden states
        with torch.no_grad():
            outputs = model(**tokenized_input, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last hidden layer

        # Compute embeddings (mean pooling)
        embedding = hidden_states.mean(dim=1)  # Shape: [batch_size, hidden_size]

        # Compute similarity (assuming single batch item)
        score = cosine_similarity(embedding[0], embedding[0], dim=0).item()
        scores.append(score)

    # Rank claims by scores
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    ranked_claims = [
        {
            "original_claim": claims[idx]["original_claim"],
            "fact_check_id": claims[idx]["fact_check_id"],
            "score": scores[idx]
        }
        for idx in ranked_indices
    ]

    return ranked_claims[:50]

if __init__ == "__main__":
    print("started with lama")
    top_50_claims_per_post, post_ids, posts, claims, post_texts = start()
    crosslingual_predictions = {}
    csv_data = []

    for post_entry in top_50_claims_per_post:
        post_id = post_ids[post_entry["post_id"]]
        if post_id not in post_texts.index:
            print(f"Warning: post_id {post_id} not found in post_texts index. Skipping.")
            continue
        post_text = post_texts.loc[post_id]
        top_claims = post_entry["top_claims"]
        #print(f"len of claims is {len(top_claims)}")
        ranked_claims = rank_claims_docArray(post_text, top_claims, model, tokenizer)
        #print("ranked claims are...")
        #print(ranked_claims)

        ranked_fact_check_ids = [claim["fact_check_id"] for claim in ranked_claims]
        crosslingual_predictions[str(post_id)] = ranked_fact_check_ids

        for rank, claim in enumerate(ranked_claims, start=1):
            csv_data.append({
                "post_id": post_id,
                "post_text": post_text,
                "rank": rank,
                "fact_check_id": claim["fact_check_id"],
                "claim_text": claim["original_claim"],
                "score": float(claim["score"])
            })

    crosslingual_predictions_native = convert_to_native_types(crosslingual_predictions)

    output_file = 'crosslingual_predictions5.json'
    with open(output_file, 'w') as f:
        json.dump(crosslingual_predictions_native, f, indent=4)
    print(f"Updated crosslingual_predictions saved to {output_file}")

#csv_file = 'retrieved_claims.csv'
#csv_df = pd.DataFrame(csv_data)
#csv_df.to_csv(csv_file, index=False)
#print(f"Retrieved claims saved to CSV at {csv_file}")

