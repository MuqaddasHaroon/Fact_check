import logging
import os
import pandas as pd
import multiprocessing
import re
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from preprocess import TextPreprocessor, replace_text_with_ocr, split_text_column
from merge import merge_data
from dataset_creation import split_data
from preprocess import *
from proccessclaims import *

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

def pre_process():
    logging.debug("Starting data preprocessing.")

    fact_checks = pd.read_csv("/home/stud/haroonm0/localdisk/FactCheck/Dataset/fact_checks.csv")
    posts = pd.read_csv("/home/stud/haroonm0/localdisk/FactCheck/Dataset/posts.csv")
    pairs = pd.read_csv("/home/stud/haroonm0/localdisk/FactCheck/Dataset/pairs.csv")
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

    mergedata = merge_data(posts, fact_checks, pairs)
    mergedata = mergedata.drop_duplicates(subset="translated_claim", keep="first")
    mergedata = mergedata.drop_duplicates(subset="original_claim", keep="first")
    train_df, val_df, test_df = split_data(posts, mergedata)

    posts = test_df["text_translated"].tolist()
    ground_truth_relevance = test_df["translated_claim"].tolist()
    claims = test_df["translated_claim"].tolist()
    logging.debug("Data preprocessing completed.")

    post_to_truth = {}
    for post, truth in zip(posts, ground_truth_relevance):
        if post not in post_to_truth:
            post_to_truth[post] = []
        post_to_truth[post].append(truth)

    return posts, claims, post_to_truth

def create_dataset(posts, chunked_claims):
    data = {"post": [], "claims": []}
    assert len(chunked_claims) == 4, "The number of claim chunks must be exactly 4."

    for post in posts:
        for chunk in chunked_claims:
            data["post"].append(post)
            data["claims"].append(chunk)

    return Dataset.from_dict(data)

def process_dataset_with_pipeline(dataset, model, tokenizer, batch_size=8):
    results = [[] for _ in range(len(dataset))]
    progress_bar = tqdm(total=len(dataset), desc="Evaluating posts")

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_input_ids = [torch.tensor(item["input_ids"]).view(-1) for item in batch]
        batch_attention_mask = [torch.tensor(item["attention_mask"]).view(-1) for item in batch]

        padded_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to("cuda")
        padded_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to("cuda")

        outputs = model.generate(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95
        )

        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            if generated_text:
                pattern = r"Claim:\s*(.*?)\s*-\s*Relevance Score:\s*([\d.]+)"
                matches = re.findall(pattern, generated_text)
                if matches:
                    for text, score in matches:
                        results[i + j].append({"claim": text, "score": float(score)})

        progress_bar.update(len(batch))

    progress_bar.close()
    return results

def create_fixed_chunks(data, num_chunks):
    if len(data) < num_chunks:
        raise ValueError(f"Not enough data to create {num_chunks} chunks.")

    avg = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0

    for i in range(num_chunks):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end

    return chunks

def initialize_model_and_tokenizer():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
        offload_folder="./offload",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def process_with_multiprocessing(data_chunks, num_workers=4):
    print("I come here")
    try:
        if not isinstance(data_chunks, list) or not all(isinstance(chunk, list) for chunk in data_chunks):
            raise ValueError("data_chunks must be a list of lists (chunks of examples).")
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(process_single_chunk, data_chunks)
        return results
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")
        return []

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    posts, claims, post_to_truth = pre_process()
    posts = posts[:500]

    chunked_claims = create_fixed_chunks(claims, 4)

    dataset = create_dataset(posts, chunked_claims)

    model, tokenizer = initialize_model_and_tokenizer()

    data_chunks = [[entry] for entry in list(dataset)]

    results = process_with_multiprocessing(data_chunks, num_workers=4)

    results_by_post = {post: [] for post in posts}
    for post_idx, post in enumerate(posts):
        post_results = []
        for chunk_idx in range(4):
            result_idx = post_idx * 4 + chunk_idx
            if result_idx < len(results):
                post_results.extend(results[result_idx])
        results_by_post[post] = post_results

  
    
    success = 0
    for post_num, (post, claims) in enumerate(results_by_post.items(), start=1):
        try:
            valid_claims = [
                claim for claim_list in claims
                for claim in claim_list
                if isinstance(claim, dict) and 'score' in claim
            ]
            top_claims = sorted(valid_claims, key=lambda x: x['score'], reverse=True)[:10]
            relevant_truths = post_to_truth.get(post, [])
            for truth in relevant_truths:
                success += success_k(top_claims, truth)
                with open("post_success_log.txt", "w", encoding="utf-8") as log_file:
                    log_file.write(f"Post {post_num}: Success so far: {success}")
            logging.info(f"Post {post_num}: Success so far: {success}")
        except Exception as e:
            logging.error(f"Error processing post {post_num}: {e}")
    

#prompt 2
""""
import logging
import os
import pandas as pd
import multiprocessing
import re
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from lamas.preprocess import TextPreprocessor, replace_text_with_ocr, split_text_column
from lamas.merge import merge_data
from lamas.dataset_creation import split_data
from lamas.preprocess import *
from proccessclaims import *
from prompt import *

# Initialize logging
logging.basicConfig(level=logging.DEBUG)


def create_dataset(posts, chunked_claims):
    data = {"post": [], "claims": []}
    assert len(chunked_claims) == 4, "The number of claim chunks must be exactly 4."

    for post in posts:
        for chunk in chunked_claims:
            data["post"].append(post)
            data["claims"].append(chunk)

    return Dataset.from_dict(data)




def process_with_multiprocessing(data_chunks, num_workers=4):
    try:
        if not isinstance(data_chunks, list) or not all(isinstance(chunk, list) for chunk in data_chunks):
            raise ValueError("data_chunks must be a list of lists (chunks of examples).")
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(process_single_chunk2, data_chunks)
        return results
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")
        return []
    
def flatten_results(results):

    flattened_posts = []
    for post_results in results:
        flattened_claims = [
            claim
            for group in post_results
            for subgroup in group
            for claim in subgroup
            if isinstance(claim, dict) and 'claim' in claim and 'relevance' in claim
        ]
        flattened_posts.append(flattened_claims)
    return flattened_posts


def map_claims_to_posts(posts, raw_results):

    post_results = {}
    for post, result in zip(posts, raw_results):
        post_results[post] = flatten_results(result)
    return post_results

def assign_missing_scores(claims):

    descending_scores = list(reversed(range(1, len(claims) + 1)))
    for claim, score in zip(claims, descending_scores):
        if 'relevance' not in claim or claim['relevance'] is None:
            claim['relevance'] = score
    return claims

def group_claims_by_post(results):

    grouped_claims = {}

    for post_index, result in enumerate(results):
        grouped_claims[post_index] = []

        for sublist in result:
            for group in sublist:
                if isinstance(group, list):
                    # Process lists within the sublist
                    for claim in group:
                        if isinstance(claim, dict) and 'claim' in claim:
                            grouped_claims[post_index].append(claim)
                elif isinstance(group, dict) and 'claim' in group:
                    # Process individual claims in the sublist
                    grouped_claims[post_index].append(group)

    return grouped_claims


def map_posts_to_claims(posts, grouped_claims_by_post):

    post_to_claims_mapping = {}

    for index, post in enumerate(posts):
        # Ensure index does not exceed grouped_claims_by_post length
        if index < len(grouped_claims_by_post):
            claims = grouped_claims_by_post[index]
        else:
            claims = []  # Default to empty if no claims are present for this index

        post_to_claims_mapping[post] = claims

    return post_to_claims_mapping



if __name__ == "__main__":
    import logging
    import multiprocessing

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    # Handle multiprocessing start method
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Preprocess data
    posts, claims, post_to_truth = pre_process()
    posts = posts[:500]  

    # Divide claims into chunks
    chunked_claims = create_fixed_chunks(claims, 4)

    # Create dataset
    dataset = create_dataset(posts, chunked_claims)

    # Create data chunks for multiprocessing
    data_chunks = [[entry] for entry in list(dataset)]
    # Process data chunks
    results = process_with_multiprocessing(data_chunks, num_workers=4)

    # Flatten and group claims by posts
    grouped_claims_by_post = [sum(post_claims, []) for post_claims in results]
    post_to_claims = map_posts_to_claims(posts, grouped_claims_by_post)
    print(f"post to claim are {post_to_claims}")




    # Initialize success count
    success = 0

    # Iterate through posts and their associated claims
    for post_num, (post, claims) in enumerate(post_to_claims.items(), start=1):
        print(f"post and claim are {post} and claim {claims}")
        try:
            unique_claims = {}
            claimList = []
            for claim in claims:
                claim_text = claim['claim']
                relevance = claim['relevance']
                # Add only if it's not already in the dictionary
                if claim_text not in unique_claims:
                    unique_claims[claim_text] = relevance
            unique_claims_list = [{'claim': text, 'relevance': relevance} for text, relevance in unique_claims.items()]
            unique_claims_list = sorted(unique_claims_list, key=lambda x: x['relevance'], reverse=True)
            #for claim in unique_claims_list:
                #claimList.append(claim['claim'])
            #print(f"list is {len(claimList)}")
            
            #top_claims = process_single_chunk3(post, claimList)
            #if not top_claims:
            top_claims = unique_claims_list[:10]

            # Step 3: Retrieve relevant ground truth claims for the post
            relevant_truths = post_to_truth.get(post, [])

            # Step 4: Evaluate Success@K for each ground truth
            for truth in relevant_truths:
                success += success_k(top_claims, truth)

                # Log progress after processing each truth
                with open("post_success_log.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(f"Post {post_num}: Success so far: {success}\n")

            # Log progress in console
            logging.info(f"Post {post_num}: Success so far: {success}")

        except Exception as e:
            # Log any errors during processing
            logging.error(f"Error processing post {post_num}: {e}")

    # Final Success Count
    logging.info(f"Final Success Count: {success}")
""""