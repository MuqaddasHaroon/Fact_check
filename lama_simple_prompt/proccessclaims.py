import logging
import os
import pandas as pd
import multiprocessing
import Levenshtein
import re
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from lamas.preprocess import TextPreprocessor
from lamas.preprocess import *
from lamas.merge import *
from prompt import *

import logging
import multiprocessing
import pandas as pd
import Levenshtein
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from lamas.preprocess import *
from lamas.merge import *
from lamas.dataset_creation import *


logging.basicConfig(level=logging.DEBUG)

# ---------------------- MODEL INITIALIZATION ----------------------
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


# ---------------------- CLAIM PROCESSING ----------------------
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

    posts = test_df["text_original"].tolist()
    ground_truth_relevance = test_df["original_claim"].tolist()
    claims = test_df["original_claim"].tolist()
    logging.debug("Data preprocessing completed.")

    post_to_truth = {}
    for post, truth in zip(posts, ground_truth_relevance):
        if post not in post_to_truth:
            post_to_truth[post] = []
        post_to_truth[post].append(truth)

    return posts, claims, post_to_truth




def filter_top_claims(claims_with_scores, top_n=10):

    sorted_claims = sorted(claims_with_scores, key=lambda x: x['score'], reverse=True)
    return sorted_claims[:top_n]



def success_k(claims, truth):
   
    similarity_threshold=3
    matches = []
   
    for claim in claims:

        with open("checkingtrueprompt1.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"Post {claim}: Success so far: {truth}\n\n")
        

        if claim['claim'] is None:
            continue
        distance = Levenshtein.distance(claim['claim'], truth)
        claims = claim['claim']
        with open("claim-truth.txt", "a", encoding="utf-8") as f:
                f.write(f"{claims} ====== {truth} \n\n")
       
        if distance <= similarity_threshold:
            return 1
    return 0
       

def success_k_prompt2(claims, truth):
   
    similarity_threshold=4
    matches = []
    for claim in claims:

        with open("checkingtrueprompt2.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"Post {claim}: Success so far: {truth}\n\n")
        

        if claim['claim'] is None:
            continue
        distance = Levenshtein.distance(claim['claim'], truth)
        if distance <= similarity_threshold:
            matches.append(claim)
    
    # Log matches if any
    if matches:
        with open("success@kprompt1.txt", "a", encoding="utf-8") as f:
            for match in matches:
                f.write(str(match) + "  " + str(truth) + "\n\n")
        return 1
    
    return 0

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




def process_single_chunk3(post, claims):
    """
    Processes a single post and claims using the model to extract ranked claims.
    Args:
        post (str): The post content.
        claims (list): List of candidate claims.
    Returns:
        list: Extracted claims with relevance scores.
    """
    try:
        logging.info(f"Processing post: {post} with claims.")

        # Initialize the model and tokenizer
        model, tokenizer = initialize_model_and_tokenizer()

        # Create prompt
        prompt = create_refined_prompt2(post, claims)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        # Generate model output
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95
        )

        # Decode the output
        if isinstance(outputs, torch.Tensor):
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            with open("formatted_responses3.txt", "a", encoding="utf-8") as f:
                f.write(generated_text + "\n\n")
        else:
            raise TypeError(f"Unexpected structure for outputs: {outputs}")

        # Normalize the text
        normalized_text = generated_text.replace("“", "\"").replace("”", "\"").replace("–", "-").replace("\u00A0", " ")

        # Define patterns for claim extraction
        patterns = [
            r"Claim:\s*\"(.*?)\"\s*-\s*Relevance:\s*([\d.]+)",  # Claims with explicit scores
            r"\*\*Claim \d+:\*\* (.*?) - Relevance: ([\d.]+)",  # Markdown-style claims with scores
            r"^\d+\.\s*\"(.*?)\" - Relevance: ([\d.]+)",        # Numbered claims with scores
            r"\d+\.\s*(.*)",                                   # Generic numbered claims without scores
        ]

        # Extract claims
        all_claims = []
        for pattern in patterns:
            matches = re.findall(pattern, normalized_text)
            for match in matches:
                if len(match) == 2:  # Claims with scores
                    claim_text, relevance_score = match
                    all_claims.append({
                        "claim": claim_text.strip(),
                        "relevance": float(relevance_score)
                    })
                elif len(match) == 1:  # Claims without scores
                    claim_text = match[0]
                    all_claims.append({
                        "claim": claim_text.strip(),
                        "relevance": 0.0  # Assign a default relevance
                    })

        # Remove duplicates
        all_claims = list({claim["claim"]: claim for claim in all_claims}.values())

        # Filter results to exclude claims matching the post text
        filtered_results = [
            claim for claim in all_claims if claim["claim"] != post.strip()
        ]

        # Log extracted claims
        with open("loggedclaims.txt", "a", encoding="utf-8") as f:
            f.write(str(filtered_results) + "\n\n")

        return filtered_results

    except Exception as e:
        logging.error(f"Error processing post and claims: {e}")
        return []


def process_single_chunk(chunk):
    """
    Processes a single chunk of data for claim extraction.
    Args:
        chunk (list): A list of dictionaries with 'post' and 'claims'.
    Returns:
        list: Extracted and filtered claims from the chunk.
    """
    try:
        # Initialize model and tokenizer
        model, tokenizer = initialize_model_and_tokenizer()

        # Ensure chunk is a list
        if isinstance(chunk, dict):
            chunk = [chunk]

        if not isinstance(chunk, list):
            logging.error(f"Invalid chunk structure: Expected a list, got {type(chunk)}")
            return []

        batch_results = []
        for example in chunk:
            try:
                # Validate example structure
                if not isinstance(example, dict) or 'post' not in example or 'claims' not in example:
                    logging.error(f"Invalid example structure: {example}")
                    batch_results.append([])
                    continue

                # Create prompt and tokenize
                prompt = create_refined_prompt(example['post'], example['claims'])
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

                input_ids = inputs["input_ids"].to("cuda")
                attention_mask = inputs["attention_mask"].to("cuda")

                # Generate model output
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95
                )

                # Decode the output
                if isinstance(outputs, torch.Tensor):
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    with open("formatted_responsesprompt1.txt", "a", encoding="utf-8") as f:
                        f.write(generated_text + "\n\n")
                else:
                    raise TypeError(f"Unexpected structure for outputs: {outputs}")

                # Patterns for claim extraction
                patterns = [
                    r"(\d+)\.\s*Claim:\s*(.*?)\s*-\s*Relevance Score:\s*([\d.]+)",  # With rank
                    r"Claim:\s*(.*?)\s*-\s*Relevance Score:\s*([\d.]+)"
                    r'Claim:\s*"(.*?)"\s*-\s*Relevance:\s*([\d.]+)'
                ]

                # Extract claims
                all_claims = []
                for pattern in patterns:
                    matches = re.findall(pattern, generated_text)
                    for match in matches:
                        if len(match) == 3:  # Rank, Claim, Score
                            all_claims.append({
                                "claim": match[1].strip(),
                                "score": float(match[2])
                            })
                        elif len(match) == 2:  # Claim, Score
                            all_claims.append({
                                "claim": match[0].strip(),
                                "score": float(match[1])
                            })

                # Filter out claims that match the post
                filtered_results = [
                    claim for claim in all_claims if claim["claim"] != example['post']
                ]

                print("Extracted Claims:", filtered_results)

                # Write claims to file
                with open("claim_responsesprompt11.txt", "a", encoding="utf-8") as f:
                    f.write(str(filtered_results) + "\n\n")

                # Append to batch results
                batch_results.append(filtered_results)

            except Exception as inner_e:
                logging.error(f"Error processing example in chunk: {inner_e}")
                batch_results.append([])

        return batch_results

    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        return []
    
def get_top_claims(post, claims):
    """
    Retrieves top 10 claims for a given post using the model.
    Args:
        post (str): The post content.
        claims (list): List of candidate claims.
    Returns:
        list: Top 10 claims with relevance scores.
    """
    try:
        logging.info(f"Retrieving top claims for post: {post}")
        
        # Call process_single_chunk3
        result = process_single_chunk3(post, claims)
        
        # Ensure the result contains valid claims
        if result and isinstance(result[0], list):
            top_claims = result[0][:10]  # Retrieve top 10 from the returned results
        else:
            logging.warning("No valid claims found, returning empty list.")
            top_claims = []

        return top_claims

    except Exception as e:
        logging.error(f"Error in get_top_claims: {e}")
        return []


def process_single_chunk2(chunk):
    try:
        model, tokenizer = initialize_model_and_tokenizer()

        if isinstance(chunk, dict):
            logging.warning("Chunk is a dictionary; wrapping in a list.")
            chunk = [chunk]

        if not isinstance(chunk, list):
            logging.error(f"Invalid chunk structure: Expected a list, got {type(chunk)}")
            return []

        batch_results = []
        for example in chunk:
            try:
                # Validate example structure
                if not isinstance(example, dict):
                    logging.error(f"Invalid example structure: Expected a dict, got {type(example)}")
                    batch_results.append([])
                    continue

                # Ensure required keys are present
                if 'post' not in example or 'claims' not in example:
                    logging.error(f"Missing required keys in example: {example}")
                    batch_results.append([])
                    continue

                # Generate model outputs
                prompt = create_refined_prompt2(example['post'], example['claims'])
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

                input_ids = inputs["input_ids"].to("cuda")
                attention_mask = inputs["attention_mask"].to("cuda")
                outputs = model.generate(input_ids=input_ids, 
                                         attention_mask=attention_mask,
                                         max_new_tokens=512,
                                         temperature=0.7,
                                         top_p=0.95)

                # Decode outputs
                if isinstance(outputs, torch.Tensor):
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    with open("formatted_responses.txt", "a", encoding="utf-8") as f:
                        f.write(generated_text + "\n\n")
                else:
                    raise TypeError(f"Unexpected structure for outputs: {outputs}")

                # Patterns for claim extraction
                normalized_text = generated_text.replace("“", "\"").replace("”", "\"").replace("–", "-").replace("\u00A0", " ")

                
                patterns = [
                r"\d+\.\s*Claim:\s*(.*?)\s*-\s*Relevance:\s*([\d.]+)",  # Match `1. Claim: text - Relevance: score`
                r"\d+\.\s*(.*?)\s*-\s*Relevance:\s*([\d.]+)", 
                 r"\d+\.\s*(.*)", 
                r"\*\*Claim \d+:\*\* (.*?) - Relevance: ([\d.]+)",  # Match `**Claim X:** text - Relevance: score`
                r'"\s*(.*?)\s*"\s*-\s*Relevance:\s*([\d.]+)',  # Match `"text" - Relevance: score`
                    ]


                # Extract claims
                all_claims = []

                for pattern in patterns:
                    matches = re.findall(pattern, normalized_text)
                    for match in matches:
                        if len(match) == 2:  # Claims with scores
                            claim_text, relevance_score = match
                            all_claims.append({
                                "claim": claim_text.strip(),
                                "relevance": float(relevance_score)
                            })
                        elif len(match) == 1:  
                            claim_text = match[0]
                            all_claims.append({
                                "claim": claim_text.strip(),
                                "relevance": 0.0  
                            })

         
                all_claims = list({claim["claim"]: claim for claim in all_claims}.values())

                # Filter results to exclude claims matching the post text
                filtered_results = [
                    claim for claim in all_claims if claim["claim"] != example['post']
                ]

                # Log and append results
                with open("loggedclaims.txt", "a", encoding="utf-8") as f:
                    f.write(str(filtered_results) + "\n\n")

                batch_results.append(filtered_results)

            except Exception as inner_e:
                logging.error(f"Error processing example in chunk: {inner_e}")
                batch_results.append([])

        return batch_results

    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        return []

