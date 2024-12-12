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


logging.basicConfig(level=logging.DEBUG)

# ---------------------- MODEL INITIALIZATION ----------------------
def initialize_model_and_pipeline():
    logging.debug("Initializing model and pipeline.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
        offload_folder="./offload",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    logging.debug("Model and pipeline initialized successfully.")
    return pipe

# ---------------------- CLAIM PROCESSING ----------------------
def filter_claims(generated_claims, post, similarity_threshold=90):

    return [claim for claim in generated_claims if fuzz.ratio(claim['claim'], post) < similarity_threshold]





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
        if distance <= similarity_threshold:
            matches.append(claim)
    
    # Log matches if any
    if matches:
        with open("success@kprompt2.txt", "a", encoding="utf-8") as f:
            for match in matches:
                f.write(str(match) + "  " + str(truth) + "\n\n")
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



def process_single_chunk2(chunk):

    try:

        model, tokenizer = initialize_model_and_pipeline()

      
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
   
                prompt = create_refined_prompt2(example['post'], example['claims'])
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

                input_ids = inputs["input_ids"].to("cuda")
                attention_mask = inputs["attention_mask"].to("cuda")
                outputs = model.generate(input_ids=input_ids, 
                                         attention_mask=attention_mask,
                                          max_new_tokens=512,
                                         temperature = 0.5,
                                           top_p = 0.95 )
                

   
                if isinstance(outputs, torch.Tensor):
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                   # with open("formatted_responses.txt", "a", encoding="utf-8") as f:
                       # f.write(generated_text + "\n\n")
                else:
                    raise TypeError(f"Unexpected structure for outputs: {outputs}")
                pattern = r"^\d+\.\s*Claim:\s*\"(.*?)\"\s*-\s*Relevance:\s*([0-1]\.\d+)"
                matches = re.findall(pattern, generated_text, re.MULTILINE)
                
                # Log matches
               # with open("matches.txt", "a", encoding="utf-8") as f:
                    #f.write(str(mapytches) + "\n\n")

                if matches:
                    claim_list = [{"claim": match[0], "relevance": float(match[1])} for match in matches]


                    # Log generated responses and claims
                    
                    #with open("loggedclaims.txt", "a", encoding="utf-8") as f:
                       # f.write(str(claim_list) + "\n\n")

                    batch_results.append(claim_list)
                else:
                    logging.warning(f"No claims found in generated text for post: {example['post']}")
                    batch_results.append([])

            except Exception as inner_e:
                logging.error(f"Error processing example in chunk: {inner_e}")
                batch_results.append([])

        return batch_results

    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        return []
    
def extract_claims(response, patterns):
    extracted_claims = []

    for pattern in patterns:
        matches = re.findall(pattern, response)
     
        if matches:
            for match in matches:

                if len(match) == 3:  # Format with rank, claim, and score
                    extracted_claims.append({"rank": int(match[0]), "claim": match[1].strip(), "score": float(match[2])})
                    p

                elif len(match) == 2:  # Format with claim and score only
                    extracted_claims.append({"claim": match[1].strip(), "score": float(match[0])})
                   

                elif len(match) == 1:  # Format with claim only
                    extracted_claims.append({"claim": match[0].strip(), "score": None})
            
            print(f"extracted claims {extracted_claims}")
                    


    return extracted_claims

def process_single_chunk(chunk):

    try:
        # Initialize model and tokenizer
        model, tokenizer = initialize_model_and_pipeline()

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
                    r"Claim:\s*(.*?)\s*-\s*Relevance Score:\s*([\d.]+)"  # Claim with Score
                ]

                # Extract claims
                all_claims = []
                for pattern in patterns:
                    matches = re.findall(pattern, generated_text)
                    for match in matches:
                        if len(match) == 2:  # Claim and Score only
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


