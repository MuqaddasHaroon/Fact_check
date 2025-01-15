
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
#from config.config import *
#from lama_utils import *
from real_work.pre_process.utilities import *
from real_work.pre_process.pre_process_orig import *
import json
import numpy as np
import pandas as pd

alpaca_prompt = """ ### Instruction:
Determine if the following claim is supported by the given post.

### Post:
{}

### Claim:
{}

### Response:
{}."""



# Embed posts and claims using SentenceTransformer
def embed_posts_and_claims(posts, claims, embedding_model_name='all-MiniLM-L6-v2'):
    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer(embedding_model_name)
    
    print("Generating embeddings for posts...")
    post_embeddings = embedder.encode(posts, convert_to_tensor=True)
    
    print("Generating embeddings for claims...")
    claim_embeddings = embedder.encode(claims, convert_to_tensor=True)
    
    return post_embeddings, claim_embeddings

# Retrieve top-k claims for each post
def retrieve_top_claims(post_embeddings, claim_embeddings, claims, top_k=150):
    print("Calculating cosine similarities...")
    results = []
    for post_idx, post_embedding in enumerate(post_embeddings):
        similarities = cosine_similarity(
            post_embedding.unsqueeze(0).cpu().numpy(),
            claim_embeddings.cpu().numpy()
        )[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Get top-k indices sorted by descending similarity
        top_claims = [{"claim_text": claims[i], "claim_id": i} for i in top_indices]
        
        results.append({
            "post_id": post_idx,  # Use post_id from JSON if available
            "top_claims": top_claims  # Include claims and their IDs
        })
    return results

# Rank claims for a post
def rank_claims(post, claims, model, tokenizer, max_length=1024, batch_size=4):
    """
    Ranks claims for a given post by relevance using the fine-tuned model.
    """
    scores = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i : i + batch_size]
        inputs = [alpaca_prompt.format(post, claim["claim_text"], "") for claim in batch_claims]

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

        # Use the last token's logits as relevance scores
        logits = logits.to(torch.float32)
        batch_scores = logits[:, -1, tokenizer.convert_tokens_to_ids("Supported")].cpu().numpy()
        scores.extend(batch_scores)

    # Rank claims by scores
    ranked_indices = torch.tensor(scores).argsort(descending=True).numpy()  # Descending order
    ranked_claims = [{"claim_text": claims[idx]["claim_text"], "claim_id": claims[idx]["claim_id"], "score": scores[idx]} for idx in ranked_indices]

    return ranked_claims[:10]  # Return top 10 claims


