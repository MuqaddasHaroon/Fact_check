
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
from docarray import DocList, BaseDoc
from docarray.typing import NdArray
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from docarray import DocList
from sentence_transformers import SentenceTransformer
import numpy as np

"""

alpaca_prompt = ### Instruction:
Determine if the following claim is supported by the given post.

### Post:
{}

### Claim:
{}

### Response:
{}.

"""

alpaca_prompt = """ ### Instruction:
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
    {}. """



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

    return ranked_claims[:10]  





class MultilingualDoc(BaseDoc):
    text: str
    embedding: NdArray[384]  





class MultilingualDoc(BaseDoc):
    text: str
    embedding: NdArray[384]
    metadata: dict  # Add this to store metadata like post_id and fact_check_id


def docArray(posts, claims):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    #paraphrase-multilingual-MiniLM-L12-v2
    # Ensure `posts` and `claims` are in list format
    if isinstance(posts, pd.Series):
        posts = posts.tolist()  # Convert Series to list of strings
    if isinstance(claims, pd.DataFrame):
        claims = claims.to_dict(orient="records")  # Convert DataFrame to list of dicts

    post_embeddings = model.encode(posts)  # Directly use posts as they are now a list of strings
    claim_embeddings = model.encode([claim["original_claim"] for claim in claims])

    # Create DocLists for posts and claims with metadata
    posts_docs = DocList[MultilingualDoc]([
        MultilingualDoc(text=post, embedding=embedding, metadata={"post_id": i})
        for i, (post, embedding) in enumerate(zip(posts, post_embeddings))
    ])
    claims_docs = DocList[MultilingualDoc]([
        MultilingualDoc(text=claim["original_claim"], embedding=embedding, metadata={"fact_check_id": claim["fact_check_id"]})
        for claim, embedding in zip(claims, claim_embeddings)
    ])

    # Find the top 50 claims for each post
    results = []
    for post in posts_docs:
        similarities = [
            np.dot(post.embedding, claim.embedding) 
            for claim in claims_docs
        ]
        top_indices = np.argsort(similarities)[-50:][::-1]
        top_claims = [{"original_claim": claims_docs[i].text, "similarity": similarities[i], **claims_docs[i].metadata} for i in top_indices]
        results.append({"post_id": post.metadata["post_id"], "top_claims": top_claims})

    return results
