
import os
import time
import ast


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample


from config import DEVICE, model_save_path, MODEL_NAME, EPOCHS
from utils import custom_collate_fn

def evaluate_model(working_path):
    model = SentenceTransformer(MODEL_NAME).to(DEVICE)

    test_data = pd.read_csv(f"{working_path}test_data.csv")
    org_data = pd.read_csv(f"{working_path}train_data_with_negatives.csv")

    positive_test_data = test_data[test_data['label'] == 1]

    anchor_embeddings = model.encode(positive_test_data['processed_text'].tolist(), convert_to_tensor=True)
    claim_embeddings = model.encode(org_data['original_claim'].tolist(), convert_to_tensor=True)

    similarities = cosine_similarity(anchor_embeddings.cpu(), claim_embeddings.cpu())
    predictions = [org_data.iloc[sim.argmax()]['original_claim'] for sim in similarities]
    binary_predictions = [1 if pred == gt else 0 for pred, gt in zip(predictions, positive_test_data['original_claim'])]

    print("Evaluation complete.")
    print(classification_report([1] * len(binary_predictions), binary_predictions))

def evaluate_model_successK(WORKING_PATH):
    import logging

    # Setup logging
    log_file = "../data_processing/distillUseBase/evaluation_log.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    logger.info("Starting evaluation...")


    model_save_path = '../kaggle/working/minilm-finetuned'
    model = SentenceTransformer(model_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(f"Loaded model from {model_save_path} on device {device}")


    test_data = pd.read_csv('../kaggle/working/test_data.csv')
    org_data = pd.read_csv('../kaggle/working/train_data_with_negatives.csv')

  
    positive_test_data = test_data[test_data['label'] == 1]
    logger.info(f"Filtered positive test data size: {positive_test_data.shape}")

    missing_claims = positive_test_data[~positive_test_data['original_claim'].isin(org_data['original_claim'])]
    if not missing_claims.empty:
        logger.warning(f"{len(missing_claims)} positive claims are missing in the claim pool.")
        print(f"Warning: {len(missing_claims)} positive claims are missing in the claim pool.")
        positive_test_data = positive_test_data[positive_test_data['original_claim'].isin(org_data['original_claim'])]


    logger.info("Encoding test posts...")
    anchor_embeddings = model.encode(positive_test_data['processed_text'].tolist(), convert_to_tensor=True)
    logger.info("Encoding claims...")
    claim_embeddings = model.encode(org_data['original_claim'].tolist(), convert_to_tensor=True)


    logger.info("Computing cosine similarity...")
    anchor_embeddings = anchor_embeddings.cpu().numpy()
    claim_embeddings = claim_embeddings.cpu().numpy()
    similarities = cosine_similarity(anchor_embeddings, claim_embeddings)


    k = 10
    success_at_k = 0
    logger.info(f"Calculating Success@{k}...")

    for i, sim_scores in enumerate(similarities):
        test_post = positive_test_data.iloc[i]['processed_text']
        logger.info(f"Test post {i}: {test_post}")

   
        top_k_indices = sim_scores.argsort()[-k:][::-1]
        top_k_claims = org_data.iloc[top_k_indices]['original_claim'].tolist()
        logger.info(f"Test post {i} - Top-{k} claims: {top_k_claims}")

 
        positive_claim = positive_test_data.iloc[i]['original_claim']
        logger.info(f"Test post {i} - Positive claim: {positive_claim}")


        try:
            positive_index = org_data[org_data['original_claim'] == positive_claim].index[0]
            logger.info(f"Positive claim index: {positive_index}")
        except IndexError:
            logger.warning(f"Positive claim for test post {i} not found in claim pool.")
            continue

        if positive_index in top_k_indices:
            success_at_k += 1


    success_at_k /= len(positive_test_data)

    # Log and print the final Success@K score
    logger.info(f"Final Success@{k}: {success_at_k:.4f}")
    print(f"Final Success@{k}: {success_at_k:.4f}")


    if success_at_k == 0:
        logger.warning("Success@k is 0. Debugging failure case:")
        logger.warning("- Check if positive claims for all test posts are included in the claim pool.")
        logger.warning("- Check if embeddings for posts and claims are meaningful.")
        logger.warning("- Check if top-k indices are consistent with the expected results.")

    print(f"Evaluation completed. Logs saved to {log_file}.")
