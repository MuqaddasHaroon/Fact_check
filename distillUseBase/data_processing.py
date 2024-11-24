import pandas as pd
import ast
from utils import extract_original_from_list, extract_original_from_tuple

def preprocess_data(data_path, working_path):
    # Load data
    posts = pd.read_csv(f"{data_path}posts.csv")
    pairs = pd.read_csv(f"{data_path}pairs.csv")
    fact_checks = pd.read_csv(f"{data_path}fact_checks.csv")

    # Process text in posts and claims
    posts['processed_text'] = posts.apply(
        lambda row: extract_original_from_tuple(row['text'])
        if pd.notna(row['text'])
        else extract_original_from_list(row['ocr']),
        axis=1
    )
    posts = posts.dropna(subset=['processed_text'])
    fact_checks['original_claim'] = fact_checks['claim'].apply(extract_original_from_tuple)
    fact_checks = fact_checks.dropna(subset=['original_claim'])

    # Save cleaned data
    posts[['post_id', 'processed_text']].to_csv(f"{working_path}cleaned_posts.csv", index=False)
    fact_checks[['fact_check_id', 'original_claim']].to_csv(f"{working_path}cleaned_fact_checks.csv", index=False)

    # Pair cleaned data
    train_data = pairs.merge(posts, on='post_id').merge(fact_checks, on='fact_check_id')
    train_data = train_data[['processed_text', 'original_claim']].dropna()
    train_data.to_csv(f"{working_path}train_data.csv", index=False)

    print("Data preprocessing complete.")
