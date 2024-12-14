import pandas as pd
from sklearn.model_selection import train_test_split
import random

def create_pos_neg_pairs(train_df, merged_data):
    pos_pairs = []
    for _, row in train_df.iterrows():
        pos_pairs.append({
            'post_id': row['post_id'],
            'text_original': row['text_original'],
            'text_translated': row['text_translated'],
            'fact_check_id': row['fact_check_id'],
            'original_claim': row['original_claim'],
            'translated_claim': row['translated_claim'],
            'label': 1  # Positive label
        })

    unique_fact_checks = merged_data.drop_duplicates(subset='fact_check_id')
    fact_check_ids = merged_data['fact_check_id'].unique()

    neg_pairs = []
    for _, row in train_df.iterrows():
        correct_fact_check_id = row['fact_check_id']
        available_fact_check_ids = list(set(fact_check_ids) - {correct_fact_check_id})
        negative_fact_checks = random.sample(available_fact_check_ids, k=min(3, len(available_fact_check_ids)))

        for neg_fact_id in negative_fact_checks:
            neg_fact_check_row = merged_data[merged_data['fact_check_id'] == neg_fact_id].iloc[0]
            neg_pairs.append({
                'post_id': row['post_id'],
                'text_original': row['text_original'],
                'text_translated': row['text_translated'],
                'fact_check_id': neg_fact_id,
                'original_claim': neg_fact_check_row['original_claim'],
                'translated_claim': neg_fact_check_row['translated_claim'],
                'label': 0  # Negative label
            })

    all_pairs = pos_pairs + neg_pairs
    return pd.DataFrame(all_pairs)

def split_data(posts, merged_data, test_size=0.2, val_size=0.5, random_state=42):
    unique_post_ids = posts['post_id'].unique()
    train_ids, temp_ids = train_test_split(unique_post_ids, test_size=test_size, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=val_size, random_state=random_state)

    train_df = merged_data[merged_data['post_id'].isin(train_ids)]
    val_df = merged_data[merged_data['post_id'].isin(val_ids)]
    test_df = merged_data[merged_data['post_id'].isin(test_ids)]

    return train_df, val_df, test_df
