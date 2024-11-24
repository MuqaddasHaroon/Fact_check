import pandas as pd
import time

def generate_negative_pairs(working_path):
    positive_pairs = pd.read_csv(f"{working_path}preprocessed_train_data.csv")
    fact_checks = pd.read_csv(f"{working_path}cleaned_fact_checks.csv")

    positive_pairs['pair'] = positive_pairs['processed_text'] + '||' + positive_pairs['original_claim']
    positive_pairs_set = set(positive_pairs['pair'])

    all_posts = positive_pairs['processed_text'].unique()
    all_claims = fact_checks['original_claim'].sample(frac=1, random_state=42).values

    negative_samples = []
    for post in all_posts:
        for claim in all_claims:
            pair = post + '||' + claim
            if pair not in positive_pairs_set:
                negative_samples.append({'processed_text': post, 'original_claim': claim, 'label': 0})
                break

    negative_pairs_df = pd.DataFrame(negative_samples)
    positive_pairs['label'] = 1
    train_data = pd.concat([positive_pairs, negative_pairs_df], ignore_index=True)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data.to_csv(f"{working_path}train_data_with_negatives.csv", index=False)

    print("Negative sampling complete.")
