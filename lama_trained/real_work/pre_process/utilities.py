
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
#from real_work.config.config import *
from lama_utils import *

import json
import numpy as np
import pandas as pd


import ast
import io
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional
from unsloth import FastLanguageModel, is_bfloat16_supported

import numpy as np
import pandas as pd
import torch
from accelerate import init_empty_weights, infer_auto_device_map
from peft import LoraConfig, get_peft_model
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    T5EncoderModel,
    TrainingArguments,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from trl import SFTTrainer
import re
import ast
import re
from collections import Counter
import  pandas as pd
from unsloth import FastLanguageModel

from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pandas as pd
import spacy
import re
import emoji
from transformers import pipeline


def get_primary_language(lang_conf):
    if lang_conf and isinstance(lang_conf, list):
        # Sort by confidence descending and return the language code with highest confidence
        return sorted(lang_conf, key=lambda x: x[1], reverse=True)[0][0]
    return None


def clean_text(text, lang):
    try:
        # Load the spaCy model for the specified language
        nlp = spacy.load(f"{lang}_core_news_sm")
    except OSError:
        # If the specific language model isn't available, check for supported blank models
        try:
            nlp = spacy.blank(lang)  # Try loading a blank model
        except Exception as e:
            # Handle unsupported languages
            return f"Error: Unsupported language '{lang}'. Details: {str(e)}"
    
    # Process the text
    doc = nlp(text)
    return " ".join([token.text for token in doc])
def clean_with_eng(text, lang="es"):
    nlp = spacy.load(f"{lang}_core_news_sm")  # Load spaCy model for the given language
    doc = nlp(text)

    # Tokenize and remove unwanted elements
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]

    # Remove emojis
    text = emoji.replace_emoji(" ".join(tokens), replace='')

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def summarize_column(df, column_name):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    
    for text in df[column_name]:
        if len(text) < 50:
            return text
        else:
            summary = summarizer(text, max_length=80, min_length=50, do_sample=False)
            summaries.append(summary[0]["summary_text"])

    
    return pd.Series(summaries)


def append_ocr_to_text(posts):
    """
    Append `ocr_original` and `ocr_translated` to `text_original` and `text_translated`
    only if they haven't been appended before.
    """
    for index, row in posts.iterrows():

        ocr_original = row.get("ocr_original", None)
        if ocr_original is not None:
            ocr_original = str(ocr_original).strip() 
        else:
            ocr_original = ""

        text_original = row.get("text_original", None)
        if text_original is not None:
            text_original = str(text_original).strip()  
        else:
            text_original = ""

    
        if pd.notna(ocr_original) and ocr_original != "" and ocr_original not in text_original:
            if pd.isna(text_original) or text_original == "":
                posts.at[index, "text_original"] = ocr_original
            else:
                posts.at[index, "text_original"] += f" {ocr_original}"


        ocr_translated = row.get("ocr_translated", None)
        if ocr_translated is not None:
            ocr_translated = str(ocr_translated).strip() 
        else:
            ocr_translated = ""

        text_translated = row.get("text_translated", None)
        if text_translated is not None:
            text_translated = str(text_translated).strip()  
        else:
            text_translated = ""

   
        if pd.notna(ocr_translated) and ocr_translated != "" and ocr_translated not in text_translated:
            if pd.isna(text_translated) or text_translated == "":
                posts.at[index, "text_translated"] = ocr_translated
            else:
                posts.at[index, "text_translated"] += f" {ocr_translated}"
    
    return posts

def replace_short_text(row):
    if isinstance(row['text_original'], str) and len(row['text_original'].split()) < 5:
        return row['ocr_original'] if isinstance(row['ocr_original'], str) else row['text_original']
    return row['text_original']


def append_claim_to_text(fact_check):
    """
    Append `title_original` and `title_translated` to `claim_original` and `claim_translated`
    only if they haven't been appended before.
    """
    for index, row in fact_check.iterrows():
        # Get and process `title_original`
        title_original = row.get("title_original", None)
        if title_original is not None:
            title_original = str(title_original).strip()  # Ensure it's a string and strip whitespace
        else:
            title_original = ""

        claim_original = row.get("original_claim", None)
        if claim_original is not None:
            claim_original = str(claim_original).strip()  # Ensure it's a string and strip whitespace
        else:
            claim_original = ""

        # Append `title_original` to `claim_original` if not already appended
        if pd.notna(title_original) and title_original != "" and title_original not in claim_original:
            if pd.isna(claim_original) or claim_original == "":
                fact_check.at[index, "original_claim"] = title_original
            else:
                fact_check.at[index, "original_claim"] += f" {title_original}"

        # Get and process `title_translated`
        title_translated = row.get("title_translated", None)
        if title_translated is not None:
            title_translated = str(title_translated).strip()  # Ensure it's a string and strip whitespace
        else:
            title_translated = ""

        claim_translated = row.get("translated_claim", None)
        if claim_translated is not None:
            claim_translated = str(claim_translated).strip()  # Ensure it's a string and strip whitespace
        else:
            claim_translated = ""

        # Append `title_translated` to `claim_translated` if not already appended
        if pd.notna(title_translated) and title_translated != "" and title_translated not in claim_translated:
            if pd.isna(claim_translated) or claim_translated == "":
                fact_check.at[index, "translated_claim"] = title_translated
            else:
                fact_check.at[index, "translated_claim"] += f" {title_translated}"
    
    return fact_check

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


def merge_data(posts, fact_checks, pairs):
    """
    Merge posts and fact_checks using pairs as the bridge.
    """

    posts = posts.drop_duplicates(subset='post_id')
    fact_checks = fact_checks.drop_duplicates(subset='fact_check_id')
    pairs = pairs.drop_duplicates(subset=['post_id', 'fact_check_id'])


    merged_data = pairs.merge(posts, on='post_id', how='left').merge(fact_checks, on='fact_check_id', how='left')


    merged_data.drop(
        columns=['instances_x', 'verdicts', 'ocr_confidence', 'instances_y', 'confidence'],
        inplace=True,
        errors='ignore'
    )

    return merged_data


def replace_with_ocr_if_special(df):
    """
    Replace text_original and text_translated if they consist only of emojis, special symbols, or are empty.
    """
    def contains_only_special_chars(text):
        if not isinstance(text, str):
            return False
        # Check if text consists only of special symbols, whitespace, or emojis
        pattern = re.compile(r"^[\s\W\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$")
        return bool(pattern.match(text))
    
    modified_rows = []  # To store rows that are modified
    
    for index, row in df.iterrows():
        original_text = row.get('text_original', None)
        translated_text = row.get('text_translated', None)
        
        if contains_only_special_chars(original_text) or pd.isna(original_text) or original_text.strip() == "":
            # Replace text_original with ocr_original
            df.at[index, 'text_original'] = row.get('ocr_original', "")
            modified_rows.append((index, 'text_original', original_text, row['text_original']))
        
        if contains_only_special_chars(translated_text) or pd.isna(translated_text) or translated_text.strip() == "":
            # Replace text_translated with ocr_translated
            df.at[index, 'text_translated'] = row.get('ocr_translated', "")
            modified_rows.append((index, 'text_translated', translated_text, row['text_translated']))
    
    return df


def replace_text_with_ocr(df, text_column='text', ocr_column='ocr'):
    df[text_column] = df.apply(
        lambda row: row[ocr_column] if pd.isna(row[text_column]) or row[text_column].strip() == '' else row[text_column],
        axis=1
    )
    return df




def safe_literal_eval(val):
    try:
        if isinstance(val, str):
            parsed_val = ast.literal_eval(val)
            if isinstance(parsed_val, list):
                for item in parsed_val:
                    if isinstance(item, tuple) and len(item) == 3:
                        return item  # Return the first valid tuple
            elif isinstance(parsed_val, tuple) and len(parsed_val) == 3:
                return parsed_val
        return val
    except (ValueError, SyntaxError) as e:
        print(f"Literal eval failed for {val}: {e}")
        return (None, None, None)


def split_text_column(row, row_name):
    try:
        parsed = safe_literal_eval(row[row_name])
        if isinstance(parsed, tuple) and len(parsed) == 3:
            first_text = parsed[0].strip() if isinstance(parsed[0], str) else None
            second_text = parsed[1].strip() if isinstance(parsed[1], str) else None

            # Extract the language with the highest confidence
            if isinstance(parsed[2], list) and all(isinstance(item, tuple) for item in parsed[2]):
                lang_conf = max(parsed[2], key=lambda x: x[1] if len(x) == 2 else 0)
            else:
                lang_conf = (None, None)

            lang = lang_conf[0]
            confidence = lang_conf[1]
            return pd.Series([first_text, second_text, lang, confidence])
    except Exception as e:
        print(f"Error processing row: {row[row_name]} -> {e}")
    return pd.Series([None, None, None, None])


