from real_work.pre_process.utilities import *
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
from real_work.pre_process.utilities import *
from real_work.pre_process.utilities import TextPreprocessor
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





def pre_process(posts, fact_checks, pairs):
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
    fact_checks[['title_original', 'title_translated', 'language', 'confidence']] = fact_checks.apply(
        lambda row: split_text_column(row, 'title'), axis=1
    )
    print("Applying preprocessing to replace non-informative text...")
    posts = replace_with_ocr_if_special(posts)
    print("Appending OCR text to text fields...")
    posts = append_ocr_to_text(posts)
    fact_checks = append_claim_to_text(fact_checks)

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

    return train_df, test_df, val_df, mergedata


    return posts, fact_checks


def pre_process_test(posts, fact_checks, pairs):
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
    fact_checks[['title_original', 'title_translated', 'language', 'confidence']] = fact_checks.apply(
        lambda row: split_text_column(row, 'title'), axis=1
    )
    print("Applying preprocessing to replace non-informative text...")
    posts = replace_with_ocr_if_special(posts)
    print("Appending OCR text to text fields...")
    posts = append_ocr_to_text(posts)
    fact_checks = append_claim_to_text(fact_checks)

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


    return posts, fact_checks
