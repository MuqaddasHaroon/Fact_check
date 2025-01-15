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
from real_work.pre_process.pre_process_orig import *
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
from datasets import Dataset
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




def formatting_prompts_func(examples):
    post = examples["text_original"]
    claim = examples["original_claim"]
    label = "Supported" if examples["label"] == 1 else "Not Supported"  # Map label to response text

    text = alpaca_prompt.format(post, claim, label) + EOS_TOKEN
    return {"text": text}

def model_init():
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # Prompt template
    alpaca_prompt = """ ### Instruction:
    Determine if the following claim is supported by the given post.

    ### Post:
    {}

    ### Claim:
    {}

    ### Response:
    {}. """
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        )
    return model, tokenizer

if __name__ == "__main__":
    fact_checks = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/lama_trained/real_work/dataset/fact_checks.csv')
    posts = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/lama_trained/real_work/dataset/posts.csv')
    pairs = pd.read_csv('/home/stud/haroonm0/localdisk/Fact_check/lama_trained/real_work/dataset/pairs.csv')

    model, tokenizer = model_init()

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    train_df, test_df, val,  mergedata = pre_process(posts, fact_checks, pairs)


    train_pairs = create_pos_neg_pairs(train_df, mergedata)


    train_dataset = Dataset.from_pandas(train_pairs)

    train_dataset2 = train_dataset.shuffle(seed=42)

    train_data = train_dataset2.map(formatting_prompts_func)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=512,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            num_train_epochs=4,
            learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=500,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    ) 

    trainer_stats = trainer.train()


    model.save_pretrained("lora_model4")
    tokenizer.save_pretrained("lora_model4")