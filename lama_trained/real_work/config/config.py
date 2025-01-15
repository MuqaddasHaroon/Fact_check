

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

path = "/home/stud/haroonm0/localdisk/Fact_check/dataset"