import torch


DATA_PATH = "../dataset/"
WORKING_PATH = "../kaggle/working/"


MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"
model_save_path = '../kaggle/working/minilm-finetuned'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.2
EPOCHS = 2
EARLY_STOPPING_PATIENCE = 3
