import torch

# Paths
DATA_PATH = "/home/stud/haroonm0/localdisk/FactCheck/Dataset/"
WORKING_PATH = "/home/stud/haroonm0/localdisk/FactCheck/kaggle/working/"

# Model configurations
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"
model_save_path = '/home/stud/haroonm0/localdisk/FactCheck/kaggle/working/minilm-finetuned'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configurations
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.2
EPOCHS = 2
EARLY_STOPPING_PATIENCE = 3
