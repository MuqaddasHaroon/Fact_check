from data_processing import preprocess_data
from negative_sampling import generate_negative_pairs
from training import train_model
from evaluation import evaluate_model, evaluate_model_successK
from config import DATA_PATH, WORKING_PATH
from data_split import create_train_val_split

if __name__ == "__main__":
    preprocess_data(DATA_PATH, WORKING_PATH)
    generate_negative_pairs(WORKING_PATH)
    create_train_val_split(WORKING_PATH)
    train_model(WORKING_PATH)
    #evaluate_model(WORKING_PATH)
    evaluate_model_successK(WORKING_PATH)
