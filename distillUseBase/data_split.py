import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_val_split(working_path):


    data = pd.read_csv(f"{working_path}train_data_with_negatives.csv")


    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

   
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"I came here with path {working_path}")
    train_data.to_csv(f"{working_path}train_data.csv", index=False)
    val_data.to_csv(f"{working_path}val_data.csv", index=False)
    test_data.to_csv(f"{working_path}test_data.csv", index=False)

    print("Train, validation, and test splits created and saved!")
