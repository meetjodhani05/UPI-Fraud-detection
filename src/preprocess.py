import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.drop_duplicates()
    data = data.fillna(0)
    return data

if __name__ == "__main__":
    data = load_data("data/upi_transactions.csv")
    clean_data = preprocess_data(data)
    clean_data.to_csv("data/cleaned_upi_transactions.csv", index=False)
    print("Data preprocessing completed.")
