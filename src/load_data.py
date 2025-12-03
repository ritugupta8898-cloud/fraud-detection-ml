import pandas as pd

def load_data(path="/Users/pratyushgupta/Documents/fraud-detection-ml/data/creditcard.csv"):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.shape)
