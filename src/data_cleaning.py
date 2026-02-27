import pandas as pd

def clean_data():
    df = pd.read_csv("data/historical_data.csv")

    # Convert date column if exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Fix: Use ffill instead of deprecated method
    df = df.ffill()

    return df