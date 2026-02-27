def create_features(df):

    if "Execution Price" in df.columns:
        df["return"] = df["Execution Price"].pct_change()

    if "Size Tokens" in df.columns:
        df["volume_change"] = df["Size Tokens"].pct_change()

    df = df.dropna()

    return df