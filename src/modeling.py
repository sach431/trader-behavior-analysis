from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model(df):

    required_cols = ["closed pnl", "classification", "volume_change"]

    for col in required_cols:
        if col not in df.columns:
            return None, None, None

    df = df.copy()

    # Target
    df["target"] = (df["closed pnl"] > 0).astype(int)

# Encode sentiment
    le = LabelEncoder()
    df["sentiment_encoded"] = le.fit_transform(df["classification"])

# Remove NaN values
    df = df.dropna(subset=["volume_change", "sentiment_encoded", "target"])

# Features
    X = df[["volume_change", "sentiment_encoded"]]
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report, X_test, y_test