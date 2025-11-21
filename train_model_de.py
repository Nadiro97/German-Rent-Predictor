import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

RAW_FILE = "immo_data.csv"
CLEAN_FILE = "DE_clean_basic.csv"
MODEL_FILE = "de_totalrent_basic.pkl"
AVG_FILE = "avg_totalrent_by_area.pkl"


def safe_read_csv(filename):
    with open(filename, "r", errors="ignore") as f:
        sample = f.read(20000)
    sep = "," if sample.count(",") > sample.count(";") else ";"
    return pd.read_csv(filename, sep=sep, engine="python", on_bad_lines="skip")


def clean_dataset():
    df = safe_read_csv(RAW_FILE)

    # Keep only the basics
    keep_cols = ["totalRent", "regio1", "regio2", "livingSpace", "noRooms"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Drop missing essentials
    df = df.dropna(subset=["totalRent", "regio1", "regio2", "livingSpace", "noRooms"])

    # Remove insane outliers
    df = df[df["totalRent"].between(100, 10000)]
    df = df[df["livingSpace"].between(10, 400)]
    df = df[df["noRooms"].between(1, 10)]

    # Save clean file
    df.to_csv(CLEAN_FILE, index=False)
    print("Saved:", CLEAN_FILE, df.shape)
    return df


def train_model(df):
    X = df[["regio1", "regio2", "livingSpace", "noRooms"]]
    y = df["totalRent"]

    cat_features = ["regio1", "regio2"]
    preproc = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preproc),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=2,
            random_state=42
        ))
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    print("R²:", round(r2_score(y_val, pred), 3))
    print("MAE:", round(mean_absolute_error(y_val, pred), 2), "EUR")

    joblib.dump(model, MODEL_FILE)
    print("Model saved:", MODEL_FILE)

    avg = df.groupby(["regio1", "regio2"])["totalRent"].mean()
    avg.to_pickle(AVG_FILE)
    print("Area averages saved:", AVG_FILE)


def main():
    df = clean_dataset()
    train_model(df)


if __name__ == "__main__":
    main()

