import pandas as pd

RAW_FILE = "immo_data.csv"
CLEAN_FILE = "DE_clean.csv"

def safe_read_csv(filename):
    with open(filename, "r", errors="ignore") as f:
        sample = f.read(20000)
    sep = "," if sample.count(",") > sample.count(";") else ";"
    return pd.read_csv(filename, sep=sep, engine="python", on_bad_lines="skip")

def main():
    df = safe_read_csv(RAW_FILE)

    keep = ["totalRent", "regio1", "regio2", "livingSpace", "noRooms"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df = df.dropna(subset=["totalRent", "regio1", "regio2", "livingSpace", "noRooms"])

    df = df[df["totalRent"].between(100, 10000)]
    df = df[df["livingSpace"].between(10, 400)]
    df = df[df["noRooms"].between(1, 10)]

    df.to_csv(CLEAN_FILE, index=False)
    print("Saved clean dataset:", CLEAN_FILE, df.shape)

if __name__ == "__main__":
    main()
