import streamlit as st
import pandas as pd
import joblib

# ----- FILES (MATCH YOUR FOLDER) -----
DATA_FILE = "DE_clean_basic.csv"
MODEL_FILE = "de_totalrent_basic.pkl"
AVG_FILE = "avg_totalrent_by_area.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_avg():
    return pd.read_pickle(AVG_FILE)


def main():
    st.set_page_config(page_title="German Rent Predictor (Simple)", layout="centered")
    st.title("- 🇩🇪 German Rent Predictor-")
    st.caption("Simple model: Location + Size + Rooms → Warmmiete : 2018-2019 (€)")

    df = load_data()
    model = load_model()
    avg = load_avg()

    # -------- 1) Select Bundesland (regio1) --------
    regio1_list = sorted(df["regio1"].dropna().unique())
    regio1 = st.selectbox("Bundesland (regio1)", regio1_list)

    df_r1 = df[df["regio1"] == regio1]
    if df_r1.empty:
        st.warning("No data for this Bundesland.")
        return

    # -------- 2) Select Stadt/Region (regio2) --------
    regio2_list = sorted(df_r1["regio2"].dropna().unique())
    regio2 = st.selectbox("Stadt / Region (regio2)", regio2_list)

    df_r2 = df_r1[df_r1["regio2"] == regio2]
    if df_r2.empty:
        st.warning("No data for this regio1/regio2 combination.")
        return

    # -------- 3) Numeric inputs --------
    rooms = st.number_input(
        "Number of rooms (noRooms)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5
    )

    size = st.number_input(
        "Living space (m²)",
        min_value=10.0,
        max_value=400.0,
        value=60.0,
        step=1.0
    )

    # -------- Predict --------
    if st.button("Predict Warmmiete"):
        input_df = pd.DataFrame([{
            "regio1": regio1,
            "regio2": regio2,
            "livingSpace": size,
            "noRooms": rooms
        }])

        pred = float(model.predict(input_df)[0])

        st.subheader("💶 Predicted miete")
        st.metric("Estimated rent Price", f"{pred:,.0f} € / month")

        # ---- Average rent sanity check ----
        try:
            avg_area = float(avg.loc[(regio1, regio2)])
            st.write(f"Average in this area: **{avg_area:,.0f} €**")

            low, high = 0.5 * avg_area, 1.5 * avg_area
            if pred < low or pred > high:
                st.warning(
                    "Prediction is far from the historical area average. "
                    "Inputs might be unusual."
                )
        except Exception:
            pass


if __name__ == "__main__":
    main()
