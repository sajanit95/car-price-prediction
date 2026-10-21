# Fair Price Checker - run with: streamlit run app.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

from src.features import CAT_FEATURES, FEATURES, prepare_input
from src.inference import InputError, load_model, predict, verdict

st.set_page_config(page_title="Fair Price Checker", page_icon="🚗", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_explainer(_model):
    # LIME needs numbers, so categories are coded as ints
    background = pd.read_csv("artifacts/lime_background.csv")
    categories = {c: sorted(background[c].dropna().astype(str).unique()) for c in CAT_FEATURES}
    coded = background.copy()
    for c in CAT_FEATURES:
        coded[c] = coded[c].astype(str).map({v: i for i, v in enumerate(categories[c])}).fillna(-1)
    coded = coded.fillna(coded.median(numeric_only=True))

    def decode(arr):
        df = pd.DataFrame(arr, columns=FEATURES)
        for c in CAT_FEATURES:
            idx = np.clip(np.rint(df[c].astype(float)).astype(int), 0, len(categories[c]) - 1)
            df[c] = [categories[c][i] for i in idx]
        return df

    explainer = LimeTabularExplainer(
        coded[FEATURES].to_numpy(dtype=float), feature_names=FEATURES, mode="regression",
        categorical_features=[FEATURES.index(c) for c in CAT_FEATURES],
        categorical_names={FEATURES.index(c): categories[c] for c in CAT_FEATURES},
        discretize_continuous=True, random_state=42)
    return explainer, categories, (lambda arr: _model.predict(decode(arr)))


model, meta = get_model()
opts = meta["options"]

st.title("🚗 Fair Price Checker")
st.caption(f"Used car price estimate for Sri Lanka (2021 listings) | model: {meta['model']} v{meta['version']}")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(opts["brand_models"]),
                         index=sorted(opts["brand_models"]).index("Toyota"))
    model_name = st.selectbox("Model", opts["brand_models"][brand])
    year = st.number_input("Year of manufacture", opts["year_range"][0], opts["year_range"][1], 2015)
    mileage = st.number_input("Mileage (km)", 0, 1_000_000, 60_000, step=5_000)
    capacity = st.number_input("Engine capacity (cc)", 0, 8_000, 1_000, step=100,
                               help="Ignored for electric cars")
with col2:
    fuel = st.selectbox("Fuel", opts["Fuel"], index=opts["Fuel"].index("Petrol"))
    transmission = st.selectbox("Transmission", opts["Transmission"])
    body = st.selectbox("Body type", opts["Body"])
    condition = st.selectbox("Condition", opts["Condition"], index=opts["Condition"].index("Used"))
    district = st.selectbox("District", opts["district"])
    unregistered = st.checkbox("Unregistered (newly imported)")

listed = st.number_input("Listed price you are checking (Rs) - optional", 0, 500_000_000, 0, step=50_000)

if st.button("Check price", type="primary"):
    car = dict(Brand=brand, Model=model_name, Year=int(year), mileage_km=mileage, capacity_cc=capacity,
               Fuel=fuel, Transmission=transmission, Body=body, Condition=condition,
               Location=district, unregistered=int(unregistered))
    try:
        row = predict([car], model, meta).iloc[0]
    except InputError as err:
        st.error(str(err))
        st.stop()

    st.subheader(f"Estimated price: Rs {row['predicted_price']:,.0f}")
    st.write(f"Typical range: **Rs {row['fair_low']:,.0f} – Rs {row['fair_high']:,.0f}**")
    if listed > 0:
        msg = verdict(listed, row)
        (st.success if "fairly" in msg else st.warning)(f"Listed at Rs {listed:,.0f}: {msg}")

    with st.spinner("Explaining the estimate ..."):
        explainer, categories, predict_fn = get_explainer(model)
        x = prepare_input([car])
        for c in CAT_FEATURES:
            x[c] = categories[c].index(str(x[c].iloc[0])) if str(x[c].iloc[0]) in categories[c] else 0
        exp = explainer.explain_instance(x[FEATURES].to_numpy(dtype=float)[0], predict_fn, num_features=8)
    st.markdown("**Why this price? (LIME)**")
    items = exp.as_list()[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=["#2e7d32" if v > 0 else "#c62828" for _, v in items])
    ax.set_xlabel("Effect on price (Rs)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    st.pyplot(fig)
    st.caption("Green = pushes price up, red = pushes price down (for this car only).")
