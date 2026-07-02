import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GARI-IMPORT Car Price Predictor",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS STYLING (WORKING BACKGROUND + SIMPLE UI LIKE YOUR 2ND APP)
# ============================================================

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
    }

    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1502877338535-766e1452684a");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0px 10px 40px rgba(0,0,0,0.18);
        max-width: 900px;
    }

    h1 {
        text-align: center;
        color: #0f2a44;
        font-size: 42px;
        margin-bottom: 8px;
        font-weight: 800;
    }

    .subtitle {
        text-align: center;
        color: #0f2a44;
        font-size: 16px;
        margin-bottom: 24px;
        opacity: 0.9;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #1f77b4, #155a8a);
        color: white;
        font-size: 18px;
        border-radius: 12px;
        width: 100%;
        font-weight: 650;
        padding: 12px;
        border: none;
    }

    div.stButton > button:hover {
        filter: brightness(0.95);
        transform: translateY(-1px);
    }

    .result {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 6px solid #2e7d32;
        padding: 18px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border-radius: 10px;
        margin-top: 16px;
    }

    /* Keep text readable on the frosted card */
    .stMarkdown, .stMarkdown p, .stMarkdown li, label {
        color: #0f2a44 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL + METADATA (OPTIONAL)
# ============================================================

@st.cache_resource
def load_model_and_metadata():
    try:
        model = pk.load(open("model.pkl", "rb"))
        try:
            metadata = pk.load(open("model_metadata.pkl", "rb"))
        except Exception:
            metadata = None
        return model, metadata
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found!")
        st.info("Please run the training script first: python train_final.py")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, metadata = load_model_and_metadata()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("Cardetails.csv")

        def get_brand_name(car_name):
            return str(car_name).split(" ")[0].strip()

        df["name"] = df["name"].apply(get_brand_name)
        return df
    except FileNotFoundError:
        st.error("Dataset file 'Cardetails.csv' not found!")
        st.info("Please place the CSV file in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

cars_data = load_dataset()

# ============================================================
# CONFIGURATION (SAME LOGIC AS YOUR LONG APP)
# ============================================================

# Brand mapping (must match training code exactly)
BRAND_MAPPING = {
    'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
    'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10,
    'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13, 'Mitsubishi': 14,
    'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
    'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24,
    'Kia': 25, 'Fiat': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29,
    'Isuzu': 30, 'Opel': 31
}

# Only show brands that exist in both dataset and mapping
unique_brands = sorted([b for b in cars_data["name"].unique() if b in BRAND_MAPPING])

# Price bounds from metadata (fallback to dataset)
if metadata:
    PRICE_MIN = metadata.get("price_min", 30000)
    PRICE_MAX = metadata.get("price_max", 5000000)
    PRICE_MEDIAN = metadata.get("price_median", 450000)
else:
    PRICE_MIN = float(cars_data["selling_price"].min())
    PRICE_MAX = float(cars_data["selling_price"].max())
    PRICE_MEDIAN = float(cars_data["selling_price"].median())

# Prediction bounds (to prevent unrealistic values)
PRED_LOWER_BOUND = PRICE_MIN * 0.5
PRED_UPPER_BOUND = PRICE_MAX * 1.3

# Currency conversion rate
INR_TO_BDT = 1.05  # update if needed

# ============================================================
# HEADER
# ============================================================

st.markdown("<h1>🚘 GARI-IMPORT</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Car Price Predictor (Simple UI)</div>', unsafe_allow_html=True)

# ============================================================
# OPTIONAL: MODEL INFO (SMALL + CLEAN)
# ============================================================

if metadata:
    with st.expander("ℹ️ Model Information"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", str(metadata.get("model_name", "N/A")).replace(" Regression", ""))
        col2.metric("R²", f"{metadata.get('test_r2', 0):.3f}")
        col3.metric("MAE", f"₹{metadata.get('test_mae', 0):,.0f}")

# ============================================================
# INPUT FORM (LIKE YOUR 2ND APP)
# ============================================================

with st.expander("🚗 Enter Car Details", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        brand_display = st.selectbox("Car Brand", unique_brands if unique_brands else cars_data["name"].unique())
        year = st.slider("Manufactured Year", 1994, 2026, 2015)
        km_driven = st.slider("Kilometers Driven", 0, 200000, 50000, step=5000)
        fuel = st.selectbox("Fuel Type", sorted(cars_data["fuel"].dropna().unique()))
        transmission = st.selectbox("Transmission", sorted(cars_data["transmission"].dropna().unique()))

    with col2:
        seller_type = st.selectbox("Seller Type", sorted(cars_data["seller_type"].dropna().unique()))
        owner = st.selectbox("Owner Type", sorted(cars_data["owner"].dropna().unique()))
        mileage = st.slider("Mileage (km/l)", 10.0, 40.0, 18.0, step=0.5)
        engine = st.slider("Engine (CC)", 700, 5000, 1500, step=50)
        max_power = st.slider("Max Power (BHP)", 0.0, 200.0, 75.0, step=5.0)
        seats = st.slider("Seats", 4, 10, 5)

# ============================================================
# ACTION BUTTONS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
btn_reset, btn_predict = st.columns([1, 2])

with btn_reset:
    if st.button("🔄 Reset"):
        # Clear session state to reset widgets
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

with btn_predict:
    predict = st.button("🔮 Predict Car Price")

# ============================================================
# PREDICTION
# ============================================================

if predict:
    try:
        # Encode brand name to number (same as your long app)
        brand_encoded = BRAND_MAPPING.get(brand_display, 0)

        # Build input data to match your training features
        input_data = pd.DataFrame(
            {
                "name": [brand_encoded],
                "year": [year],
                "km_driven": [km_driven],
                "fuel": [fuel],
                "seller_type": [seller_type],
                "transmission": [transmission],
                "owner": [owner],
                "mileage": [mileage],
                "engine": [engine],
                "max_power": [max_power],
                "seats": [seats],
            }
        )

        with st.spinner("🔍 Predicting price..."):
            predicted_price_inr = float(model.predict(input_data)[0])

        # Clip unrealistic predictions (same as your long app)
        original_prediction = predicted_price_inr
        predicted_price_inr = float(np.clip(predicted_price_inr, PRED_LOWER_BOUND, PRED_UPPER_BOUND))
        predicted_price_inr = max(0.0, predicted_price_inr)

        predicted_price_bdt = predicted_price_inr * INR_TO_BDT

        st.markdown(
            f"""
            <div class="result">
                Estimated Car Price<br>
                BDT {predicted_price_bdt:,.0f} Tk<br>
                <span style="font-size:16px; opacity:0.85;">(Approx. ₹{predicted_price_inr:,.0f} INR)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Small note if it was clipped
        if abs(original_prediction - predicted_price_inr) > 1:
            st.caption("⚠️ Note: Prediction was adjusted to stay within realistic market bounds.")

        st.caption("⚠️ Prediction is an estimate based on historical data.")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.info("Make sure the model was trained with the same feature columns and encodings.")

# ============================================================
# ABOUT SECTION (LIKE YOUR 2ND APP)
# ============================================================

with st.expander("ℹ️ About Us"):
    st.markdown(
        """
        At *GARI-IMPORT.com.bd*, we specialize in importing high-quality Japanese reconditioned vehicles to Bangladesh.
        We are committed to transparency (authentic auction sheets) and strong after-sales support.

        *Why choose us?*
        - BARVIDA member
        - Authentic documentation (auction points, original mileage, JAAI inspection)
        - Wide selection of Japanese brands
        - After-sales support (warranty, maintenance, repairs)
        - Premium showroom in Uttara, Dhaka
        """
    )

# ============================================================
# FOOTER (SOCIAL LINKS LIKE YOUR 2ND APP)
# ============================================================

st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:16px;">
        <p><b>Connect with GARI-IMPORT</b></p>
        <a href="https://www.gari-import.com.bd" target="_blank">Website</a> |
        <a href="https://www.facebook.com/gariimportbd" target="_blank">Facebook</a> |
        <a href="https://www.youtube.com/@gariimport" target="_blank">YouTube</a> |
        <a href="https://www.instagram.com/gariimport" target="_blank">Instagram</a>
        <p style="font-size:12px; margin-top:10px;">© 2025 GARI-IMPORT.com.bd | All Rights Reserved</p>
    </div>
    """,
    unsafe_allow_html=True
)