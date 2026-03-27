import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Store Sales Forecaster",
    page_icon="🛒",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS  (dark-themed, premium look)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Card container */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    /* Big prediction display */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
        margin: 20px 0;
    }
    .prediction-box .label {
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 8px;
    }
    .prediction-box .value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #fff;
        line-height: 1.1;
    }
    .prediction-box .unit {
        font-size: 1rem;
        opacity: 0.7;
        margin-top: 6px;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }

    /* Streamlit widget labels */
    label { color: #d1d5db !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 40px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Number inputs */
    input[type="number"] {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Selectboxes */
    [data-baseweb="select"] {
        background: rgba(255,255,255,0.1) !important;
    }

    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }

    /* Hide default Streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("sales_model.pkl")

model = load_model()

# ─────────────────────────────────────────────
# Known categorical values (from Favorita dataset)
# ─────────────────────────────────────────────
FAMILIES = sorted([
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI", "EGGS",
    "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME CARE",
    "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE", "LIQUOR,WINE,BEER",
    "MEATS", "PERSONAL CARE", "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS", "PRODUCE",
    "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
])

CITIES = sorted([
    "Quito", "Guayaquil", "Cuenca", "Ambato", "Santo Domingo",
    "Cayambe", "Latacunga", "Ibarra", "Riobamba", "Manta",
    "Daule", "Libertad", "Salinas", "El Carmen", "Puyo",
    "Quevedo", "Guaranda", "Babahoyo", "Esmeraldas", "Playas",
    "Loja", "Machala", "Shushufindi",
])

STATES = sorted([
    "Pichincha", "Guayas", "Azuay", "Tungurahua",
    "Santo Domingo de los Tsachilas", "Imbabura", "Chimborazo",
    "Manabi", "Los Rios", "Santa Elena", "Pastaza", "Cotopaxi",
    "Esmeraldas", "El Oro", "Sucumbios", "Loja",
])

STORE_TYPES = ["A", "B", "C", "D", "E"]

HOLIDAY_TYPES = ["None", "Holiday", "Event", "Transfer", "Additional", "Bridge", "Work Day"]

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("## 🛒 Store Sales Forecaster")
    st.markdown(
        "<p style='color:#9ca3af;margin-top:-10px;'>Powered by a Random Forest pipeline trained on Favorita Grocery Sales data</p>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        "<div style='text-align:right;padding-top:10px;'>"
        "<span style='background:rgba(102,126,234,0.25);border:1px solid #667eea;"
        "border-radius:20px;padding:4px 14px;font-size:0.8rem;color:#a78bfa;'>ML Model Active ✓</span>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────
# Input form  — two-column layout
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("<div class='section-title'>🏪 Store Information</div>", unsafe_allow_html=True)

    store_nbr = st.number_input(
        "Store Number", min_value=1, max_value=54, value=1, step=1,
        help="Store identifier (1–54)"
    )
    city = st.selectbox("City", CITIES, index=0)
    state = st.selectbox("State", STATES, index=0)
    store_type = st.selectbox("Store Type", STORE_TYPES, index=3, help="Store category A–E")
    cluster = st.number_input(
        "Store Cluster", min_value=1, max_value=17, value=13, step=1,
        help="Cluster grouping (1–17)"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📦 Product & Promotion</div>", unsafe_allow_html=True)

    family = st.selectbox("Product Family", FAMILIES, index=FAMILIES.index("BEVERAGES"))
    onpromotion = st.number_input(
        "Items on Promotion", min_value=0, value=0, step=1,
        help="Number of items currently on promotion"
    )

with right_col:
    st.markdown("<div class='section-title'>📅 Date & Context</div>", unsafe_allow_html=True)

    holiday_type = st.selectbox(
        "Holiday Type", HOLIDAY_TYPES, index=0,
        help="Type of holiday for the forecast date"
    )
    transactions = st.number_input(
        "Transactions", min_value=0.0, value=1500.0, step=50.0,
        help="Expected number of customer transactions"
    )
    dcoilwtico = st.number_input(
        "Oil Price (WTI, USD/bbl)", min_value=0.0, value=65.0, step=0.5,
        help="Daily WTI oil price — affects Ecuador's economy & sales"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>ℹ️ About the Inputs</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card' style='font-size:0.85rem;color:#9ca3af;'>"
        "<b style='color:#d1d5db;'>Store Number</b>: Favorita operates 54 stores across Ecuador.<br><br>"
        "<b style='color:#d1d5db;'>Product Family</b>: 31 product categories ranging from Beverages to Seafood.<br><br>"
        "<b style='color:#d1d5db;'>Oil Price</b>: Ecuador's economy is oil-dependent; price impacts consumer spending.<br><br>"
        "<b style='color:#d1d5db;'>Transactions</b>: Higher foot-traffic correlates tightly with higher sales."
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔮  Predict Sales")

# ─────────────────────────────────────────────
# Prediction output
# ─────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([{
        "store_nbr":    int(store_nbr),
        "family":       family,
        "onpromotion":  int(onpromotion),
        "city":         city,
        "state":        state,
        "type":         store_type,
        "cluster":      int(cluster),
        "transactions": float(transactions),
        "dcoilwtico":   float(dcoilwtico),
        "holiday_type": holiday_type,
    }])

    try:
        prediction = model.predict(input_df)[0]
        prediction = max(0.0, prediction)   # sales can't be negative

        res_left, res_center, res_right = st.columns([1, 2, 1])
        with res_center:
            st.markdown(
                f"<div class='prediction-box'>"
                f"<div class='label'>Predicted Sales</div>"
                f"<div class='value'>{prediction:,.2f}</div>"
                f"<div class='unit'>units sold</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Quick interpretation
        if prediction < 50:
            emoji, note = "🟡", "Low expected sales — likely a niche product or quiet period."
        elif prediction < 300:
            emoji, note = "🟢", "Moderate sales — typical daily volume for this configuration."
        elif prediction < 1000:
            emoji, note = "🔵", "Strong sales — high-demand product or busy store."
        else:
            emoji, note = "🟣", "Very high sales — top-performing combination!"

        st.markdown(
            f"<div class='card' style='text-align:center;'>"
            f"{emoji} &nbsp; <span style='color:#d1d5db;'>{note}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Show the input summary
        with st.expander("📋 Input Summary"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
