import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("Scaler (1).pkl")

# Page config
st.set_page_config(page_title="📊 Ad Budget Predictor", page_icon="📈", layout="wide")

# Advanced CSS (Premium UI)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141e30, #243b55);
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    color: #cfd8dc;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

.metric-card {
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: #00ffcc;
    font-size: 26px;
    font-weight: bold;
}

.stSlider > div > div {
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00c6ff;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>📊 Advertising Sales Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict how your marketing budget impacts sales 🚀</div>", unsafe_allow_html=True)

st.markdown("---")

# Input Section
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    tv = st.slider("📺 TV Ad Budget ($)", 0.0, 300.0, 50.0)

with col2:
    radio = st.slider("📻 Radio Ad Budget ($)", 0.0, 50.0, 10.0)

with col3:
    newspaper = st.slider("📰 Newspaper Ad Budget ($)", 0.0, 120.0, 20.0)

st.markdown("</div>", unsafe_allow_html=True)

# Show budget summary
st.markdown("\n")
col4, col5, col6 = st.columns(3)

col4.metric("TV Budget", f"${tv}")
col5.metric("Radio Budget", f"${radio}")
col6.metric("Newspaper Budget", f"${newspaper}")

st.markdown("---")

# Prediction
if st.button("🔮 Predict Sales"):
    try:
        data = np.array([[tv, radio, newspaper]])
        scaled = scaler.transform(data)
        prediction = model.predict(scaled)[0]

        st.markdown(f"""
        <div class='metric-card'>
            💰 Predicted Sales: {prediction:.2f}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# Insights section
st.markdown("---")
st.markdown("### 📌 Insights")
st.info("👉 TV and Radio typically have higher impact on sales compared to Newspaper.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)