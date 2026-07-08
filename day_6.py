import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)

# -------------------- CSS --------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 40%, #ffffff 100%);
    background-attachment: fixed;
}

.main{
    padding-top: 10px;
}

/* ---------- Header Card ---------- */
.header-card{
    background: linear-gradient(135deg, #1E3A8A, #2563EB, #3B82F6);
    padding: 35px 25px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(37,99,235,0.35);
    margin-bottom: 25px;
    text-align: center;
}

.header-card h1{
    color: white !important;
    font-size: 40px;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

.header-card p{
    color: #DBEAFE !important;
    font-size: 17px !important;
    margin-top: 8px;
    font-weight: 400;
}

/* ---------- Section subtitle ---------- */
h3{
    color:#1E3A8A;
    font-weight: 700;
    text-align: center;
}

p, li{
    color:#111827 !important;
    font-size:17px;
}

/* ---------- Upload box ---------- */
[data-testid="stFileUploader"]{
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 2px dashed #3B82F6;
    box-shadow: 0 6px 18px rgba(59,130,246,0.15);
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover{
    border-color: #1D4ED8;
    box-shadow: 0 8px 24px rgba(59,130,246,0.25);
}

/* ---------- Image preview card ---------- */
[data-testid="stImage"]{
    display: flex;
    justify-content: center;
    margin: 20px 0;
}

[data-testid="stImage"] img{
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    border: 4px solid white;
}

/* ---------- Button ---------- */
.stButton>button{
    width:100%;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color:white;
    font-size:20px;
    font-weight:700;
    border-radius:14px;
    padding:14px;
    border:none;
    box-shadow: 0 6px 16px rgba(37,99,235,0.4);
    transition: all 0.25s ease;
    letter-spacing: 0.5px;
}

.stButton>button:hover{
    background: linear-gradient(135deg, #1D4ED8, #1E3A8A);
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(29,78,216,0.5);
    color:white;
}

.stButton>button:active{
    transform: translateY(0px);
}

/* ---------- Result boxes ---------- */
.success-box{
    background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
    padding:22px;
    border-radius:16px;
    font-size:24px;
    font-weight:800;
    color:#166534;
    text-align:center;
    box-shadow: 0 8px 20px rgba(22,101,52,0.15);
    border: 2px solid #86EFAC;
    margin-top: 15px;
    animation: fadeIn 0.5s ease;
}

.error-box{
    background: linear-gradient(135deg, #FEE2E2, #FECACA);
    padding:22px;
    border-radius:16px;
    font-size:24px;
    font-weight:800;
    color:#B91C1C;
    text-align:center;
    box-shadow: 0 8px 20px rgba(185,28,28,0.15);
    border: 2px solid #FCA5A5;
    margin-top: 15px;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn{
    from{ opacity: 0; transform: translateY(8px); }
    to{ opacity: 1; transform: translateY(0px); }
}

/* ---------- Progress bar ---------- */
.stProgress > div > div{
    background: linear-gradient(90deg, #3B82F6, #1D4ED8);
    border-radius: 10px;
}

.stProgress{
    margin-top: 10px;
}

/* ---------- Disclaimer ---------- */
.disclaimer{
    background: #FEF9C3;
    border-left: 5px solid #EAB308;
    padding: 14px 18px;
    border-radius: 10px;
    font-size: 14px !important;
    color: #713F12 !important;
    margin-top: 20px;
}

/* ---------- Footer ---------- */
.footer{
    text-align:center;
    font-size:14px;
    color:#374151;
    padding-top:25px;
    padding-bottom: 10px;
}

hr{
    margin-top: 30px;
    border: none;
    border-top: 1px solid #CBD5E1;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# -------------------- Header --------------------
st.markdown("""
<div class="header-card">
    <h1>🩺 COVID-19 Detection</h1>
    <p>AI-powered analysis of chest X-ray images to help identify potential COVID-19 indicators</p>
</div>
""", unsafe_allow_html=True)

st.write("### 📤 Upload a Chest X-Ray Image")

uploaded_file = st.file_uploader(
    "Choose an Image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=350)

    img = image.resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing X-ray..."):
            prediction = model.predict(img)
            score = float(prediction[0][0])

        if score > 0.5:
            confidence = score * 100
            st.markdown(
                f"<div class='error-box'>🔴 COVID-19 DETECTED<br><br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            confidence = (1 - score) * 100
            st.markdown(
                f"<div class='success-box'>🟢 NORMAL<br><br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.progress(confidence / 100)

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This tool is for educational/demo purposes only and is not a substitute for professional medical diagnosis. Please consult a certified radiologist or physician for accurate diagnosis.
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>Developed with ❤️ using TensorFlow & Streamlit</div>",
    unsafe_allow_html=True
)
