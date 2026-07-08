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

.stApp{
background: linear-gradient(135deg,#dbeafe,#ffffff);
}

.main{
padding-top:20px;
}

h1{
text-align:center;
color:#0F172A !important;
font-size:45px;
font-weight:700;
}

h3{
color:#1E3A8A;
}

p,li{
color:#111827 !important;
font-size:18px;
}

[data-testid="stFileUploader"]{
background:white;
padding:20px;
border-radius:15px;
border:2px solid #3B82F6;
}

.stButton>button{
width:100%;
background:#2563EB;
color:white;
font-size:20px;
font-weight:bold;
border-radius:12px;
padding:12px;
border:none;
}

.stButton>button:hover{
background:#1D4ED8;
color:white;
}

.success-box{
background:#DCFCE7;
padding:15px;
border-radius:10px;
font-size:22px;
font-weight:bold;
color:#166534;
text-align:center;
}

.error-box{
background:#FEE2E2;
padding:15px;
border-radius:10px;
font-size:22px;
font-weight:bold;
color:#B91C1C;
text-align:center;
}

.footer{
text-align:center;
font-size:15px;
color:#374151;
padding-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# -------------------- Title --------------------

st.markdown(
"<h1>🩺 COVID-19 Detection From Chest X-Ray</h1>",
unsafe_allow_html=True
)

st.write("### Upload a chest X-ray image to predict whether it is Normal or COVID-19.")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,width=350)

    img=image.resize((299,299))
    img=np.array(img)/255.0
    img=np.expand_dims(img,0)

    if st.button("Predict"):

        prediction=model.predict(img)

        score=float(prediction[0][0])

        if score>0.5:
            confidence=score*100

            st.markdown(
            f"<div class='error-box'>🔴 COVID-19 DETECTED<br><br>Confidence : {confidence:.2f}%</div>",
            unsafe_allow_html=True)

        else:
            confidence=(1-score)*100

            st.markdown(
            f"<div class='success-box'>🟢 NORMAL<br><br>Confidence : {confidence:.2f}%</div>",
            unsafe_allow_html=True)

        st.progress(confidence/100)

st.markdown("<hr>",unsafe_allow_html=True)

st.markdown(
"<div class='footer'>Developed using TensorFlow & Streamlit</div>",
unsafe_allow_html=True
)
