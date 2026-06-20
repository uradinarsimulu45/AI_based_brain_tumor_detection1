
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.main-title{
    text-align:center;
    font-size:55px;
    font-weight:700;
    color:white;
    margin-bottom:0px;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:30px;
}

.card{
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);
    text-align:center;
}

.metric{
    font-size:28px;
    font-weight:bold;
    color:#38bdf8;
}

.metric-label{
    color:white;
}

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:40px;
}

.result-success{
    background:#14532d;
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
}

.result-danger{
    background:#7f1d1d;
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class='main-title'>
🧠 Brain Tumor Detection System
</div>
<div class='subtitle'>
Deep Learning Based MRI Analysis
</div>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD CARDS ----------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='card'>
        <div class='metric'>CNN</div>
        <div class='metric-label'>Model</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='card'>
        <div class='metric'>224×224</div>
        <div class='metric-label'>Input Size</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='card'>
        <div class='metric'>AI Powered</div>
        <div class='metric-label'>Detection</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

# ---------------- MAIN SECTION ----------------

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader(
        "📤 Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

with right:
    st.info("""
    ### Instructions

    ✔ Upload MRI Scan

    ✔ AI analyzes image

    ✔ View prediction result

    ✔ Check confidence score
    """)

# ---------------- PREDICTION ----------------

if uploaded:

    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(
            img,
            caption="Uploaded MRI Scan",
            use_container_width=True
        )

    img_resize = img.resize((224,224))
    img_array = image.img_to_array(img_resize)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("Analyzing MRI Scan..."):
        prediction = model.predict(img_array)[0][0]

    confidence = prediction if prediction > 0.5 else 1-prediction

    st.write("")
    st.progress(float(confidence))

    if prediction > 0.5:

        st.markdown(f"""
        <div class='result-danger'>
            <h1>🚨 Tumor Detected</h1>
            <h2>Confidence: {confidence*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class='result-success'>
            <h1>✅ No Tumor Detected</h1>
            <h2>Confidence: {confidence*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🧠 Brain Tumor AI")

    st.markdown("---")

    st.write("""
    This application uses a trained
    deep learning model to analyze
    MRI scans and predict whether
    a brain tumor is present.
    """)

    st.warning(
        "Educational Use Only.\nNot a medical diagnosis."
    )

# ---------------- FOOTER ----------------

st.markdown("""
<div class='footer'>
Developed with Streamlit • TensorFlow • Deep Learning
</div>
""", unsafe_allow_html=True)
