import streamlit as st

st.set_page_config(page_title="Falsifi.AI", layout="wide")


# -------------------------------
# Inject Global Dark Theme + Custom Text Selection
# -------------------------------
st.markdown("""
<style>
/* Global dark background and white text */
body {
    background-color: black !important;
    color: white !important;
}

/* All text elements white (except h1 which is overridden by class) */
h2, h3, h4, h5, h6, p, li, a, label {
    color: white;
}

/* Custom H1 color override */
.custom-h1 {
    color: #374151 !important;     /* Dark gray */
    font-weight: 800 !important;   /* Bold */
    font-size: 3.5rem !important;  /* Larger font size */
}


/* Custom text selection style */
::selection {
    background: white;
    color: black;
}
::-moz-selection {
    background: white;
    color: black;
}

/* Clean block spacing */
.block-container {
    padding: 2rem 1rem;
}

/* Module boxes */
.module-box {
    background-color: #111;
    padding: 1.5rem;
    border: 1px solid #444;
    border-radius: 10px;
    margin-bottom: 1rem;
    height: 100%;
}
.module-box a {
    color: white !important;
    font-weight: bold;
    text-decoration: none;
}

.custom-h1 {
    color: #4B5563 !important;     
    font-weight: 800 !important;   
    font-size: 3.75rem !important;  
    margin-bottom: 0rem !important;
    margin-top: 0rem !important;
}

.custom-subtitle {
    font-size: 1.25rem !important;
    color: white !important;
    margin-top: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section 
# -------------------------------
st.markdown("""
<h1 class="custom-h1">Falsifi.AI</h1>
<p class="custom-subtitle">Deep Learning-Based Signature Verification</p>
""", unsafe_allow_html=True)

# -------------------------------
# Introduction Box
# -------------------------------
st.markdown("""
<div class="module-box">
    <h3>What is Falsifi.AI?</h3>
    <p>Falsifi.AI is a deep-learning powered toolkit for detecting forged handwritten signatures. It explores around 30 different combinations of preprocessing techniques and model architectures to maximize accuracy.</p>
    <p>Built with Streamlit, HTML, and CSS. Hosted for free on Streamlit Cloud.</p>
    <p>Made for banking, legal, academic, and digital identity verification.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Workflow Section
# -------------------------------
st.markdown("""
<div class="module-box">
    <h4>How does it work?</h4>
    <ul>
        <li>Upload a confirmed (anchor) signature image</li>
        <li>Upload a signature image to be verified</li>
        <li>Select a pipeline/model from the dropdown menu to try different combinations</li>
    </ul>
    <p>Output includes authenticity label (Genuine or Forged), confidence score, and visual explanation using GradCAM or SHAP.</p>
    <p><b>Live Demo:</b> <a href="https://falsifi-ai.streamlit.app/" target="_blank">https://falsifi-ai.streamlit.app/</a></p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Dataset Section
# -------------------------------
st.markdown("""
<div class="module-box">
    <h4>Dataset Used</h4>
    <ul>
        <li><b>Dataset:</b> CEDAR Signature Dataset</li>
        <li><b>Split:</b> 80% Training, 10% Validation, 10% Testing</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div style="background-color: #111827; padding: 20px 0; max-width: 100%; margin-bottom: 0;">
    <div style="text-align: center; color: gray;">
        © 2025 Shail K Patel · Crafted out of boredom.
    </div>
    <div style="text-align: center; color: gray;">
        <a href="https://github.com/ShailKPatel/Falsifi.AI/" style="color: gray; text-decoration: none;">GitHub Repo</a> · MIT License
    </div>
</div>
""", unsafe_allow_html=True)
