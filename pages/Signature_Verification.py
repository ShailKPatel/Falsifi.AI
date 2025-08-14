import streamlit as st

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
<h1 class="custom-h1">Signature Verification</h1>
""", unsafe_allow_html=True)

st.markdown("""
### Project Understanding

This page allows you to verify the authenticity of a signature by comparing it against an anchor image. 

1.  **Upload Anchor Image**: This is the known, genuine signature.
2.  **Upload Image to be Verified**: This is the signature you want to check.
3.  **Select a Model**: Choose the model you want to use for verification.
4.  **Submit**: Click the submit button to perform the verification (currently disabled).
""")

# -------------------------------
# Image Upload Section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.header("Anchor Image")
    anchor_image = st.file_uploader("Upload an anchor image", type=['png', 'jpg', 'jpeg'])

with col2:
    st.header("Image to be Verified")
    verify_image = st.file_uploader("Upload an image to be verified", type=['png', 'jpg', 'jpeg'])

# -------------------------------
# Model Selection and Submission
# -------------------------------
st.header("Model Selection")
model_list = ["Model 1", "Model 2", "Model 3"]
selected_model = st.selectbox("Select a model", model_list)

st.write(f"You selected: {selected_model}")

if st.button("Submit"):
    pass

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