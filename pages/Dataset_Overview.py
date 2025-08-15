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
<h1 class="custom-h1">Dataset Overview</h1>
""", unsafe_allow_html=True)

# -------------------------------
# Signature Dataset Usage Section
# -------------------------------
st.markdown("""
#### Signature Dataset Usage

For signature forgery detection, the **CEDAR** dataset is used for training, validation, and testing.
The data is split by signer into 80% for training, 10% for validation, and 10% for testing. This ensures that evaluation is done on unseen signers, helping to measure true generalization performance.

- **Dataset:** CEDAR Signature Dataset
- **Split:** 80% Training, 10% Validation, 10% Testing
- **Kaggle Link:** [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
""")



# -------------------------------
# Dataset Preparation Section
# -------------------------------
st.markdown("""
#### Dataset Preparation

1. **Dataset Structure**  
   CEDAR signatures are stored in two folders:  
   - `original/original_{signer_id}_{signature_no}.png`  
   - `forgeries/forgeries_{signer_id}_{signature_no}.png`  
   where signer_id ranges from 1–55 and signature_no from 1–24.

2. **Usage**  
   The CEDAR dataset is used for training, validation, and testing, with an 80-10-10 split based on signer IDs. This approach evaluates the model's ability to generalize to unseen signers.


3. **Siamese Network Dataset**   
   A CSV file is generated containing pairs of signatures for training the Siamese network.

   **Columns**:  
   - `signer_id` → The signer’s ID (1–55)  
   - `path1` → Anchor signature (always genuine)  
   - `path2` → Another signature (genuine or forged)  
   - `authentic` → 1 for genuine-genuine pairs, 0 for genuine-forgery pairs  

   **Counts**:  
   - Real pairs: 30,360  
   - Fake pairs: 31,680  
   - Total: 62,040   
   


4. **Triplet Network Dataset**   
   A CSV file is generated containing triplets for triplet loss training.

   **Columns**:  
   - `signer_id` → The signer’s ID (1–55)
   - `anchor` → Genuine signature (reference)
   - `positive` → Another genuine signature from the same signer
   - `negative` → A forged signature of the same signer

   **Counts**:  
    Total triplets: 728,640 (all possible anchor-positive-negative combinations)

- Anchor and positive belong to the same signer but are different images, while the negative is a forgery of that signer.
---

### Final Dataset Splits

The dataset is split by signer ID:
- **Training Set:** 80% of signers
- **Validation Set:** 10% of signers
- **Test Set:** 10% of signers

---

This setup ensures:
- No overlap in signers between the training, validation, and test sets.
- Evaluation is performed on completely unseen signers to measure generalization.
""")

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
