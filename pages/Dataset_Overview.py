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

For signature forgery detection, **GPDS 1-150** is used for training and validation (split 80-20 by signer ID), while **CEDAR** is reserved exclusively for testing.  
This ensures that evaluation is done on a completely unseen dataset from a different source, helping to measure true generalization performance.

- GPDS 1-150 (train/val, 80-20 split): [Kaggle link](https://www.kaggle.com/datasets/adeelajmal/gpds-1150)
- CEDAR (test): [Kaggle link](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
""")

# -------------------------------
# GPDS Dataset Preparation Section
# -------------------------------
st.markdown("""
#### GPDS Dataset Preparation

1. **Dataset Reorganization**  
   GPDS was reorganized into a consistent folder structure:  
   - `gpds1-150/original/original_{signer_id}_{signature_no}`  
   - `gpds1-150/forgeries/forgeries_{signer_id}_{signature_no}`  
   where signer_id ranges from 001–150 and signature_no from 01–24.

2. **Merging Train and Test Splits**  
   The original split by signature_no could cause data leakage.  
   All samples were merged and then split by **signer_id** to prevent leakage.

3. **Cleaning and Reducing Forgeries**  
   For each signer, only signature_no 01–24 were kept for both genuine and forged signatures.

4. **Initial Pair Counts (Before Balancing)**  
   - Real pairs: 82,800  
   - Fake pairs: 86,400  
   - Total: 169,200

5. **Balancing the Dataset**  
   Randomly removed fake pairs to match real pairs:  
   - Final: 82,800 real pairs, 82,800 fake pairs  
   - Total: 165,600

---

### Siamese Network Dataset

**`gpds_siamese.csv`** contains all possible balanced pairs of signatures.

**Columns**:  
- `signer_id` → The signer’s ID (001–150)  
- `path1` → Anchor signature (always genuine)  
- `path2` → Another signature (genuine or forged)  
- `authentic` → 1 for genuine-genuine pairs, 0 for genuine-forgery pairs  

**Counts**:  
- Total pairs: 165,600 (anchor-positive/negative combinations)  

Dataset is split **by signer_id** into train/val.

---

### Triplet Loss Dataset

**`gpds_triplet.csv`** contains triplets for triplet loss training.  

**Columns**:  
- `signer_id` → The signer’s ID (001–150)  
- `anchor` → Genuine signature (reference)  
- `positive` → Another genuine signature from the same signer  
- `negative` → A forged signature of the same signer  

**Counts**:  
- Total triplets: 1,987,200 (all possible anchor-positive-negative combinations)  

Anchor and positive belong to the same signer but are different images, while the negative is a forgery of that signer.
""")

# -------------------------------
# CEDAR Dataset Preparation Section
# -------------------------------
st.markdown("""
#### CEDAR Dataset Preparation

1. **Dataset Structure**  
   CEDAR signatures are stored in two folders:  
   - `original/original_{signer_id}_{signature_no}.png`  
   - `forgeries/forgeries_{signer_id}_{signature_no}.png`  
   where signer_id ranges from 1–55 and signature_no from 1–24.

2. **Usage**  
   CEDAR is **only used for testing** to evaluate the model’s generalization on unseen signers from a different dataset source.

3. **Siamese Network Test Set**

   **Columns**:  
   - `signer_id` → The signer’s ID (1–55)  
   - `path1` → Anchor signature (always genuine)  
   - `path2` → Another signature (genuine or forged)  
   - `authentic` → 1 for genuine-genuine pairs, 0 for genuine-forgery pairs  

   **Counts**:  
   - Real pairs: 30,360  
   - Fake pairs: 31,680  
   - Total: 62,040  
   Saved as **`cedar_test.csv`**.

---

### Final Dataset Splits

**Siamese Dataset**:  
- **Train**: 132,463 samples  
- **Val**: 33,137 samples  
- **Test** (CEDAR): 62,040 samples  
- Unique signers in train: 120  
- Unique signers in val: 30  

**Triplet Dataset**:  
- **Train**: 1,589,760 samples  
- **Val**: 397,440 samples  
- Test set for triplet loss can also be generated from CEDAR if needed (structure matches GPDS triplets).  

---

This setup ensures:  
- No overlap in signers between train and val.  
- Evaluation is performed on a completely different dataset (CEDAR) to measure cross-dataset performance.
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
