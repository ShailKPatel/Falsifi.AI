#  Falsifi.AI  – Deep Learning-Based Signature Verification

**Falsifi.AI** is an AI system designed to detect signature forgery using deep learning. It leverages multiple pipelines with varied preprocessing and model architectures to verify the authenticity of scanned signatures. Built with PyTorch and Streamlit, Falsifi.AI serves as a tool for applications such as identity verification, banking, legal documents, and academic records.

---

## Capabilities

- Classifies scanned signatures as genuine or forged using handwriting verification techniques.
- Experiments with around 18 different pipelines, each combining unique preprocessing steps and model architectures to maximize accuracy and robustness.

---

## Dataset Details

- **Dataset:** CEDAR Signature Dataset
- **Split:** 80% Training, 10% Validation, 10% Testing
- **Kaggle Link:** [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)

The dataset is split by signer, ensuring that the model is evaluated on unseen identities to simulate real-world scenarios and improve generalization.

---

## Sample Use Workflow

Upload two images:
- A confirmed (anchor) signature image
- A signature image to be verified
- Select a pipeline/model from the dropdown menu to try different combinations

Output includes:
- Authenticity label (Genuine or Forged)
- Confidence score
- Visual explanation using GradCAM or SHAP

---

## Tech Stack

| Component        | Technology                  |
|------------------|-----------------------------|
| Backend          | Python                      |
| Models           | PyTorch, CNNs, EfficientNet |
| Frontend         | Streamlit, HTML, CSS        |
| Explainability   | GradCAM, SHAP               |
| Deployment       | Streamlit Cloud (free hosting) |

---

## Project Structure

```
 Falsifi.AI /
 to do
```

---

## Running the App Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ShailKPatel/Falsifi.AI.git
   cd Falsifi.AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit server:
   ```bash
   streamlit run main.py
   ```

4. Open your browser and go to:
   ```
   http://localhost:8501
   ```

---

## Live Demo

A live demo is available at:
```
https://falsifi-ai.streamlit.app/
```

---

## Datasets Used

| Purpose                | Dataset                                     |
|------------------------|---------------------------------------------|
| Signature Verification | CEDAR (80-10-10 split for train-val-test)   |

---

## License and Acknowledgments

- Open-sourced under the MIT License
- Built for machine learning research and portfolio demonstration
- Credits to the academic datasets and open-source communities

---

## Author

Created by [Shail Patel](https://www.linkedin.com/in/shailkpatel/)