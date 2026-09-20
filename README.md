# Intent Classification & Named Entity Recognition (NER) Engine

An interactive Natural Language Understanding (NLU) pipeline that processes unstructured text to simultaneously extract user intent and key entities. This project demonstrates foundational machine learning engineering skills by leveraging state-of-the-art transformer models (BERT) in a localized, non-generative AI architecture.

## 🚀 Features

* **Joint Processing:** Analyzes text for both sequence-level classification (intent) and token-level classification (NER) in a single workflow.
* **Banking Domain Intelligence:** Utilizes a fine-tuned BERT model capable of categorizing queries into 77 distinct banking intents (e.g., `lost_or_stolen_card`, `top_up_failed`).
* **Visual Entity Extraction:** Highlights named entities (Locations, Organizations, Persons, Misc) inline using `spacy-streamlit`.
* **Confidence Thresholding:** Outputs the top 3 predicted intents and automatically flags low-confidence predictions (< 70%) for human-in-the-loop review.
* **Interactive Dashboard:** Wraps the complex ML backend in a clean, user-friendly Streamlit web interface.

## 🛠️ Technology Stack

* **Language:** Python
* **Machine Learning:** Hugging Face `transformers`, PyTorch
* **NLP Processing:** spaCy, token-classification, sequence-classification
* **Frontend UI:** Streamlit, `spacy-streamlit`

## ⚙️ Installation & Setup

Follow these steps to run the application locally on your machine.

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Create and activate a virtual environment**
* **Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **Mac/Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

**3. Install the dependencies**
```bash
pip install -r requirements.txt
pip install spacy spacy-streamlit
```
*(Note for Windows users: If you encounter an `ImportError` regarding Application Control policies when running the app, you may need to add your `.venv` folder to your Windows Security exclusions or enable Developer Mode.)*

**4. Run the application**
```bash
streamlit run app.py
```
The app will automatically open in your default web browser at `http://localhost:8501`.

## 🧠 How It Works

1. **Input:** The user types a natural language query (e.g., "I lost my credit card in Paris").
2. **Intent Pipeline:** The text is passed through `philschmid/BERT-Banking77`, a DistilBERT model that maps the sentence to mathematical coordinates and scores it against known banking categories.
3. **NER Pipeline:** The text is simultaneously passed through `dslim/bert-base-NER`, which analyzes the relationships between tokens to locate and classify entities (e.g., extracting "Paris" as a Location).
4. **Output:** Streamlit renders the probabilities and visually highlights the extracted entities.