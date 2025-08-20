# Intelligent Question Generator (IQG)

Generate MCQs, fill-in-the-blanks, True/False, and short-answer questions from any study text.

## Quickstart

```bash
# 1) Create & activate venv (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) One-time model data
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4) Try the CLI
python main.py --in data/sample.txt --out questions.json --num-qs 12

# 5) Launch the Streamlit app
streamlit run app.py
