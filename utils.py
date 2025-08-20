import os
import re
import json
import random
from typing import List, Dict, Tuple

import spacy
import textstat
from dotenv import load_dotenv

# NLTK WordNet (optional)
try:
    from nltk.corpus import wordnet as wn
    _WORDNET_OK = True
except Exception:
    _WORDNET_OK = False

# Groq
from groq import Groq

# -------------------------
# Config & Setup
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY in .env file!")

client = Groq(api_key=GROQ_API_KEY)
QG_MODEL_NAME = "llama3-70b-8192"

random.seed(42)

_NLP = None

def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_md")
        except OSError as e:
            raise OSError(
                "spaCy model 'en_core_web_md' not found. Install it with:\n"
                "    python -m spacy download en_core_web_md"
            ) from e
    return _NLP

# -------------------------
# Helpers
# -------------------------
STOPLIKE = set([
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "from", "with",
    "as", "by", "is", "was", "were", "be", "been", "are", "that", "this", "these",
    "those", "it", "its", "at", "into", "than", "then"
])

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def split_sentences(text: str, max_sents: int = 10) -> List[str]:
    nlp = get_nlp()
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sents[:max_sents]

def estimate_difficulty(sent: str) -> str:
    try:
        score = textstat.flesch_reading_ease(sent)
    except Exception:
        score = 50
    if score >= 60:
        return "Easy"
    elif score >= 35:
        return "Medium"
    return "Hard"

def bloom_level(sent: str) -> str:
    s = sent.lower()
    if any(k in s for k in ["define", "identify", "list"]):
        return "Remember (Knowledge)"
    if any(k in s for k in ["explain", "summarize", "describe", "interpret"]):
        return "Understand (Comprehension)"
    if any(k in s for k in ["apply", "use", "calculate", "solve", "demonstrate"]):
        return "Apply (Application)"
    if any(k in s for k in ["compare", "contrast", "analyze", "impact", "effect", "examine"]):
        return "Analyze"
    if any(k in s for k in ["evaluate", "assess", "justify", "argue", "critique"]):
        return "Evaluate"
    if any(k in s for k in ["design", "create", "propose", "develop", "formulate"]):
        return "Create"
    return "Understand (Comprehension)"

# -------------------------
# Question Generation
# -------------------------
def qg_from_sentence(sent: str, num_return: int = 3) -> List[str]:
    prompt = f"""
You are an expert educational question generator.

Task: Generate exactly {num_return} diverse, exam-ready questions from the sentence below.
- Use complete question stems.
- DO NOT include answers.
- Return as a simple numbered list, each line ending with a question mark.
- No preamble or extra text.

Sentence:
\"\"\"{sent}\"\"\"
"""
    response = client.chat.completions.create(
        model=QG_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You generate concise, high-quality exam questions."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.6,
    )

    text = response.choices[0].message.content.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    qs = []
    for ln in lines:
        ln = re.sub(r"^\s*(?:\d+[\.\)]|-)\s*", "", ln)
        m = re.search(r"(.+?\?)", ln)
        if m:
            qs.append(m.group(1).strip())
    return qs[:num_return]

# -------------------------
# Answer Extraction
# -------------------------
def extract_candidate_answer(sent: str) -> str:
    nlp = get_nlp()
    doc = nlp(sent)

    # 1) Named entities
    ents = [e for e in doc.ents if 1 <= len(e.text.split()) <= 4]
    if ents:
        return ents[0].text

    # 2) Proper nouns
    propns = [t.text for t in doc if t.pos_ == "PROPN" and 1 <= len(t.text.split()) <= 3]
    if propns:
        return propns[0]

    # 3) Short noun chunks
    chunks = [c.text for c in doc.noun_chunks if 1 <= len(c.text.split()) <= 3 and c.text.lower() not in STOPLIKE]
    if chunks:
        return chunks[0]

    # 4) Token fallback
    toks = [t.text for t in doc if t.is_alpha and not t.is_stop and t.lemma_.lower() not in STOPLIKE]
    toks = sorted(toks, key=len, reverse=True)
    return toks[0] if toks else ""

# -------------------------
# Distractors
# -------------------------
def _valid_option(opt: str, answer: str) -> bool:
    if not opt:
        return False
    o = opt.strip()
    if o.lower() in STOPLIKE:
        return False
    if o.lower() == answer.strip().lower():
        return False
    if len(o) < 3:
        return False
    return True

def wordnet_distractors(answer: str, topn: int = 3) -> List[str]:
    if not _WORDNET_OK or not answer:
        return []
    try:
        synsets = wn.synsets(answer.lower().strip())
        distractors = set()
        for syn in synsets[:2]:
            for hyper in syn.hypernyms():
                for hyponym in hyper.hyponyms():
                    for lemma in hyponym.lemma_names():
                        cand = lemma.replace("_", " ")
                        if cand.lower() != answer.lower():
                            distractors.add(cand)
        out = [d for d in sorted(distractors, key=lambda x: abs(len(x) - len(answer))) if _valid_option(d, answer)]
        return out[:topn]
    except Exception:
        return []

def semantic_distractors_from_text(answer: str, sent: str, topn: int = 3) -> List[str]:
    nlp = get_nlp()
    doc = nlp(sent)
    a_doc = nlp(answer)
    cands = set()
    for e in doc.ents:
        if e.text.lower() != answer.lower() and _valid_option(e.text, answer):
            cands.add(e.text)
    for c in doc.noun_chunks:
        txt = c.text.strip()
        if txt.lower() != answer.lower() and 1 <= len(txt.split()) <= 3 and _valid_option(txt, answer):
            cands.add(txt)
    scored = []
    for c in cands:
        try:
            score = a_doc.similarity(nlp(c))
        except Exception:
            score = 0.0
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:topn]]

def assemble_distractors(answer: str, sent: str, needed: int = 3) -> List[str]:
    pool = []
    pool.extend(wordnet_distractors(answer, topn=needed))
    remain = needed - len(pool)
    if remain > 0:
        pool.extend(semantic_distractors_from_text(answer, sent, topn=remain + 3))
    # Clean and truncate
    clean = []
    seen = set()
    for p in pool:
        if _valid_option(p, answer):
            k = p.lower()
            if k not in seen:
                clean.append(p)
                seen.add(k)
        if len(clean) >= needed:
            break
    return clean[:needed]

# -------------------------
# Builders
# -------------------------
def ensure_period(s: str) -> str:
    s = s.strip()
    if not s.endswith(('.', '?')):
        s += '.'
    return s

def build_fill_blank(sent: str) -> Dict:
    ans = extract_candidate_answer(sent)
    if not ans:
        return {}
    pattern = re.compile(re.escape(ans), flags=re.IGNORECASE)
    blanked = pattern.sub("_____", sent, count=1)
    if blanked == sent:
        return {}
    return {
        "type": "fill_blank",
        "question": ensure_period(blanked),
        "answer": ans,
        "difficulty": estimate_difficulty(sent),
        "bloom": bloom_level(sent),
        "source": sent,
    }

def build_true_false(sent: str) -> Dict:
    make_true = random.choice([True, False])
    statement = sent.strip()
    if not statement.endswith("."):
        statement += "."
    if not make_true:
        # Generic numeric or keyword perturbation
        m = re.search(r"(\d+)", statement)
        if m:
            val = int(m.group(1))
            statement = statement.replace(str(val), str(val + random.choice([-1, +1, +10, -10])))
        else:
            statement = "It is not correct that " + statement
    return {
        "type": "true_false",
        "statement": ensure_period(statement),
        "answer": "True" if make_true else "False",
        "difficulty": estimate_difficulty(sent),
        "bloom": "Remember (Knowledge)",
        "source": sent,
    }

def build_mcq(qs: List[str], sent: str) -> Dict:
    if not qs:
        return {}
    stem = random.choice(qs)
    ans = extract_candidate_answer(sent)
    if not ans or len(ans.split()) > 5:
        return {}
    distractors = assemble_distractors(ans, sent, needed=3)
    if len(distractors) < 3:
        # Generic backup distractors
        backups = ["component", "process", "factor", "structure"]
        for b in backups:
            if len(distractors) >= 3:
                break
            if _valid_option(b, ans):
                distractors.append(b)
    options = distractors + [ans]
    random.shuffle(options)
    answer_index = options.index(ans)
    return {
        "type": "mcq",
        "question": stem,
        "options": options,
        "answer_index": answer_index,
        "answer": ans,
        "difficulty": estimate_difficulty(sent),
        "bloom": bloom_level(sent),
        "source": sent,
    }

def build_short_answer(qs: List[str], sent: str) -> Dict:
    if not qs:
        return {}
    
    ans = extract_candidate_answer(sent) or sent
    stem = random.choice(qs)
    
    return {
        "type": "short_answer",
        "question": stem,
        "answer": ans,   # <- FIX: use "answer" instead of "sample_answer"
        "difficulty": estimate_difficulty(sent),
        "bloom": bloom_level(sent),
        "source": sent,
    }



# -------------------------
# Orchestrator
# -------------------------
def generate_from_text(
    text: str,
    max_sents: int = 8,
    num_q_per_sent: int = 3,
    include_types: Tuple[str, ...] = ("mcq", "fill_blank", "true_false"),
) -> List[Dict]:
    text = clean_text(text)
    sentences = split_sentences(text, max_sents=max_sents)
    out: List[Dict] = []
    for s in sentences:
        qs = qg_from_sentence(s, num_return=num_q_per_sent)
        if "mcq" in include_types:
            mcq = build_mcq(qs, s)
            if mcq:
                out.append(mcq)
        if "fill_blank" in include_types:
            fb = build_fill_blank(s)
            if fb:
                out.append(fb)
        if "true_false" in include_types:
            tf = build_true_false(s)
            if tf:
                out.append(tf)
        if "short_answer" in include_types:
            sa = build_short_answer(qs, s)
            if sa:
                out.append(sa)
    # Deduplicate
    seen = set()
    deduped = []
    for q in out:
        key = json.dumps({"type": q.get("type"), "text": q.get("question") or q.get("statement")}, sort_keys=True)
        if key not in seen:
            deduped.append(q)
            seen.add(key)
    return deduped
