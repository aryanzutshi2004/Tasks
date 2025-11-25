# streamlit_nlp_genai_roadmap.py
# Streamlit app: Interactive checklist & study tracker for NLP + Generative AI roadmap
# The app creates hierarchical checkboxes (main topics + subtopics). Main topic is marked done when all its subtopics are done.
# It saves progress locally to a JSON file and allows export/import.
# NOTE: The dev environment will transform the FILE_PATH into a proper URL when serving. Keep the local path as-is.

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path

# Path to uploaded image (provided by the environment). We'll keep this exact path so the platform can transform it.
FILE_PATH = "/mnt/data/8c81d458-5a40-4c0c-a2be-5f32557aecbb.png"

# State file to save progress
SAVE_PATH = Path(".nlp_genai_progress.json")

st.set_page_config(page_title="NLP + GenAI Roadmap Tracker", layout="wide")

st.title("NLP + Generative AI Roadmap — Interactive Checklist")
st.markdown("Use this page to mark subtopics as done. A main topic will be considered done only when all its subtopics are checked.")

# Utility functions

def load_progress():
    if SAVE_PATH.exists():
        try:
            with open(SAVE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(data):
    with open(SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def export_progress(data):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"nlp_genai_progress_{ts}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    return fname

# The master roadmap data structure
# Each main topic maps to a list of subtopics. Each subtopic is a dict with a short title, a "details" string to show inside an expander,
# and optionally a "weight" or importance field. We include "depth" suggestion as text: surface / medium / deep.

ROADMAP = {
    "Foundations": [
        {"id": "python_ml", "title": "Python for ML (numpy, pandas, OOP)", "depth": "medium",
         "details": "Practice: vectorized ops (numpy), dataframe manipulations (pandas), write small classes. Interview tasks: clean data pipelines."},
        {"id": "prob_stats", "title": "Probability & Basic Statistics", "depth": "medium",
         "details": "Mean, variance, distributions, Bayes rule, conditional probability — used in Naive Bayes and evaluation metrics."},
        {"id": "linear_algebra", "title": "Linear Algebra Essentials", "depth": "medium",
         "details": "Vectors, matrices, dot products, matrix multiplication, projections. Understand shapes: important for embeddings and attention math."},
    ],

    "Classical NLP": [
        {"id": "preprocessing", "title": "Text preprocessing (tokenization, stopwords, lemmatization)", "depth": "medium",
         "details": "Implement tokenizers, compare stemming vs lemmatization, practice with NLTK/spacy. Understand edgecases like contractions and unicode."},
        {"id": "vectorization", "title": "Bag-of-Words, TF-IDF, hashing", "depth": "medium",
         "details": "Implement TF-IDF from scratch (math: tf, idf, cosine similarity). Know pros/cons and memory tradeoffs."},
        {"id": "classical_ml", "title": "Naive Bayes / Logistic Regression / SVM", "depth": "medium",
         "details": "Train and evaluate with TF-IDF. Know assumptions (Naive Bayes independence), regularization for LR, kernel intuition for SVM."},
    ],

    "Embeddings & Representation": [
        {"id": "word2vec", "title": "Word2Vec (CBOW, Skip-gram)", "depth": "high",
         "details": "Do the math: objective functions, negative sampling vs hierarchical softmax. Implement a small SGNS model or study gensim's internals."},
        {"id": "glove", "title": "GloVe & Co-occurrence", "depth": "low",
         "details": "Understand co-occurrence matrices and how GloVe factorizes them. Try loading pretrained vectors and doing analogy tasks."},
        {"id": "sentence_embeddings", "title": "Sentence embeddings (SBERT)", "depth": "medium",
         "details": "Learn contrastive training ideas, fine-tune on semantic similarity datasets."},
    ],

    "Sequence Models (RNNs)": [
        {"id": "rnn_basic", "title": "Vanilla RNN intuition & math", "depth": "medium",
         "details": "Understand recurrence, hidden states, the forward equations, and why vanishing gradients happen."},
        {"id": "lstm_gru", "title": "LSTM & GRU internals (gates math)", "depth": "high",
         "details": "Derive gate equations, implement a cell, understand forget/input/output gates and how they mitigate vanishing gradients."},
        {"id": "bptt", "title": "Backprop Through Time (BPTT)", "depth": "medium",
         "details": "Work through a simple BPTT example step-by-step and compute gradients for a 3-step sequence."},
    ],

    "Attention & Transformers": [
        {"id": "attention", "title": "Attention (Q,K,V) — math and intuition", "depth": "high",
         "details": "Compute scaled dot-product attention by hand for small matrices. Understand shapes and masking. Implement simple attention in numpy."},
        {"id": "transformer_arch", "title": "Transformer architecture (encoder/decoder)", "depth": "high",
         "details": "Study multi-head attention, positional encoding, layernorm, residuals. Step through a forward pass with shapes."},
        {"id": "training_transformers", "title": "Training tricks (warmup, optimizers) & efficiency", "depth": "medium",
         "details": "Learn AdamW, learning rate schedules, gradient clipping, mixed precision, and batching strategies."},
    ],

    "Understanding Models (BERT etc.)": [
        {"id": "bert_basics", "title": "BERT: masked LM, input formatting", "depth": "high",
         "details": "Prepare inputs (CLS, SEP, token ids, masks). Fine-tune BERT for classification and token-level tasks."},
        {"id": "roberta_albert", "title": "Variants (RoBERTa, ALBERT, DistilBERT)", "depth": "medium",
         "details": "Understand differences: pretraining corpora, parameter sharing, distillation."},
    ],

    "Generation Models (GPT etc.)": [
        {"id": "gpt_arch", "title": "Decoder-only models & autoregressive generation", "depth": "high",
         "details": "Causal masking, generation decoding algorithms (greedy, beam, sampling), temperature, top-k/top-p."},
        {"id": "beam_search", "title": "Beam search, sampling strategies & evaluation", "depth": "medium",
         "details": "Implement beam search for small toy model and compare outputs to sampling strategies."},
    ],

    "Fine-tuning & Parameter-Efficient FT": [
        {"id": "full_ft", "title": "Full fine-tuning (practical)", "depth": "high",
         "details": "Fine-tune on downstream datasets, monitor for catastrophic forgetting, use appropriate LR and batch sizes."},
        {"id": "lora", "title": "LoRA, QLoRA, PEFT approaches", "depth": "high",
         "details": "Understand low-rank adaptations, memory & speed tradeoffs. Practice LoRA with a small LLM on a dataset."},
    ],

    "RAG & Retrieval": [
        {"id": "embeddings_models", "title": "Embedding models & vector similarity", "depth": "high",
         "details": "Train/use embeddings (SBERT/Instructor). Compute cosine similarity, faiss indexing, approximate NN."},
        {"id": "vector_db", "title": "Vector DBs (FAISS, Chroma, Milvus)", "depth": "medium",
         "details": "Indexing, persistence, recall tuning, reranking, and retrieval latency tradeoffs."},
        {"id": "rag_pipeline", "title": "End-to-end RAG pipeline", "depth": "high",
         "details": "Chunking strategies, metadata, retriever + generator loop, hallucination mitigation."},
    ],

    "Prompting, Evaluation & Safety": [
        {"id": "prompting", "title": "Prompt engineering & CoT", "depth": "medium",
         "details": "Zero/few-shot, chain-of-thought, structured prompts, and prompt injection risks."},
        {"id": "evaluation", "title": "Evaluation metrics & human eval", "depth": "medium",
         "details": "Perplexity, BLEU, ROUGE, human annotations, and building evaluation suites for hallucination."},
        {"id": "safety", "title": "Safety & guardrails", "depth": "medium",
         "details": "Content filters, refusal behavior, moderation APIs, and traceability."},
    ],

    "Production & MLOps": [
        {"id": "deploy", "title": "Deployment (FastAPI, Docker, GPUs)", "depth": "medium",
         "details": "Wrap models in APIs, containerize, GPU inference tips, batching, quantization (8-bit/4-bit)."},
        {"id": "monitor", "title": "Monitoring & cost optimization", "depth": "low",
         "details": "Latency/cost tradeoffs, monitoring drift, logging prompts/outputs for audits."},
    ],
}

# load or init progress
progress = load_progress()
if "completed" not in progress:
    # store completed items as mapping id->bool
    progress["completed"] = {}

# ensure all ids exist
for main, subs in ROADMAP.items():
    for s in subs:
        if s["id"] not in progress["completed"]:
            progress["completed"][s["id"]] = False

# UI layout
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Topics & Subtopics")

    overall_total = 0
    overall_done = 0

    # iterate through main topics
    for main_topic, subs in ROADMAP.items():
        st.subheader(main_topic)
        # compute main topic progress
        total = len(subs)
        done = sum(1 for s in subs if progress["completed"].get(s["id"], False))
        overall_total += total
        overall_done += done

        percent = int(done / total * 100)
        st.progress(percent)

        # show main topic done checkbox (disabled) — auto controlled when all subtopics done
        main_done = (done == total)
        st.checkbox(f"{main_topic} — completed (auto)", value=main_done, key=f"main_{main_topic}", disabled=True)

        # render subtopics
        for s in subs:
            cols = st.columns([0.02, 1])
            with cols[1]:
                checked = st.checkbox(f"{s['title']} — ({s['depth']})", value=progress["completed"].get(s["id"], False), key=s["id"])
                # update progress state
                progress["completed"][s["id"]] = checked

                with st.expander("Details & suggested exercises"):
                    st.write(s.get("details", ""))
                    # small interactive items for deep topics
                    if s.get("id") == "attention":
                        st.markdown("**Mini-task:** compute scaled-dot product attention for Q,K,V below (3x2 matrices).")
                        st.write("Try doing this on paper or in a small numpy script. Shapes matter: Q(K^T)/sqrt(dk) -> softmax -> V")
                    if s.get("id") == "lstm_gru":
                        st.markdown("**Mini-task:** write down LSTM gate equations and compute a forward pass for a toy input. Compare to GRU.")
                    if s.get("id") == "word2vec":
                        st.markdown("**Mini-task:** implement Skip-gram negative sampling objective for tiny corpus and observe embeddings.")

        st.markdown("---")

    # overall progress
    st.header("Overall Progress")
    overall_pct = int(overall_done / overall_total * 100)
    st.metric("Completion", f"{overall_pct}%", delta=f"{overall_done}/{overall_total} subtopics done")

    # export/import
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Save progress"):
            save_progress(progress)
            st.success("Progress saved locally.")
    with col_b:
        if st.button("Export progress JSON"):
            fname = export_progress(progress)
            st.success(f"Exported to {fname}")
            st.markdown(f"[Download exported file](./{fname})")
    with col_c:
        uploaded = st.file_uploader("Import progress JSON", type=["json"])
        if uploaded is not None:
            try:
                loaded = json.load(uploaded)
                if "completed" in loaded:
                    progress["completed"].update(loaded["completed"])
                    save_progress(progress)
                    st.success("Imported progress and saved.")
                else:
                    st.error("JSON must contain a top-level 'completed' mapping.")
            except Exception as e:
                st.error(f"Error loading JSON: {e}")

with col2:
    st.header("Study Assistant")
    st.image(FILE_PATH, caption="Reference image (provided)")

    st.markdown("### Quick tips for each depth level:")
    st.markdown("- **Low:** Read and understand, try a 30-min exercise.\n- **Medium:** Implement from libraries + one from-scratch component.\n- **High:** Derive math, implement simplified version, and build a mini-project.")

    st.markdown("---")
    st.markdown("### Generate a study schedule")
    days = st.number_input("Days available (e.g. 40)", min_value=7, max_value=365, value=56)
    hours = st.number_input("Hours per day", min_value=1, max_value=12, value=5)

    if st.button("Generate naive schedule"):
        # simple allocation: count all items and weight by depth
        weight_map = {"low": 1, "medium": 2, "high": 4}
        items = []
        total_weight = 0
        for main, subs in ROADMAP.items():
            for s in subs:
                d = s.get("depth", "medium").lower()
                w = weight_map.get(d, 2)
                items.append((main, s["title"], d, w))
                total_weight += w
        total_hours = days * hours
        st.write(f"Total study hours: **{total_hours}**")
        st.write("Assignment (topic -> hours):")
        for main, title, d, w in items:
            alloc = max(1, int(total_hours * (w / total_weight)))
            st.write(f"- **{title}** ({d}) → ~{alloc} hours")

    st.markdown("---")
    st.markdown("### Pro tips:")
    st.markdown("- For **High** depth items: implement the math and a tiny from-scratch version in numpy.\n- Use small toy datasets first, then scale.\n- Keep a public repo with notebooks for interviews.")

# persist any changes automatically
save_progress(progress)

st.info("Progress autosaved.")
