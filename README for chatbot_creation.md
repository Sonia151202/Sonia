# 🧠 Class 9 Science RAG-Based AI Chatbot

This project is an AI-powered educational assistant designed to answer questions from the Class 9 Science NCERT textbook using a Retrieval-Augmented Generation (RAG) pipeline. It is built to support students of all learning levels by providing context-aware, accurate answers.

---

## 🛠️ Tech Stack

| Component               | Library / Tool                            |
|------------------------|-------------------------------------------|
| PDF Parsing            | PyPDF2                                    |
| Text Preprocessing     | NLTK                                       |
| Embedding Model        | Sentence Transformers (`all-MiniLM-L6-v2`)|
| Vector Store           | FAISS                                      |
| Language Model         | FLAN-T5 (`google/flan-t5-large`) via HuggingFace |
| Web Interface          | Streamlit                                 |
| Logging                | JSON logs (custom format)                 |

---

## 🏗️ Architecture Diagram

```
📄 NCERT PDF Textbook
       ↓
[Text Extraction and Chunking]
       ↓
[Text Embedding using SentenceTransformer]
       ↓
[FAISS Vector Store]
       ↓
User Query
       ↓
[Retrieve Top-k Relevant Chunks]
       ↓
[FLAN-T5 LLM: Generate Answer using Context]
       ↓
Answer + Logging
```

---

## 📌 Assumptions Made

- The NCERT PDF textbook has extractable text.
- Text chunks of 3–5 sentences are sufficient for semantic context.
- Student level is selected manually (`weak` or `strong`).
- All interactions are logged in `chat_log.json`.

---

## ⚠️ Limitations & Areas for Improvement

### Current Limitations

- Limited to static PDF input (no multiple textbook support).
- FAISS and SentenceTransformers may be challenging to set up on Windows.
- No semantic reranking beyond cosine similarity.
- No memory or follow-up question support.

### Future Improvements

- Use `LangChain` or `LlamaIndex` for advanced RAG flow.
- Replace FAISS with a scalable vector DB like `Pinecone` or `ChromaDB`.
- Integrate voice interaction and chatbot history.
- Improve chunking via semantic splitting.
- Add ability to choose between multiple subjects/textbooks.

---

## 🧾 Sample Logs

```json
{
  "timestamp": "2025-06-04 13:30:45",
  "user_id": "student_1",
  "student_level": "weak",
  "query": "Why does a ball thrown upwards fall down?",
  "retrieved_chunks": [54, 12, 76, 83, 5],
  "answer": "A ball falls back down due to the force of gravity pulling it towards the Earth..."
}
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run chatbot.py
```

---

Made with ❤️ for Class 9 students using open-source AI.
