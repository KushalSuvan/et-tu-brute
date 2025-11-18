# **Et tu, Brute?**

## **Group 4 — Contributors**

* **Shashank Tippanavar — IMT2022014**
* **Kushal Suvan Jenamani — IMT2022057**
* **Aditya Saraf — IMT2022067**

---

# **Project Overview**

This project presents a complete, containerized **Retrieval-Augmented Generation (RAG)** system designed to act as an *Expert Shakespearean Scholar* for *The Tragedy of Julius Caesar (Folger Edition)*. The system extracts, processes, indexes, retrieves, and generates answers grounded strictly in the original text.

The system is deployed end-to-end using **Docker Compose**, enabling seamless local execution.

---

# **System Architecture**

```
          User Query
              │
              ▼
      FastAPI RAG Backend
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
 Vector Store     Embedding Model
   (Chroma)     (all-MiniLM-L6-v4)
              Retrieved Chunks
                     │
                     ▼
              Gemini LLM
                     │
                     ▼
             Final Response
```

**Explanation:**

1. User sends a query to the FastAPI server.
2. Query is embedded and relevant chunks are retrieved from ChromaDB.
3. Retrieved context + query are sent to **Gemini**.
4. Gemini generates a grounded answer.

---

# **Chunking Strategy**

To ensure faithful retrieval aligned with dramatic structure, we used a **two-layer chunking approach**:

### **1. Scene + Act-Based Primary Chunking**

* Mirrors Shakespeare’s structure.
* Preserves semantic continuity in scenes and monologues.
* Ensures retrieval relevance.

### **2. Overlapping Split Within Each Scene**

* Provides finer granularity.
* Helps capture transitions and dialogue shifts.
* Improves context coverage for questions requiring specific citations.

This hybrid approach yielded strong retrieval consistency.

---

# **Embedding Model Choice**

We selected **`all-MiniLM-L6-v4`** as the embedding model because it consistently produced:

* Better recall for classical English phrasing
* Stable clustering for dramatic dialogue
* Efficient retrieval performance

It outperformed heavier models in terms of retrieval precision and semantic alignment.

---

# **Generation Model**

We used **Google Gemini** as the LLM for response generation, chosen for:

* Strong grounding and low hallucination rates
* Excellent contextual reasoning
* Clean handling of literary language

Gemini produces responses *only* using retrieved chunks.

---

# **Running the System**

To start all services:

### **1. Open the project folder**

```
cd <project-folder>
```

### **2. Launch the containerized stack**

```
docker compose up -d
```

### **3. Access the Frontend**

Visit:

```
http://localhost:8501
```

### **4. API Documentation**

FastAPI interactive docs:

```
http://localhost:8000/docs
```

---

# **Evaluation Summary**

The system was evaluated on:

* 25 baseline questions
* 10 additional analytical questions
* Metrics: **Faithfulness** and **Answer Relevancy**

Our chunking structure and MiniLM embeddings produced strong grounding and accurate retrieval.

---

# **Conclusion**

Group 4’s RAG system successfully delivers:

* Robust Shakespearean scholar persona
* High-quality retrieval and grounded generation
* Fully containerized end-to-end deployment

All assignment requirements—from ETL to evaluation—are satisfied.
