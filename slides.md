---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
style: |
  section { font-family: 'Segoe UI', sans-serif; }
  h1 { color: #2c3e50; }
  .small-text { font-size: 0.8em; }
---

# Chatting with Your Data 🧠
### Local RAG with LlamaIndex & Ollama

**Cat Russon** | Technical Workshop
April 2026

---

# The "Why": The Knowledge Gap

- **Frozen in Time:** LLMs have training cut-offs (Llama 3.2 doesn't know 2025/2026 news).
- **The Confident Liar:** Without data, models **hallucinate** to please the user.
- **Privacy:** Sending internal specs to cloud APIs is a security risk.

---

# RAG vs. "Long Context"

Many models now have massive context windows (128k+ tokens). Why not just paste the whole document?

1. **Cost:** Feeding 100k tokens into every prompt is 100x more expensive than retrieving 2 relevant chunks.
2. **Speed:** Large contexts slow down "Time to First Token." RAG is snappy.
3. **Accuracy:** Models often suffer from **"Lost in the Middle"**—they ignore facts buried in the center of a massive prompt.
4. **Model Choice:** RAG allows a "small" model (like Llama 3.2 3B) to outperform a "giant" model by giving it exactly what it needs.

---

# The 5 Steps of RAG

1. **Load:** Import files (PDF, TXT, APIs).
2. **Chunk:** Break long docs into small, overlapping pieces so meaning isn't lost at the edges.
3. **Embed:** Convert those chunks into Vectors using an Embedding Model (e.g., `nomic-embed-text`).
4. **Retrieve:** Find the "Top K" chunks mathematically closest to the user's query.
5. **Generate:** Send the query + chunks to the LLM for the final answer.

![alt text](rag_architecture.png)

---

# What are Vector Embeddings?

An LLM doesn't see "Piranha" and "Fish" as words; it sees them as coordinates in a multi-dimensional space.

- **The Vector:** A list of numbers (e.g., `[0.12, -0.59, 0.88...]`) representing the *meaning* of a text.
- **Semantic Space:** In this "map" of meaning, "Piranha" is physically close to "Amazon River" but far away from "Desktop Computer."
- **Search by Meaning:** When you ask a question, we don't look for matching *keywords*; we look for the closest *coordinates*.

![alt text](vector_embeddings.png)
---

# Our Tech Stack

### 🦙 Ollama
The "Engine." Runs models locally on your GPU/CPU. **100% Private.**

### 🗂️ LlamaIndex
The "Data Framework." It is the library that handles the "gnarly parts" of the 5 steps above with minimal code.

---

# 💻 Workshop Time!
### Open: `03_local_ollama_rag.ipynb`

---

# Deep Dive: What is Reranking?

**The Problem:** Vector search is "fuzzy." It might grab a paragraph about a *snake* when you asked about a *fish* because they both mention "Amazon."

**The Solution:**
1. Grab the top 10 most "similar" chunks.
2. Use a **Reranker** (a "Cross-Encoder" model) to act as a judge.
3. Re-score those 10 chunks and keep only the top 2.

> **Analogy:** Vector Search is the *Librarian* (fast), Reranking is the *Subject Matter Expert* (accurate).

---

# Deep Dive: What is Routing?

**"The First Step to AI Agents"**

Not every question needs a search. 
- **User:** "Give me a 3-sentence summary of this whole file." 
- **Vector Search:** Will only see 2 paragraphs. It will fail.

**The Router** acts as a traffic controller:
- **Summarization Tool:** Reads the whole index.
- **Vector Tool:** Searches for specific facts.
- **The LLM decides** which tool to use based on your prompt.

---

# Limitations & Reality Check

1. **Hardware:** Local models are only as fast as your hardware.
2. **The "Chunking" Art:** Too small = loss of context. Too big = distraction.
3. **Retrieval Failure:** If the information isn't in your data, RAG can't find it!

---

# Summary & Next Steps

- **Cost Efficiency:** Use RAG to make small, cheap models act like geniuses.
- **Privacy:** Keep your data on your machine.
- **Next Steps:** Try changing the `CHUNK_SIZE` in the sandbox to see how it changes the AI's "understanding."

### Questions?