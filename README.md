# Local RAG Workshop: LlamaIndex + Ollama 🧠

This repository contains a hands-on technical workshop for building **Local Retrieval-Augmented Generation (RAG)** pipelines. Using the 2024 Natural History Museum new species discoveries as a case study, we demonstrate how to ground local LLMs in private data to eliminate hallucinations and bridge knowledge cut-offs.

---

## 🚀 Quick Start: Run in Google Colab

The easiest way to get started is to run the notebook directly in your browser. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cafoala/technical-llamaindex/blob/main/technical_intro_to_rag.ipynb)

**Instructions for Colab:**
1. Click the **Open in Colab** badge above.
2. Go to **Runtime** > **Change runtime type** and select **T4 GPU**.
3. Select **Runtime** > **Run all**.
4. Once the Ollama server starts in the background, you are ready to experiment!

---

## 📚 Workshop Overview

This hands-on workshop provides a technical introduction to building local RAG pipelines. Participants will learn how to ground local LLMs in private data to solve the problems of knowledge cutoffs and hallucinations. We progress from basic document indexing to advanced techniques like reranking and intent-based routing.

### Key Learning Modules:
* **The Baseline:** Witness LLM "knowledge cut-offs" by asking about 2024 events.
* **Vector Embeddings:** Learn how to turn text into mathematical "semantic" coordinates.
* **Reranking:** Implement a "judge" model to filter noise and increase factual accuracy.
* **Intent Routing:** Build a mini-agent that automatically chooses between a Fact-Finder and a Summarization tool.
* **Cost Efficiency:** See why RAG is more scalable and cost-effective than using massive "long-context" windows.

---

## 🛠️ Tech Stack

* **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/) — The leading data framework for LLM applications.
* **Local LLM Engine:** [Ollama](https://ollama.com/) — Runs models locally on your GPU/CPU for 100% privacy.
* **Models Used:** * `llama3.2:3b` (Generation)
    * `nomic-embed-text` (Embeddings)
* **Environment:** Jupyter Notebooks / Google Colab

---

## 💻 Local Setup (Optional)

If you prefer to run this workshop on your own machine instead of Colab:

1. **Install Ollama:** Download from [ollama.com](https://ollama.com/).
2. **Pull the Models:**
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
