# Presentation + Workshop Notes

## Part 1: Presentation Script (slides.md)

### Slide 1: Chatting With Your Data
Today I am going to show you how to make a local language model useful for your own documents instead of just general internet knowledge.

The stack is simple: Ollama runs the models on google colab, and LlamaIndex handles retrieval and response orchestration.

By the end, you will see both the architecture and a notebook you can run and tune yourself.

Before we build anything, let me define RAG clearly in one sentence.

### Slide 2: What RAG Is And Why It Exists
Retrieval-Augmented Generation, or RAG, is a two-stage pattern:
1. Retrieve evidence relevant to the question from your source documents.
2. Generate an answer using that evidence as context.

Without retrieval, an LLM answers from its pretraining memory, which can be outdated, incomplete, or wrong for your domain.

With retrieval, the model is constrained by the evidence we provide at query time.
That does not make it perfect, but it makes it auditable because we can inspect the retrieved chunks.

This matters because fluent output is not the same as reliable output.

So if context matters so much, why not put the entire document into one huge prompt every time?

### Slide 3: RAG vs Long Context vs Fine-Tuning
There are three common ways to improve domain performance: long context, fine-tuning, and RAG.

Long-context prompting can work, but it is often slower, more expensive per query, and can miss details when critical facts are buried in large prompts.

Fine-tuning is absolutely a valid option, but it is usually the highest-friction path.
It requires curated training data, training runs, evaluation loops, versioning, and retraining when facts change.
It can improve style, behavior, or task performance, but it is not the easiest way to keep up with frequently changing knowledge.

RAG is usually the practical first step for knowledge-heavy use cases.
Instead of retraining the model, we retrieve the best evidence at query time and ground the answer in those passages.

A simple rule of thumb:
1. Need up-to-date factual grounding: start with RAG.
2. Need fixed behavior/style changes: consider fine-tuning.
3. Need very broad context in a one-off case: long context can be fine.

Here is the full RAG pipeline we will implement.

### Slide 4: The 5 Steps Of RAG (Technical View)
Step 1, Load: ingest raw source files into document objects.

Step 2, Chunk: split documents into overlapping passages. Chunk size and overlap are quality-critical because retrieval only sees chunks, not whole documents.

Step 3, Embed: convert each chunk into a dense vector, then store vectors in an index.

Step 4, Retrieve: embed the user query, run similarity search, and fetch top-k candidate chunks.

Step 5, Generate: send question plus retrieved chunks to the LLM to synthesize an answer.

Most production quality issues are in steps 2 to 4, not step 5.
Bad chunk boundaries, weak embeddings, or poor retrieval depth will cap quality before generation even starts.

Let me briefly zoom in on embeddings, because retrieval depends on them.

### Slide 5: Vector Embeddings
An embedding is a numerical representation of meaning.
Chunks about similar concepts end up near each other in vector space, even if they do not share exact words.

Keyword search asks, "Do these words appear?"
Vector search asks, "Is the meaning semantically close?"

In this workshop we use `nomic-embed-text` through Ollama.
For each chunk, we store its vector and metadata.
At query time, we embed the question with the same model and compare vectors.

If your embedding model is weak for your domain language, retrieval quality drops immediately.

### Slide 6: Ollama + LlamaIndex In A Bit More Detail
Ollama gives us a local model runtime and local model registry.
In this notebook, it serves both:
1. `llama3.2:3b` for answer generation.
2. `nomic-embed-text` for embeddings.

LlamaIndex is the orchestration layer.
It provides:
1. `Settings` for global model and chunking defaults.
2. Readers and indices (`SimpleDirectoryReader`, `VectorStoreIndex`, `SummaryIndex`).
3. Query engines for retrieval + synthesis.
4. Postprocessors like rerankers.
5. Routers that choose tools based on query intent.

So Ollama is runtime, LlamaIndex is pipeline.

### Slide 7: Workshop Flow
We will run this as a progressive build:
1. Set up Ollama in Colab and verify inference.
2. Download a fixed source file so everyone has identical data.
3. Compare no-RAG failure against RAG behavior.
4. Add reranking to improve evidence precision.
5. Add routing to choose different tools for different intents.

Run cells in order first.
After a clean pass, we will tune parameters and observe behavior changes.

### Slide 8: Workshop Time
Open [technical_intro_to_rag.ipynb](technical_intro_to_rag.ipynb).

First pass: run end-to-end without changing anything.
Second pass: change one parameter at a time and inspect source nodes.

If you are in Colab, runtime resets are normal.
When that happens, rerun setup and data download before debugging later sections.

Once the basic loop works, we can improve answer quality by improving evidence quality.


### Slide 10: Routing Deep Dive
Not every question wants the same tool.
A narrow fact question benefits from vector retrieval.
A broad overview question can benefit from summary-style querying.

`RouterQueryEngine` handles this by using a selector model and tool descriptions.

Mechanically, the flow is:
1. Define multiple query tools, each with a clear description.
2. The selector (`LLMSingleSelector`) reads the user query.
3. It picks one tool based on intent and the descriptions.
4. The chosen tool executes and returns the answer.

This is a lightweight agent pattern: model-mediated tool choice with explicit policy in tool descriptions.


### Slide 12: Limitations, Summary, And Next Steps
RAG is not magic.
If the fact is not in the source, retrieval cannot find it.
If chunking is poor, retrieval drifts.
If hardware is limited, latency increases.

The right mindset is iterative engineering: build, measure, tune, repeat.

Three takeaways:
1. RAG improves grounding by tying answers to retrievable evidence.
2. Local execution with Ollama can improve privacy and cost control.
3. Quality comes from tuning chunking, retrieval depth, reranking, routing, and prompts.

Your action after this session: pick one parameter, state a hypothesis, and verify it by inspecting source nodes.

## Part 2: Workshop Facilitation Script (technical_intro_to_rag.ipynb)

## Section 1: Setup Ollama In Colab
We start by creating a known-good runtime before touching any RAG logic.

First we install Python dependencies with `%pip install` so the notebook has LlamaIndex, Ollama integrations, and helper libraries.

Then we install the Ollama binary in Colab.
That gives us the local model runtime process inside the notebook environment.

Next we start the Ollama service on `127.0.0.1:11434`.
The cell checks whether the API is already up, starts `ollama serve` if needed, and polls the tags endpoint until reachable.

After that we pull two models:
1. `llama3.2:3b` for generation.
2. `nomic-embed-text` for embedding.

Then we run a smoke test with `ollama run` to confirm inference works.
At this point we are only testing runtime health, not answer quality.

Checkpoint I want to see:
1. Ollama API reachable.
2. Models appear in `ollama list`.
3. Smoke test returns normal text.

If setup fails, we stop here and fix it first.

## Section 2: Download Source Data
Now we create the source file the whole workshop will use: `data/new_species_2024.txt`.

The download cell pulls text from a fixed Google Drive URL, validates a successful response, and writes to disk.

Using one fixed source means everyone has the same retrieval corpus, which makes comparisons meaningful across the room.

After this cell, I check three things:
1. The saved path is correct.
2. Character count is non-zero.
3. The preview text looks valid.

No source file means no index, and no index means no RAG.

## Section 3: Intro To RAG
This section is the core learning loop.

At this point, we have already seen why we are choosing RAG first for this workshop.
Fine-tuning is a real option, but for frequently changing factual content it is usually more effort and cost than retrieval-based grounding.

We first configure LlamaIndex settings:
1. `Settings.llm = Ollama(...)` sets the generation model.
2. `Settings.embed_model = OllamaEmbedding(...)` sets semantic encoding.
3. `Settings.chunk_size` and `Settings.chunk_overlap` control chunk boundaries.

Then we intentionally run a no-RAG query with `Settings.llm.complete(prompt)`.
This shows the baseline failure mode: fluent output without guaranteed grounding.

Next we build the retrieval pipeline:
1. `SimpleDirectoryReader` loads the text file into document objects.
2. `VectorStoreIndex.from_documents(...)` chunks and embeds documents, then builds the vector index.
3. `as_query_engine(similarity_top_k=3)` creates a query engine that retrieves top-k chunks and synthesizes an answer.

We ask the same question again with `vector_query_engine.query(prompt)` and compare outputs.

Then we inspect `rag_response.source_nodes` so we can verify whether the answer is supported by retrieved evidence.

The key point here is trust through inspectability.

### Slide 9: Reranking Deep Dive
Nearest-neighbor retrieval gives us a candidate set.
Some chunks will be relevant, some only loosely related.

Reranking is a second-stage relevance pass over those candidates.
In this notebook we use `LLMRerank`, so yes, the reranker is LLM-based.

Mechanically, the flow is:
1. Retrieve top-k candidates by vector similarity (`similarity_top_k`).
2. Ask the reranker to score or choose the strongest chunks.
3. Keep only top-n (`top_n`) before final synthesis.

So reranking trades extra compute for better evidence quality.
In practice, that often improves factual precision more than swapping to a bigger base model.

## Section 4: Advanced Retrieval (Reranking)
Now we improve retrieval quality.

First we run baseline retrieval:
`baseline_engine = index.as_query_engine(similarity_top_k=8)`.

Then we add LLM reranking:
1. `LLMRerank(choice_batch_size=4, top_n=3)` creates a reranker.
2. We retrieve broadly with `similarity_top_k=12`.
3. `node_postprocessors=[reranker]` filters candidates down to the best `top_n` chunks.

So yes, in this notebook reranking uses another LLM call as a relevance judge over candidate chunks.

We compare baseline and reranked source nodes side by side.
I focus the group on relevance and claim support, not just numeric scores.

If reranking underperforms, I tune `similarity_top_k` and `top_n` before making conclusions.

## Section 5: Intent Routing
Now we add dynamic tool selection.

We build two tools:
1. A vector tool for specific factual lookup.
2. A summary tool backed by `SummaryIndex` for broader synthesis.

Each tool has a description.
Those descriptions are not documentation only; they are routing policy because the selector reads them when deciding.

`RouterQueryEngine` is configured with:
1. `LLMSingleSelector` to choose one tool.
2. A list of `QueryEngineTool` objects.

At query time, the selector predicts intent, picks a tool, and only that tool executes.

When we test with one summary prompt and one fact prompt, we print `response.metadata["selector_result"]` so everyone can see the decision path, not just the final answer.

This is the first agent-like pattern in the notebook: model chooses among tools under explicit policy.

## Suggested End-Of-Workshop Exercises
1. Change `similarity_top_k` and observe recall vs noise.
2. Change reranker `top_n` and observe precision vs coverage.
3. Change chunk size and overlap, then inspect source-node quality.
4. Try multi-part and ambiguous prompts and inspect routing behavior.

Exercise rule:
Pick one parameter, state your hypothesis before running, then verify against source nodes.

## Troubleshooting Script
1. If model calls fail, rerun service startup and model pull cells.
2. If source file is missing, rerun Section 2 download cell.
3. If outputs look inconsistent, rerun from Section 3.1 to reset settings and rebuild index.
4. If Colab runtime reset occurs, rerun Sections 1 and 2 first.

Closing line:
Do not debug five variables at once. Re-establish baseline state, then change one thing at a time.
