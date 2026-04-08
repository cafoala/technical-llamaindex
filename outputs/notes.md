# Presentation + Workshop Notes

## Part 1: Presentation Talk Track (slides.md)

### Slide 1: Chatting with Your Data
This session is about taking a local language model, which is normally generic, and making it useful for a specific set of documents. Mention that there are two themes running through everything today: trust and control. Trust means we want answers grounded in evidence. Control means we want to decide where the model runs and what data it sees.

Then introduce the stack in one sentence: Ollama runs local models, and LlamaIndex gives us the data workflow for retrieval and response generation. Close this slide by telling people they are going to see both the conceptual architecture and a practical notebook they can run themselves.

Transition line:
"Before we build anything, let us anchor on the core problem we are trying to solve."

### Slide 2: The Why - The Knowledge Gap
Base LLMs are frozen snapshots of the world. Even very good models can be confidently wrong when asked about recent events or internal documents they have never seen. Emphasize that this is not a bug in one model, it is a structural limitation of pretraining.

At this point, highlight the practical risk: if teams treat fluent output as reliable output, mistakes appear quickly in analysis, reporting, and decisions. RAG exists to reduce that gap by injecting the right context at query time. RAG does not make the model omniscient, but it does make it far more accountable to provided evidence.

Transition line:
"So if context matters, why not just paste everything into one huge prompt?"

### Slide 3: RAG vs Long Context
Talk track:
Start by acknowledging that long-context models are useful and often impressive. Then explain the tradeoff: long context is frequently expensive, slower, and can still degrade in quality when critical details are buried. Mention the "lost in the middle" effect in everyday terms: the model can miss key facts that are literally present in the prompt.

Now position RAG as a targeting strategy. Instead of sending everything, we send the best evidence for this question. This makes performance more stable, especially for smaller local models, and gives a clearer tuning surface for engineering teams.

Transition line:
"If RAG is a targeting strategy, here is the full pipeline from raw files to grounded answers."

### Slide 4: The 5 Steps of RAG
Talk track:
Walk through the five steps slowly and tie each one to a concrete mental model. Load means getting your source into the system. Chunk means choosing useful passage boundaries. Embed means turning text into comparable vectors. Retrieve means finding likely evidence. Generate means writing the final answer from question plus context.

Emphasize that most real-world quality issues are not in the final generation step. They are usually in chunking and retrieval configuration. This helps people understand why the workshop spends time on retrieval behavior and reranking.

Transition line:
"To understand retrieval, we need one concept that sounds abstract but is actually very practical: embeddings."

### Slide 5: Vector Embeddings
Talk track:
Describe embeddings as coordinates for meaning, not words. If two passages are about similar ideas, their vectors should be near each other. That allows semantic search: we can retrieve conceptually related passages even when wording differs.

You can use a short analogy: keyword search asks "does this exact phrase appear," while vector search asks "does this passage mean something close to my question." Remind the audience that embedding quality strongly influences retrieval quality.

Transition line:
"Now that the concepts are in place, let us ground this in the exact tools we are using."

### Slide 6: Our Tech Stack
Talk track:
Explain why this stack is practical for workshops and teams. Ollama keeps execution local, which is useful for privacy, reproducibility, and low-friction experimentation. LlamaIndex provides high-level building blocks so we can focus on behavior and tuning rather than low-level plumbing.

Also note that this stack is portable. The same conceptual workflow can be moved to different models or deployment environments later. What the audience is learning today is a reusable pattern, not a one-off demo.

Transition line:
"Let us map this directly onto the notebook you will run."

### Slide 7: Workshop Flow
Talk track:
Read the flow as a narrative journey. Section 1 gets the runtime ready. Section 2 gives us a stable source document. Section 3 demonstrates the no-RAG failure and RAG improvement. Section 4 improves retrieval quality with reranking. Section 5 introduces routing so different query intents can use different tools.

Set expectations that this is intentionally progressive. Each section builds state used by later cells. Encourage participants to run in order first, then revisit tuning cells after a full successful pass.

Transition line:
"Great, now we switch from concept mode to hands-on mode."

### Slide 8: Workshop Time
Talk track:
Ask everyone to open technical_intro_to_rag.ipynb. If needed, pause briefly so the room can catch up. Explain that the first pass is purely operational: run cells in sequence and verify checkpoints. The second pass is exploratory: change parameters and observe behavior.

Give one practical warning for Colab users: runtime resets are normal, so setup and source download may need to be rerun. Framing this early prevents frustration later.

Transition line:
"Once the basic loop works, we can improve answer quality by improving evidence quality."

### Slide 9: Reranking Deep Dive
Talk track:
Explain the baseline problem clearly: nearest-neighbor retrieval can include passages that are semantically close but not truly best for the specific question. Reranking acts as a second-stage judge that evaluates candidate passages with tighter relevance scoring.

Stress the engineering point: reranking is often a quality lever that is easier to tune than changing the base model. In this workshop we retrieve broadly, then keep a smaller, cleaner set before synthesis.

Transition line:
"Quality is not only about better ranking, it is also about choosing the right tool for the question."

### Slide 10: Routing Deep Dive
Talk track:
Introduce routing as policy-driven tool selection. Some prompts ask for a precise fact, others ask for a broad summary. A single retrieval strategy is not always optimal for both. RouterQueryEngine lets the model choose between tools based on intent.

Position this as a gentle entry into agentic behavior. You are not building a complex autonomous agent here, but you are introducing one key ingredient: dynamic decision-making over available tools.

Transition line:
"Before we close, let us be realistic about where this approach can still fail."

### Slide 11: Limitations and Reality Check
Talk track:
Reinforce that RAG is not magic. If facts are absent from the source, the system cannot retrieve them. If chunking is poor, retrieval can drift. If hardware is constrained, latency will be visible. This makes evaluation and iteration part of the workflow, not optional extras.

It helps to say this explicitly so participants leave with a practical mindset: build, measure, tune, repeat.

Transition line:
"So what should people take away and do next after today?"

### Slide 12: Summary and Next Steps
Talk track:
Summarize in three points. First, RAG improves grounding and reduces confident errors for domain questions. Second, local execution can improve privacy and cost control. Third, quality comes from tuning choices such as chunk size, overlap, top-k, reranker top-n, and prompt design.

End with an action-oriented invitation: ask each participant to choose one parameter to change and one hypothesis to test. This turns the session into a repeatable experimentation habit rather than a one-time demo.

## Part 2: Workshop Facilitation Script (technical_intro_to_rag.ipynb)

## Section 1: Setup Ollama In Colab
Goal:
- Bring every participant to a known-good starting point before any RAG logic.

Long-form facilitation script:
Start by telling the room this section is about removing uncertainty. Everyone should install dependencies, install Ollama in Colab, start the service, pull models, and run a smoke test. Explain that if this section is unstable, everything downstream becomes noisy and hard to diagnose.

When running the model pull cell, prepare participants for waiting time. Say that this is expected on first run and usually faster on repeat within the same runtime. During smoke test output, remind them that you are not evaluating answer quality yet. You are only verifying that inference works.

What to watch for:
- Ollama service reachable at 127.0.0.1:11434.
- Required models available.
- Smoke test returns normal text output.

Recovery lines:
- "If this fails, do not debug later cells yet. Stay here and fix setup first."
- "In Colab, runtime resets are common. Re-running setup is normal, not a mistake."

## Section 2: Download Source Data
Goal:
- Produce data/new_species_2024.txt from the shared Drive source and verify it exists.

Long-form facilitation script:
Explain that this workshop intentionally uses a fixed source file so everyone retrieves against the same content. That gives cleaner discussion when comparing outputs across participants. Mention that a fixed source also avoids live scraping failures and page structure changes during teaching.

After running the cell, ask participants to check the printed path and character count. Briefly show the first few lines of output preview and connect this to the later retrieval steps: "This text is the entire knowledge base your RAG pipeline can use."

What to watch for:
- Saved file path is correct.
- Output is non-empty.
- Source URL printed is the Drive download URL.

Recovery lines:
- "If this cell fails, rerun it before touching retrieval cells."
- "No source file means no index, and no index means no RAG."

## Section 3: Intro To RAG
Goal:
- Demonstrate the baseline failure and then the RAG improvement with evidence inspection.

Long-form facilitation script:
This is the teaching core. First run the no-RAG question. Ask participants to observe not just whether it is wrong, but how it sounds wrong. Often it will be plausible, fluent, and unsupported. That is the key behavior to notice.

Next run model configuration and index creation. As you do this, narrate what each object represents: Settings for global behavior, vector index for document chunks, and query engine for retrieval plus synthesis. Then ask the same question again and compare output tone, specificity, and factual grounding.

Finally, run the source-node inspection cell. Tell participants this is where trust is built. They can inspect the retrieved evidence directly and validate whether the answer is justified.

What to watch for:
- Clear contrast between no-RAG and RAG responses.
- Retrieved chunks that visibly support answer claims.
- Participants can explain why the RAG answer is more trustworthy.

Recovery lines:
- "If the RAG answer looks weak, inspect source nodes first before changing models."
- "Most fixes start in retrieval settings, not in prompt wording."

## Section 4: Advanced Retrieval (Reranking)
Goal:
- Show how reranking can improve relevance of final evidence passed to the model.

Long-form facilitation script:
Start by framing baseline retrieval as a broad net. It often catches useful passages and some noisy ones. Then introduce reranking as a second-stage filter that revisits candidate chunks and keeps the strongest matches.

Run baseline and reranked outputs, then compare source-node lists side by side. Encourage participants to comment on relevance, not just score values. Ask: "Which set of chunks would you trust more if this answer were going into a report?"

Connect this to practice: reranking is often a high-impact improvement when answers feel close but inconsistent.

What to watch for:
- Reranked context appears tighter and less off-topic.
- Participants can articulate why selected chunks are better.

Recovery lines:
- "If reranking seems worse, adjust top-k and top-n before concluding it does not help."
- "Always compare source nodes, not just final prose."

## Section 5: Intent Routing
Goal:
- Demonstrate dynamic tool selection based on prompt intent.

Long-form facilitation script:
Introduce routing with a simple distinction: fact lookup versus broad summary. Explain that one query engine is tuned for specific retrieval while another is tuned for whole-document summarization behavior. The router selects between them at runtime.

Run the two test prompts and read the selection metadata aloud. This is useful because participants can see not only the answer but the decision pathway behind the answer. Then invite custom prompts and ask participants to predict which tool will be selected before execution.

Position this as a bridge to agent patterns: multiple tools, explicit descriptions, model-mediated selection.

What to watch for:
- Summary prompt routes to summary-oriented tool.
- Fact prompt routes to vector-oriented tool.
- Participants understand that tool descriptions influence routing behavior.

Recovery lines:
- "If routing looks odd, tighten tool descriptions so intent boundaries are clearer."
- "Treat routing as a policy prompt problem as much as a retrieval problem."

## Suggested Exercises For The End Of Workshop
Use this block if you have extra time or want a take-home challenge:

- Change similarity_top_k and describe how evidence diversity changes.
- Change reranker top_n and observe precision versus coverage.
- Change chunk size and overlap, then compare source-node quality.
- Ask one multi-part question and inspect whether routing still chooses the right tool.
- Ask one deliberately ambiguous question and discuss routing uncertainty.

Facilitator prompt:
"Pick one parameter, state your hypothesis before running, then check whether the source nodes support your conclusion."

## Troubleshooting Script
Use these lines exactly when people get stuck:

- If model calls fail, rerun setup cells for service startup and model pull.
- If source file is missing, rerun the Section 2 download cell.
- If responses look inconsistent, rerun from Section 3.1 to reset settings and rebuild index.
- If Colab runtime reset happened, rerun Sections 1 and 2 first.

Closing troubleshooting message:
"Do not debug five variables at once. Re-establish baseline state, then change one thing at a time."
