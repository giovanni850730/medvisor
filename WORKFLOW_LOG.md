# 🤖 Agent Collaboration Workflow Log — MedVisor

This document records the agent-assisted development process for the MedVisor capstone project, following the four-phase Agent Workflow required by the assignment.

---

## Phase 1 — Ideation & Planning

**Goal:** Decide on a project that combines an LLM and a Diffusion model in a coherent medical-education application.

**Key prompt given to the agent:**
> "I'm building a capstone for a deep generative models course. It must combine an LLM (RAG or API) with a diffusion model, be medical-themed, and run on Google Colab. Propose a concrete project with a clear architecture and a feature list."

**Agent output (summarized):**
- Proposed **MedVisor**, a medical education assistant.
- Core loop: user question → LLM+RAG explanation → diffusion-generated illustration → optional quiz.
- Rationale: RAG keeps the LLM grounded (reduces hallucination), the diffusion model adds visual learning value, and both can run on Colab (LLM via API, diffusion via GPU).

**Decision:** Adopted the proposal. Chose to keep the LLM behind an OpenAI-compatible API so the same code works with OpenRouter, Big Pickle, or local Ollama.

---

## Phase 2 — Architecture Design & Task Decomposition

**Key prompt:**
> "Break MedVisor into modules with clear responsibilities. Define the data flow between the RAG module, the image generator, and the UI. Specify the API contract between modules."

**Resulting task breakdown:**

| Task | Module | Output contract |
|------|--------|-----------------|
| Retrieval + LLM generation | `modules/llm_rag.py` | `{explanation, sources, image_prompt, primary_topic}` |
| Image synthesis | `modules/image_gen.py` | `PIL.Image` from a text prompt |
| Knowledge base | `knowledge_base/medical_facts.json` | list of `{topic, category, content}` |
| UI + orchestration | `app.py` | Gradio Blocks |

**Design decisions made with the agent:**
- The LLM also writes the **image prompt** (LLM → diffusion handoff), so the two models are genuinely chained rather than independent.
- Embeddings normalized so cosine similarity reduces to a dot product (cheap retrieval, no FAISS dependency needed for a small KB).
- Image style fixed via prompt suffix for visual consistency.

---

## Phase 3 — Code Generation & Implementation

**Tools used:** IDE-based agent + CLI for environment setup.

### Key prompts
1. **RAG module:**
   > "Write a `MedicalRAG` class: load a JSON KB, embed docs with sentence-transformers, retrieve top-k by cosine similarity, then call an OpenAI-compatible chat endpoint with the retrieved context. Include a method that also asks the LLM to produce a short image prompt."

2. **Image module:**
   > "Write a `MedicalImageGenerator` using diffusers. Default to SDXL-Turbo (1-4 steps, guidance 0) for speed on a T4; provide an SD 1.5 fallback. Enforce an educational illustration style and a safety negative prompt."

3. **App:**
   > "Wire both modules into a Gradio Blocks UI with a question box, checkboxes for image/quiz, an examples panel, and outputs for explanation, image, image-prompt, and quiz."

### Technical bottlenecks resolved with the agent

| Bottleneck | Symptom | Resolution (agent-assisted) |
|------------|---------|------------------------------|
| SDXL VAE produced NaN images in FP16 | Black/garbage output | Use SDXL-Turbo (more FP16-stable) and `enable_attention_slicing()` for T4 memory |
| LLM hallucinating beyond the KB | Made-up facts | Strengthened the system prompt to *use only the provided context* + "say so if not covered" |
| Long image-generation latency | UI felt slow | Switched from 25-step SD 1.5 to 4-step SDXL-Turbo (≈6× faster) |
| Provider lock-in | Hard to switch LLMs | Abstracted to an OpenAI-compatible client configurable by env vars |
| Colab missing API key | Crash on startup | Added a graceful fallback that returns the retrieved context if the LLM call fails |

---

## Phase 4 — Interface Encapsulation & Finalization

**Key prompt:**
> "Generate a Gradio UI with a soft theme, example questions, and a visible medical disclaimer. Then draft the README and a one-click Colab notebook."

**Outputs produced:**
- `app.py` with a two-column Gradio layout (input/controls on the left, image on the right; explanation and quiz below).
- `MedVisor_Colab.ipynb` — installs dependencies, sets the API key from Colab secrets, and launches with a public share link.
- `README.md` — architecture diagram, setup, and run instructions.
- This `WORKFLOW_LOG.md`.

**Final verification:**
- Ran the full pipeline end-to-end with several questions (stroke, TIA, heart attack, diabetes).
- Confirmed the explanation stays grounded in retrieved sources, the image reflects the topic, and the quiz is answerable from the explanation.

---

## Reflections

- **What worked:** Letting the LLM produce the diffusion prompt made the two generative models cooperate naturally instead of feeling bolted together.
- **What was tricky:** FP16 stability of SDXL on a T4 — SDXL-Turbo solved both speed and stability at once.
- **If extended:** add a larger vector DB (FAISS/Chroma), allow PDF upload to expand the KB, and add ControlNet so users can sketch the anatomy they want illustrated.
