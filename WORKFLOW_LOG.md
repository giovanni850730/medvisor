# Agent Collaboration Workflow Log  



setup
- Coding agent: claude, opus 4.8
- LLM inference backend: OpenRouter, OpenAI-compatible API, free model `meta-llama/llama-3.3-70b-instruct:free`
- Image backend: Stable Diffusion via diffusers
- Google Colab (T4)

---

## Phase 1  Ideation & Planning

I started by telling the agent my constraints and background:

> "我想要做有關於神經內科的衛教內容，特別是有關於腦中風方面。 I need a capstone that combines an LLM with a diffusion model, is medical-themed, and runs on a Colab T4. Suggest something that fits my field but is safe to demo publicly."

The agent gave me a few options. I ruled out a radiology-report generator (would need real patient scans data and safety problems) and a plain stroke-risk calculator (the image model would feel tacked on). We settled on MedVisor, a medical *education* assistant: you ask a question, an LLM explains it using a curated knowledge base (RAG), and a diffusion model draws an illustration.

The one design rule I insisted on: the two models should actually work *together*. So I asked the agent to have the LLM write the image prompt that the diffusion model then renders — a real LLM → Diffusion hand-off rather than two separate features.

---

## Phase 2 Architecture Design & Task Decomposition

My prompt:
> "Break this into clean modules. What does each one return, and how does data flow from the question to the final image?"

We agreed on this structure:

| Part | File | What it returns |
|------|------|-----------------|
| Retrieval + LLM answer + image-prompt writing | `modules/llm_rag.py` | `{explanation, sources, image_prompt, primary_topic}` |
| Image generation | `modules/image_gen.py` | a `PIL.Image` |
| Knowledge base | `knowledge_base/medical_facts.json` | list of `{topic, category, content}` |
| UI / orchestration | `app.py` | Gradio app |

Decisions I made as the "architect":
- Keep the LLM behind an OpenAI-compatible client so I can switch between OpenRouter, Big Pickle, or local Ollama by only changing environment variables.
- For a small knowledge base, skip a heavy vector DB  just use normalized sentence-embeddings + cosine similarity.
- Build the knowledge base around my own field: ischemic/hemorrhagic stroke, TIA, lacunar stroke, large hemispheric infarction, malignant cerebral edema, tPA, thrombectomy, NIHSS, stroke imaging — the same topics as my END/MCE prediction research.

---

## Phase 3 Code Generation & Implementation

### Key prompts I used
- "Write a `MedicalRAG` class: load the JSON KB, embed with sentence-transformers, retrieve top-k by cosine similarity, then call an OpenAI-compatible chat endpoint with the retrieved context. Also have it generate a short image prompt and a quiz question."
- "Write a `MedicalImageGenerator` with diffusers."
- "Wire both into a Gradio UI with example stroke questions and a disclaimer."

### Technical problems the agent helped me solve

These are real issues I hit while building this and the related diffusion homeworks that fed into it:

**1. `salesforce-lavis` wouldn't install (BLIP captioning).**
While experimenting with image captioning, `lavis` forced old versions of `transformers` and `huggingface-hub` that conflicted with `diffusers`. After fighting the dependency resolver, the agent pointed out I didn't need `lavis` at all — `transformers` has BLIP built in (`BlipForConditionalGeneration`), same underlying model, zero conflicts. Switched and it just worked.

**2. SDXL VAE produced NaN images in FP16.**
My first diffusion attempts on Colab gave pure-noise / black images. I asked the agent "why are the images garbage / NaN?" and we added a debug check that showed the VAE output was `NaN` at the very first step. This is a known SDXL FP16 bug. Fix: use the community `madebyollin/sdxl-vae-fp16-fix` VAE (or run the VAE in FP32). For MedVisor I avoided the problem entirely by using SDXL-Turbo first, then full SDXL with the fixed VAE.

**3. Image quality was poor (blurry, off-topic).**
The first version used SDXL-Turbo (4 steps, `guidance_scale=0`) — fast but low quality, and it ignores negative prompts. I told the agent "the images aren't clear or ideal, how do I improve this?" Two fixes:
   - **Image side:** switched the default to full **SDXL** (30 steps, 1024px, guidance 7.5) with `enable_vae_tiling()` so it fits the T4, plus a richer style prompt and a proper negative prompt.
   - **LLM side:** rewrote the image-prompt generation so the LLM describes a *concrete anatomical scene* (specific structures, viewpoint, colors) instead of an abstract phrase, with a few-shot example and an explicit "no text in the image" rule.
   Together these made the illustrations much sharper and more on-topic.

**4. Keeping the LLM grounded.**
Early on the LLM added plausible but unsourced medical claims. I strengthened the system prompt to *use only the retrieved context* and to say so when the context doesn't cover the question — important for a medical tool.

**5. Graceful failure when the API key is missing.**
The app crashed if `LLM_API_KEY` wasn't set. Added a fallback that returns the retrieved knowledge-base text instead of crashing, so the demo never hard-fails.

---

## Phase 4 Interface Encapsulation & Finalization

Prompt:
> "Make a Gradio UI with a soft theme, stroke example questions, checkboxes for image/quiz, and a visible disclaimer. Then write the README and a one-click Colab notebook."

Outputs:
- `app.py` — two-column Gradio layout, example stroke questions, educational disclaimer.
- `MedVisor_Colab.ipynb` — installs everything, sets the API key, launches with a public share link.
- `README.md` — architecture diagram + run instructions.
- This log.

I verified the whole pipeline on stroke questions ("What is an ischemic stroke?", "stroke vs TIA", "malignant cerebral edema") and confirmed the explanation stays grounded, the image matches the topic, and the quiz is answerable from the text.


