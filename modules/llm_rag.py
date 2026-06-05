"""
llm_rag.py
==========
LLM + Retrieval-Augmented Generation (RAG) module for MedVisor.

This module:
1. Loads a curated medical knowledge base.
2. Builds dense embeddings for each document using sentence-transformers.
3. Retrieves the most relevant documents for a user query (semantic search).
4. Calls an LLM (via an OpenAI-compatible API) with the retrieved context
   to generate a grounded, patient-friendly explanation.

The LLM backend is configurable and works with any OpenAI-compatible endpoint:
- OpenRouter   (https://openrouter.ai/api/v1)
- Big Pickle   (OpenCode platform, model id: opencode/big-pickle)
- Ollama       (http://localhost:11434/v1)
"""

import os
import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from openai import OpenAI


class MedicalRAG:
    """Retrieval-Augmented Generation over a medical knowledge base."""

    def __init__(
        self,
        knowledge_base_path: str = "knowledge_base/medical_facts.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_base_url: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
    ):
        # ----- Load knowledge base -----
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        print(f"[RAG] Loaded {len(self.documents)} documents from knowledge base.")

        # ----- Build embeddings for retrieval -----
        self.embedder = SentenceTransformer(embedding_model)
        corpus = [f"{d['topic']}. {d['content']}" for d in self.documents]
        self.doc_embeddings = self.embedder.encode(
            corpus, convert_to_numpy=True, normalize_embeddings=True
        )
        print(f"[RAG] Built embeddings: {self.doc_embeddings.shape}")

        # ----- Configure LLM client (OpenAI-compatible) -----
        self.llm_base_url = llm_base_url or os.getenv(
            "LLM_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "")
        self.llm_model = llm_model or os.getenv(
            "LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
        )

        self.client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
        print(f"[RAG] LLM configured: {self.llm_model} @ {self.llm_base_url}")

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return the top_k most relevant documents for a query."""
        query_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.doc_embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = dict(self.documents[idx])
            doc["score"] = float(scores[idx])
            results.append(doc)
        return results

    # ------------------------------------------------------------------
    def generate_explanation(self, query: str, top_k: int = 3) -> Dict:
        """
        RAG pipeline: retrieve relevant context, then generate a
        patient-friendly explanation grounded in that context.
        """
        # ----- 1. Retrieve -----
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(
            f"[{d['topic']}] {d['content']}" for d in retrieved
        )

        # ----- 2. Build prompt -----
        system_prompt = (
            "You are MedVisor, a careful medical education assistant. "
            "Explain medical concepts in clear, patient-friendly language at about "
            "a high-school reading level. Use ONLY the provided context; if the "
            "context does not cover the question, say so honestly. Always include a "
            "brief disclaimer that this is educational information and not a "
            "substitute for professional medical advice. Keep the explanation to "
            "about 150-200 words."
        )

        user_prompt = (
            f"Context from medical knowledge base:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a clear educational explanation based on the context above."
        )

        # ----- 3. Call LLM -----
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=400,
            )
            explanation = response.choices[0].message.content.strip()
        except Exception as e:
            explanation = (
                f"[LLM error: {e}]\n\nFalling back to retrieved context:\n\n"
                + retrieved[0]["content"]
            )

        # ----- 4. Build an image prompt for the diffusion model -----
        image_prompt = self._build_image_prompt(retrieved[0]["topic"], query)

        return {
            "explanation": explanation,
            "sources": retrieved,
            "image_prompt": image_prompt,
            "primary_topic": retrieved[0]["topic"],
        }

    # ------------------------------------------------------------------
    def _build_image_prompt(self, topic: str, query: str) -> str:
        """Use the LLM to write a detailed, visual prompt for the diffusion model."""
        system_prompt = (
            "You are a prompt engineer for a medical text-to-image model. "
            "Given a medical topic, write ONE detailed image prompt that describes "
            "a CONCRETE, VISUAL anatomical scene a textbook illustrator would draw.\n\n"
            "Rules:\n"
            "- Describe specific visible anatomy (organs, vessels, structures) and what "
            "is happening to them, not abstract concepts.\n"
            "- Specify a clear viewpoint (e.g., 'cross-section', 'cutaway view', "
            "'front-facing diagram').\n"
            "- Name the key colors when helpful (e.g., 'red arteries, blue veins').\n"
            "- Keep it to 30-45 words, comma-separated descriptive phrases.\n"
            "- Output ONLY the prompt text. No quotes, no preamble, no explanation.\n"
            "- Do NOT include any text, words, labels, or letters in the image description.\n\n"
            "Example topic: Ischemic Stroke\n"
            "Example output: cross-section of a human brain showing a cerebral artery "
            "blocked by a dark blood clot, surrounding brain tissue pale and shaded to "
            "show reduced blood flow, red arteries and blue veins, clean anatomical diagram"
        )

        user_prompt = (
            f"Medical topic: {topic}\n"
            f"User question: {query}\n\n"
            "Write the image prompt describing the most relevant anatomical scene."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=120,
            )
            prompt = response.choices[0].message.content.strip()
            prompt = prompt.strip('"').strip("'").strip()
            if len(prompt) < 15:
                raise ValueError("prompt too short")
        except Exception:
            prompt = (
                f"detailed anatomical cross-section illustrating {topic}, "
                "clean medical textbook diagram, red arteries and blue veins, "
                "clear central focus, neutral background"
            )
        return prompt

    # ------------------------------------------------------------------
    def generate_quiz(self, topic: str, context: str) -> str:
        """Generate a single multiple-choice quiz question about the topic."""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical educator. Based on the provided context, "
                            "write ONE multiple-choice question with 4 options (A-D), "
                            "then give the correct answer and a one-sentence explanation. "
                            "Format clearly with the question, options, and answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\n\nContext: {context}\n\nWrite one quiz question.",
                    },
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Quiz generation error: {e}]"


if __name__ == "__main__":
    rag = MedicalRAG()
    result = rag.generate_explanation("What is an ischemic stroke?")
    print("\n=== EXPLANATION ===")
    print(result["explanation"])
    print("\n=== IMAGE PROMPT ===")
    print(result["image_prompt"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  - {s['topic']} (score: {s['score']:.3f})")
