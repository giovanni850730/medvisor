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
            # Strip accidental wrapping quotes
            prompt = prompt.strip('"').strip("'").strip()
            # Safety: if the model returned something too short, fall back
            if len(prompt) < 15:
                raise ValueError("prompt too short")
        except Exception:
            prompt = (
                f"detailed anatomical cross-section illustrating {topic}, "
                "clean medical textbook diagram, red arteries and blue veins, "
                "clear central focus, neutral background"
            )
        return prompt