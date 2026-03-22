"""
LLM Querier — Collect Model Responses via Hugging Face Transformers
───────────────────────────────────────────────────────────────────
Supports three modes:
  1. Local HF model (AutoModelForCausalLM / pipeline)
  2. OpenAI API (gpt-4o, gpt-4o-mini, etc.)
  3. Sample data (pre-generated, for running without GPU/API)
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


class LLMQuerier:
    """Query LLMs and collect responses with metadata.

    Usage:
        querier = LLMQuerier(model_name="meta-llama/Llama-3.1-8B-Instruct")
        responses_df = querier.collect(prompts_df)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_repeats: int = 1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_repeats = num_repeats
        self.device = device
        self._pipeline = None
        self._openai_client = None

        # Detect mode
        if model_name.startswith("openai:"):
            self.mode = "openai"
            self.api_model = model_name.replace("openai:", "")
        elif model_name == "sample":
            self.mode = "sample"
        else:
            self.mode = "hf"

    def _init_hf_pipeline(self):
        """Lazy-load the HF text-generation pipeline."""
        if self._pipeline is not None:
            return

        from transformers import pipeline, AutoTokenizer
        import torch

        print(f"Loading model: {self.model_name}")

        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        else:
            device_map = self.device

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map=device_map if device_map != "cpu" else None,
            device=0 if device_map == "cpu" and torch.cuda.is_available() else -1 if device_map == "cpu" else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        print(f"Model loaded on {device_map}")

    def _init_openai_client(self):
        """Lazy-load the OpenAI client."""
        if self._openai_client is not None:
            return
        from openai import OpenAI
        self._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _query_hf(self, prompt: str) -> str:
        """Query a local Hugging Face model."""
        self._init_hf_pipeline()

        messages = [{"role": "user", "content": prompt}]
        output = self._pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            return_full_text=False,
        )
        return output[0]["generated_text"].strip()

    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API."""
        self._init_openai_client()

        response = self._openai_client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content.strip()

    def _query_single(self, prompt: str) -> str:
        """Route to the appropriate backend."""
        if self.mode == "hf":
            return self._query_hf(prompt)
        elif self.mode == "openai":
            return self._query_openai(prompt)
        else:
            raise ValueError(f"Cannot query in mode: {self.mode}")

    def collect(
        self,
        prompts_df: pd.DataFrame,
        save_path: Optional[str] = None,
        checkpoint_every: int = 50,
    ) -> pd.DataFrame:
        """Collect LLM responses for all prompts.

        Args:
            prompts_df: DataFrame from PromptGenerator with prompt_text column.
            save_path: Optional path to save results incrementally.
            checkpoint_every: Save checkpoint every N prompts.

        Returns:
            DataFrame with original columns + response_text, response_length,
            response_time, repeat_idx.
        """
        all_rows = []
        total = len(prompts_df) * self.num_repeats

        print(f"Collecting {total} responses ({len(prompts_df)} prompts × {self.num_repeats} repeats)")
        print(f"Model: {self.model_name} | Mode: {self.mode}")

        with tqdm(total=total, desc="Querying LLM") as pbar:
            for idx, row in prompts_df.iterrows():
                for repeat_idx in range(self.num_repeats):
                    start_time = time.time()
                    try:
                        response = self._query_single(row["prompt_text"])
                        elapsed = time.time() - start_time

                        new_row = row.to_dict()
                        new_row.update({
                            "response_text": response,
                            "response_length": len(response.split()),
                            "response_time": round(elapsed, 3),
                            "repeat_idx": repeat_idx,
                            "error": None,
                        })
                    except Exception as e:
                        elapsed = time.time() - start_time
                        new_row = row.to_dict()
                        new_row.update({
                            "response_text": "",
                            "response_length": 0,
                            "response_time": round(elapsed, 3),
                            "repeat_idx": repeat_idx,
                            "error": str(e),
                        })

                    all_rows.append(new_row)
                    pbar.update(1)

                # Checkpoint
                if save_path and (idx + 1) % checkpoint_every == 0:
                    pd.DataFrame(all_rows).to_json(save_path, orient="records", indent=2)

        result_df = pd.DataFrame(all_rows)

        if save_path:
            result_df.to_json(save_path, orient="records", indent=2)
            print(f"Saved {len(result_df)} responses to {save_path}")

        errors = result_df["error"].notna().sum()
        if errors > 0:
            print(f"Warning: {errors} queries failed. Check 'error' column.")

        return result_df

    @staticmethod
    def load_responses(path: str) -> pd.DataFrame:
        """Load previously collected responses."""
        return pd.read_json(path, orient="records")
