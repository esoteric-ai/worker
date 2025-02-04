# backends/tabby.py
from openai import AsyncOpenAI
from typing import List, Dict, Any
import httpx

from .base import Backend

class TabbyBackend(Backend):
    """
    Concrete implementation for TabbyAPI using OpenAI library with a single client instance.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.active_model = None
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1",
        )

    async def load_model(self) -> str:
        """
        Fetch the active model using HTTP to remain compatible.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Api-Key": self.api_key
        }
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.get(f"{self.base_url}/v1/model", headers=headers)
            resp.raise_for_status()
            data = resp.json()

        self.active_model = data.get("id", "unknown_model")
        return self.active_model

    async def chat_completion(self, conversation: List[Dict[str, Any]]) -> str:
        """
        Use the shared OpenAI client for chat completions with generation parameters.
        """
        request_body = {
            "model": self.active_model,
            "messages": conversation,
        }

        # Generation parameters with defaults (adjust as needed)
        extra_body = {
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "penalty_range": -1,
            "max_tokens": 300,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": 1.0,
            "top_a": 0.0,
            "temperature": 1.0,
            "temp_last": False,
            "typical": 1.0,
            "tfs": 1.0,
            "logit_bias": None,
            "mirostat_mode": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
        }
        extra_body = {k: v for k, v in extra_body.items() if v is not None}

        response = await self.client.chat.completions.create(
            **request_body,
            extra_body=extra_body,
        )

        if not response.choices:
            return ""
        
        return response.choices[0].message.content