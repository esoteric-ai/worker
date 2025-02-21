# backends/openai.py

from typing import List, Dict, Any, Optional
import asyncio
import time
import httpx
from openai import AsyncOpenAI
from .base import Backend

class OpenAIBackend(Backend):
    """
    Backend for OpenAI endpoint (Targon) that supports aliasing of models.
    
    Configuration settings (in config.json) should include:
      - "openai_api_url": URL for the OpenAI API endpoint.
      - "openai_api_key": API key for authentication.
      - "model_aliases": A dictionary mapping internal model names to the API model names.
      - "supported_models": (Optional) List of supported model names.
    
    This backend implements 3 retries with a 5-second interval on failed requests.
    If all attempts fail, an error is raised.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_aliases: Optional[Dict[str, str]] = None,
        supported_models: Optional[List[str]] = None,
        rpm = -1,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_aliases = model_aliases or {}
        self.supported_models = supported_models if supported_models is not None else []
        self.active_model: Optional[str] = None  # Internal (unalias-ed) model name
        self.api_model: Optional[str] = None     # API model name returned from backend
        self.rpm = rpm

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=f"{self.api_url}/v1",
        )

        # Initialize list to track API call timestamps for rate limiting.
        self._api_call_timestamps = []

    async def load_model(self) -> str:
        self.internal_model = self.supported_models[0]
        self.api_model = self.model_aliases[self.supported_models[0]]
        # The worker will see the internal (unalias-ed) model name.
        self.active_model = self.supported_models[0]
        return self.active_model

    async def chat_completion(self, conversation: List[Dict[str, Any]]) -> str:
        """
        Call the OpenAI chat completion endpoint using streaming.
        Uses the shared OpenAI client and streams the response,
        aggregating the content from each chunk.
        
        Implements 3 retries with a 5-second interval on failed requests.
        Also rate limits to 6 API calls per minute.
        """
        if not self.api_model:
            raise ValueError("No active model loaded. Call load_model() first.")

        # Rate limiting: Ensure no more than 6 API calls per minute.
        if self.rpm != -1:
            while True:
                now = time.time()
                # Remove timestamps older than 60 seconds
                self._api_call_timestamps = [t for t in self._api_call_timestamps if now - t < 60]
                if len(self._api_call_timestamps) < self.rpm:
                    break
                else:
                    earliest = min(self._api_call_timestamps)
                    wait_time = 60 - (now - earliest)
                    await asyncio.sleep(wait_time)

            # Record the timestamp for this API call.
            self._api_call_timestamps.append(time.time())

        request_body = {
            "model": self.api_model,
            "messages": conversation,
        }
        extra_body = {
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": 1.0,
        }
        # Remove any None values.
        extra_body = {k: v for k, v in extra_body.items() if v is not None}

        retries = 3
        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    **request_body,
                    stream=True,
                    temperature=0.1,
                    max_tokens=512,
                    frequency_penalty=0,
                    presence_penalty=0,
                    extra_body=extra_body,
                )
                ans = ""
                async for chunk in response:
                    if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                        ans += chunk.choices[0].delta.content
                return ans
            except Exception as e:
                if attempt < retries - 1:
                    print(f"[OpenAIBackend] Error during chat_completion (attempt {attempt+1}): {e}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    print(f"[OpenAIBackend] Failed to complete chat after {retries} attempts.")
                    raise e
