# backends/tabby.py
from typing import List, Dict, Any, Optional
import httpx
from openai import AsyncOpenAI

from .base import Backend

class TabbyBackend(Backend):
    """
    Concrete implementation for TabbyAPI using OpenAI library with a single client instance.
    
    This backend now supports aliasing: the active model is queried from the API.
    Its API name is then "unalias"ed by reverse–mapping through model_aliases to obtain the
    internal model name (e.g. "llama8b") that the worker uses.
    """

    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        model_aliases: Optional[Dict[str, str]] = None, 
        supported_models: Optional[List[str]] = None  # This parameter is now ignored.
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.active_model = None  # Internal model name used by the worker.
        self.internal_model: Optional[str] = None  # The unaliased model name.
        self.api_model: Optional[str] = None       # The API model name returned from the backend.
        self.model_aliases = model_aliases or {}
        # Note: supported_models is no longer used to pick a model.
        self.supported_models = supported_models if supported_models is not None else []
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1",
        )

    async def load_model(self) -> str:
        """
        Load the active model from the backend and unalias it.
        
        This method queries the backend's /v1/model endpoint to retrieve the API model name.
        It then reverses the model_aliases mapping: if the returned API model name matches one
        of the alias values, the corresponding key is used as the internal model name.
        If no match is found, the API model name is used directly.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Api-Key": self.api_key
        }
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.get(f"{self.base_url}/v1/model", headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Retrieve the model ID returned by the backend.
        api_model = data.get("id", "unknown_model")

        # Reverse lookup in model_aliases: if any alias value equals api_model,
        # then use the corresponding key as the internal model name.
        internal_model = None
        for key, value in self.model_aliases.items():
            if value == api_model:
                internal_model = key
                break
        if internal_model is None:
            # If no alias mapping applies, fallback to the API model name.
            internal_model = api_model

        self.internal_model = internal_model
        self.api_model = api_model
        # The worker will see the internal (unalias-ed) model name.
        self.active_model = internal_model

        print(f"[TabbyBackend] Active model set to: {self.active_model} (API model: {self.api_model})")
        return self.active_model

    async def chat_completion(self, conversation: List[Dict[str, Any]]) -> str:
        """
        Use the shared OpenAI client for chat completions with generation parameters.
        The API call is made using the aliased API model name.
        """
        if not self.api_model:
            raise ValueError("No active model loaded. Call load_model() first.")

        request_body = {
            "model": self.api_model,
            "messages": conversation,
        }

        # Generation parameters with defaults (adjust as needed)
        extra_body = {
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "penalty_range": -1,
            "max_tokens": 700,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": 1.0,
            "top_a": 0.0,
            "temperature": 0.1,
            "temp_last": False,
            "typical": 1.0,
            "tfs": 1.0,
            "logit_bias": None,
            "mirostat_mode": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
        }
        # Remove any None values.
        extra_body = {k: v for k, v in extra_body.items() if v is not None}

        response = await self.client.chat.completions.create(
            **request_body,
            extra_body=extra_body,
        )

        if not response.choices:
            return ""
        
        return response.choices[0].message.content
