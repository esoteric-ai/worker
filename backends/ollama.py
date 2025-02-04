# backends/ollama.py

import asyncio
from typing import List, Dict, Any, Optional

from ollama import AsyncClient  # Ensure you have installed: pip install ollama
from ollama import ResponseError

from .base import Backend


class OllamaBackend(Backend):
    """
    Concrete implementation for the Ollama API using the async client.

    This backend supports aliasing: you can refer to your model internally
    by a simple name (e.g. "llama8b") and have that mapped to an API name (e.g.
    "llama-8b:latest" or "llama3.1:8b-instruct-q4_K_M"). The API name is used
    only for calling the Ollama library; the worker sees only the internal name.
    """

    def __init__(self, base_url: str, model_aliases: Dict[str, str], supported_models: Optional[List[str]] = None):
        """
        :param base_url: The Ollama API base URL (e.g. "http://localhost:11434")
        :param model_aliases: A dict mapping internal model names to Ollama API identifiers.
                              For example: { "llama8b": "llama-8b:latest" }
        :param supported_models: A list of models supported by this worker (from config).
        """
        self.base_url = base_url.rstrip("/")
        self.model_aliases = model_aliases
        self.supported_models = supported_models if supported_models is not None else []
        # The active model field to be accessed by the worker (internal alias).
        self.active_model: Optional[str] = None
        # The internal name (alias) used for registration.
        self.internal_model: Optional[str] = None
        # The API model name to be used with the Ollama client.
        self.api_model: Optional[str] = None
        self.client = AsyncClient(host=self.base_url)

    async def load_model(self) -> str:
        """
        Load (or ensure loading of) the active model.

        We select the first supported model (if any) and map it using the model_aliases.
        We then call the Ollama ps endpoint to check if the model is running.
        If not, we force the model to load by invoking chat with an empty messages list.

        Returns the internal model identifier (e.g. "llama8b"), not the API name.
        """
        # Choose the target from supported models or use a default.
        target = self.supported_models[0] if self.supported_models else "default-model"
        internal_model = target
        # Map via alias; if no mapping exists, use target as the API name.
        api_model = self.model_aliases.get(target, target)

        try:
            # Check the currently running models using ps.
            ps_response = await self.client.ps()
            # Depending on the response type, extract the list of running models.
            if hasattr(ps_response, "models"):
                running_models = [m.model for m in ps_response.models]
            elif isinstance(ps_response, dict) and "models" in ps_response:
                running_models = [m["model"] for m in ps_response["models"]]
            else:
                running_models = []
        except Exception as e:
            print("[OllamaBackend] Warning: could not retrieve running models via ps():", e)
            running_models = []

        if api_model not in running_models:
            # Force the model to load by calling chat with an empty conversation.
            try:
                await self.client.chat(api_model, messages=[])
            except Exception as e:
                print(f"[OllamaBackend] Error forcing load of model {api_model}:", e)
                raise e

        # Store the internal and API model names.
        self.internal_model = internal_model
        self.api_model = api_model
        # Set active_model to the internal alias so that the worker sees it.
        self.active_model = internal_model

        # Print the internal name (the worker will see this)
        print(f"[OllamaBackend] Active model set to: {self.active_model}")
        return self.active_model

    async def chat_completion(self, conversation: List[Dict[str, Any]]) -> str:
        """
        Given a conversation, produce a completion using the Ollama chat endpoint.

        The conversation is passed directly to the Ollama API using the API model name.
        Returns the generated text.
        """
        if not self.api_model:
            raise ValueError("No active model loaded. Call load_model() first.")

        try:
            response = await self.client.chat(self.api_model, messages=conversation)
        except ResponseError as re:
            print(f"[OllamaBackend] Response error from Ollama: {re}")
            raise re
        except Exception as e:
            print(f"[OllamaBackend] Unexpected error during chat_completion: {e}")
            raise e

        # Try dictionary access first.
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            # Fallback to attribute access.
            content = getattr(getattr(response, "message", {}), "content", "")

        return content
