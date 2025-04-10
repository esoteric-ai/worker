from backends.base import Backend, ModelConfig, GenerationParams, PRECISE_PARAMS
from typing import AsyncIterator, List, Dict, Literal, Any, Optional, Union, TypedDict, AsyncGenerator

from openai import AsyncOpenAI

class OpenAIBackendConfig(TypedDict):
    api_key: str
    api_base: Optional[str]

class OpenAIBackend(Backend):
    def __init__(self, config: OpenAIBackendConfig):
        self.config = config
        self.client = None
        self.model_config = None
    
    async def get_type(self) -> Literal["Managed", "Instant"]:
        return "Instant"
    
    async def load_model(self, model: ModelConfig) -> None:
        self.model_config = model
        # Configure OpenAI client
        client_kwargs = {
            "api_key": self.config["api_key"]
        }
        
        # Check for model-specific API URL first, then fall back to config
        base_url = None
        if "api_url" in model:
            base_url = model.get("api_url")
            print(f"[OpenAIBackend] Using model-specific API URL: {base_url}")
        elif self.config.get("api_base"):
            base_url = self.config["api_base"]
        
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = AsyncOpenAI(**client_kwargs)
    
    async def unload_model(self) -> None:
        if self.client:
            if hasattr(self.client, 'close') and callable(getattr(self.client, 'close')):
                await self.client.close()
            self.client = None
        
    
    

    async def chat_completion(
        self, 
        conversation: List[Dict[str, Any]], 
        stream: bool = False, 
        tools = [],
        max_tokens: int = 500, 
        params: GenerationParams = PRECISE_PARAMS, mm_processor_kwargs={}, extra={}
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Use the shared OpenAI client for chat completions with generation parameters.
        The API call is made using the aliased API model name.
        Supports both streaming and non-streaming responses.
        """

        request_body = {
            "model": self.model_config.get("api_name"),
            "messages": conversation,
            "stream": stream,
            "tools": tools
        }

        extra_body = {
            "temperature": params.get("temperature", 0.1),
            "max_tokens": max_tokens,
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", 1),
            "min_p": params.get("min_p", 0.0),
            "repetition_penalty": params.get("repetition_penalty", 1.0),
            "frequency_penalty": params.get("frequency_penalty", 0.0),
            "presence_penalty": params.get("presence_penalty", 0.0),
        }

        try:
            if stream:
                # Return an async generator that yields chunks
                async def response_generator():
                    stream_response = await self.client.chat.completions.create(
                        **request_body,
                        extra_body=extra_body,
                    )

                    async for chunk in stream_response:
                        yield chunk.model_dump()

                return response_generator()
            else:
                response = await self.client.chat.completions.create(
                    **request_body,
                    extra_body=extra_body,
                )
                return response.model_dump()
            
        except Exception as e:
            print(f"ERROR in chat_completion: {type(e).__name__}: {str(e)}")
            # Re-raise to ensure benchmark correctly detects the failure
            raise
        
    async def _get_pid(self) -> Optional[int]:
        return None