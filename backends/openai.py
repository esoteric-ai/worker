from backends.base import Backend, ModelConfig, GenerationParams, PRECISE_PARAMS
from typing import AsyncIterator, List, Dict, Literal, Any, Optional, Union, TypedDict, AsyncGenerator

from openai import AsyncOpenAI

class OpenAIBackendConfig(TypedDict):
    api_key: str
    base_url: Optional[str]

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
        
        if "api_key" in model.get("load_options", {}):
            client_kwargs["api_key"] = model['load_options'].get("api_key")
        
        # Check for model-specific API URL first, then fall back to config
        base_url = None
        
        if "api_url" in model.get("load_options", {}):
            base_url = model['load_options'].get("api_url")
            print(f"[OpenAIBackend] Using model-specific API URL: {base_url}")
        elif self.config.get("base_url"):
            base_url = self.config["base_url"]
        
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
    
    async def completion(
        self, 
        prompt: str,  
        stream: bool = False, 
        max_tokens: int = 500, 
        params: GenerationParams = PRECISE_PARAMS
    ) -> Union[str, AsyncIterator]:
        """
        Convert text completion to chat completion format for OpenAI API.
        OpenAI has deprecated direct completions in favor of chat completions.
        """
        # Convert the text prompt to chat format
        conversation = [
            {"role": "user", "content": prompt}
        ]
        
        # Use the existing chat_completion method
        response = await self.chat_completion(
            conversation=conversation,
            stream=stream,
            max_tokens=max_tokens,
            params=params
        )
        
        if stream:
            # For streaming, yield the content from each chunk
            async def content_generator():
                async for chunk in response:
                    if "choices" in chunk and chunk["choices"]:
                        if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                            yield chunk["choices"][0]["delta"]["content"]
            
            return content_generator()
        else:
            # For non-streaming, return just the content string
            if "choices" in response and response["choices"] and "message" in response["choices"][0]:
                return response["choices"][0]["message"].get("content", "")
            return ""
    
    async def _get_pid(self) -> Optional[int]:
        return None