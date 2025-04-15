from backends.base import Backend, ModelConfig, GenerationParams, PRECISE_PARAMS
from typing import AsyncIterator, List, Dict, Literal, Any, Optional, Union, TypedDict
import aiohttp
import json
import asyncio
from datetime import datetime

class YandexBackendConfig(TypedDict):
    api_key: str
    folder_id: str
    base_url: Optional[str]

class YandexBackend(Backend):
    def __init__(self, config: YandexBackendConfig):
        self.config = config
        self.model_config = None
        self.base_url = config.get("base_url", "https://llm.api.cloud.yandex.net/foundationModels/v1/completion")
        self.session = None
    
    async def get_type(self) -> Literal["Managed", "Instant"]:
        return "Instant"
    
    async def load_model(self, model: ModelConfig) -> None:
        self.model_config = model
        # Configure HTTP session
        self.session = aiohttp.ClientSession()
        
        # Handle model-specific API URL if provided
        if "api_url" in model.get("load_options", {}):
            self.base_url = model['load_options'].get("api_url")
            print(f"[YandexBackend] Using model-specific API URL: {self.base_url}")
    
    async def unload_model(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
    
    def _convert_messages(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI message format to Yandex format."""
        yandex_messages = []
        
        for msg in conversation:
            yandex_msg = {
                "role": msg["role"]
            }
            
            # In Yandex API, content is called "text"
            if "content" in msg:
                yandex_msg["text"] = msg["content"]
                
            yandex_messages.append(yandex_msg)
            
        return yandex_messages

    async def chat_completion(
        self, 
        conversation: List[Dict[str, Any]], 
        stream: bool = False, 
        tools = [],
        max_tokens: int = 500, 
        params: GenerationParams = PRECISE_PARAMS, 
        mm_processor_kwargs={}, 
        extra={}
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Use Yandex API for chat completions with generation parameters.
        Supports both streaming and non-streaming responses.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.config['api_key']}",
            "x-folder-id": self.config.get("folder_id", ""),
        }
        
        # Create Yandex request
        request_body = {
            "modelUri": self.model_config.get("api_name"),
            "completionOptions": {
                "stream": stream,
                "temperature": params.get("temperature", 0.1),
                "maxTokens": str(max_tokens),
            },
            "messages": self._convert_messages(conversation)
        }
        
        # Add reasoning_options_mode if provided
        if "reasoning_options_mode" in extra:
            request_body["completionOptions"]["reasoningOptions"] = {
                "mode": extra["reasoning_options_mode"]
            }
        
        # Add json_schema if provided
        if "json_schema" in extra:
            request_body["jsonSchema"] = {
                "schema": extra["json_schema"]
            }

        try:
            if stream:
                # Return an async generator that yields chunks
                async def response_generator():
                    async with self.session.post(
                        self.base_url,
                        json=request_body,
                        headers=headers,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Yandex API error: {response.status}, {error_text}")
                        
                        # Process streaming response
                        async for line in response.content:
                            line = line.strip()
                            if not line or line == b'data: [DONE]':
                                continue
                                
                            if line.startswith(b'data: '):
                                try:
                                    data = json.loads(line[6:].decode('utf-8'))
                                    yield self._map_yandex_chunk_to_openai(data)
                                except json.JSONDecodeError:
                                    print(f"Failed to decode JSON: {line}")
                                    continue

                return response_generator()
            else:
                # Non-streaming request
                async with self.session.post(
                    self.base_url,
                    json=request_body,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Yandex API error: {response.status}, {error_text}")
                    
                    yandex_response = await response.json()
                    print(yandex_response)
                    return self._map_yandex_response_to_openai(yandex_response)
                    
        except Exception as e:
            print(f"ERROR in chat_completion: {type(e).__name__}: {str(e)}")
            raise
    
    def _map_yandex_response_to_openai(self, yandex_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Yandex response format to OpenAI response format."""
        # Extract the result object which contains the actual response data
        result = yandex_response.get("result", {})
        
        openai_response = {
            "id": f"yandex-{result.get('modelVersion', 'unknown')}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": self.model_config.get("api_name", "yandex-model"),
            "choices": [],
            "usage": {}
        }
        
        # Map usage information
        if "usage" in result:
            openai_response["usage"] = {
                "prompt_tokens": int(result["usage"].get("inputTextTokens", 0)),
                "completion_tokens": int(result["usage"].get("completionTokens", 0)),
                "total_tokens": int(result["usage"].get("totalTokens", 0))
            }
        
        # Map alternatives to choices
        if "alternatives" in result and result["alternatives"]:
            for i, alt in enumerate(result["alternatives"]):
                if "message" in alt:
                    choice = {
                        "index": i,
                        "message": {
                            "role": alt["message"].get("role", "assistant"),
                            "content": alt["message"].get("text", "")
                        },
                        "finish_reason": "stop" if alt.get("status") == "ALTERNATIVE_STATUS_FINAL" else "incomplete"
                    }
                    openai_response["choices"].append(choice)
                    
        return openai_response

    def _map_yandex_chunk_to_openai(self, yandex_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Yandex streaming chunk to OpenAI chunk format."""
        # Extract the result object which contains the actual chunk data
        result = yandex_chunk.get("result", {})
        
        openai_chunk = {
            "id": f"yandex-{result.get('modelVersion', 'unknown')}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": self.model_config.get("api_name", "yandex-model"),
            "choices": []
        }
        
        # Map alternatives to choices
        if "alternatives" in result and result["alternatives"]:
            for i, alt in enumerate(result["alternatives"]):
                if "message" in alt:
                    choice = {
                        "index": i,
                        "delta": {},
                        "finish_reason": None
                    }
                    
                    if "text" in alt["message"]:
                        choice["delta"]["content"] = alt["message"]["text"]
                    
                    if "status" in alt:
                        if alt["status"] == "ALTERNATIVE_STATUS_FINAL":
                            choice["finish_reason"] = "stop"
                        elif alt["status"] != "PENDING":
                            choice["finish_reason"] = "incomplete"
                        
                    openai_chunk["choices"].append(choice)
                
        return openai_chunk
    
    async def completion(
        self, 
        prompt: str,  
        stream: bool = False, 
        max_tokens: int = 500, 
        params: GenerationParams = PRECISE_PARAMS
    ) -> Union[str, AsyncIterator]:
        """
        Not implemented as per requirements.
        """
        raise NotImplementedError("completion() is not implemented for YandexBackend")
    
    async def _get_pid(self) -> Optional[int]:
        return None