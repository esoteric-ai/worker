from backends.base import Backend, ModelConfig, GenerationParams, PRECISE_PARAMS
from typing import AsyncIterator, List, Dict, Literal, Any, Optional, Union, TypedDict, AsyncGenerator
import base64
import aiohttp
import os
import uuid
import json

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from gigachat.models import Function, FunctionParameters, FunctionCall

from http.client import HTTPConnection


class GigaChatBackendConfig(TypedDict):
    api_key: str

class GigaChatBackend(Backend):
    def __init__(self, config: GigaChatBackendConfig):
        self.config = config
        self.temp_files = []  # Track temporary files
    
    async def get_type(self) -> Literal["Managed", "Instant"]:
        return "Instant"
    
    async def load_model(self, model: ModelConfig) -> None:
        self.model_config = model
        self.client = GigaChat(
            model=model['api_name'],
            credentials=self.config["api_key"],
            verify_ssl_certs=False
        )
    
    async def unload_model(self) -> None:
        if self.client:
            if hasattr(self.client, 'close') and callable(getattr(self.client, 'close')):
                self.client.close()
            self.client = None
        
        # Clean up any remaining temp files
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Clean up all temporary files created during this session"""
        for temp_path in self.temp_files:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        self.temp_files = []
        
    async def completion(self, prompt: str, stream: bool = False, max_tokens: int = 500, params: GenerationParams = PRECISE_PARAMS) -> Union[str, AsyncIterator]:
        raise NotImplementedError("GigaChatBackend only supports chat completion")
    
    async def _download_file(self, url: str, file_extension: str = "jpg"):
        """Download a file from a URL and return the path to the temporary file"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download file from {url}, status: {response.status}")
                
                # Create a temporary file
                unique_id = str(uuid.uuid4())[:8]
                temp_path = f"temp_{unique_id}.{file_extension}"
                
                with open(temp_path, "wb") as f:
                    f.write(await response.read())
                
                # Track this temp file for later cleanup
                self.temp_files.append(temp_path)
                return temp_path
    
    async def _upload_file(self, file_data: str, file_name: str):
        """Upload a base64 encoded file to GigaChat and return the file ID"""
        # Decode the base64 data
        binary_data = base64.b64decode(file_data)
        
        # Create a temporary file with unique ID to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        temp_path = f"temp_{unique_id}_{file_name}"
        with open(temp_path, "wb") as f:
            f.write(binary_data)
        
        # Track this temp file for later cleanup
        self.temp_files.append(temp_path)
        
        # Upload file to GigaChat
        file_id = await self.client.aupload_file(open(temp_path, "rb"))
        return file_id

    def _convert_openai_tools_to_gigachat_functions(self, tools: List[Dict[str, Any]]) -> List[Function]:
        """Convert OpenAI-style tools to GigaChat functions"""
        functions = []
        
        for tool in tools:
            if tool.get("type") != "function":
                continue
                
            function_data = tool.get("function", {})
            
            # Convert parameters to GigaChat format
            params_data = function_data.get("parameters", {})
            properties = {}
            
            # Process properties if they exist
            if "properties" in params_data:
                for prop_name, prop_details in params_data.get("properties", {}).items():
                    # Create FunctionParametersProperty for each property
                    property_data = {
                        "type": prop_details.get("type"),
                        # Add a default description if none is provided
                        "description": prop_details.get("description") or f"Parameter: {prop_name}"
                    }
                    
                    # Handle enum if present
                    if "enum" in prop_details:
                        property_data["enum"] = prop_details["enum"]
                    
                    # Handle items for array types
                    if prop_details.get("type") == "array" and "items" in prop_details:
                        property_data["items"] = prop_details["items"]
                        
                    properties[prop_name] = property_data
            
            # Create function parameters
            parameters = FunctionParameters(
                type=params_data.get("type", "object"),
                properties=properties,
                required=params_data.get("required", [])
            )
            
            # Create function
            function = Function(
                name=function_data.get("name", ""),
                description=function_data.get("description", ""),
                parameters=parameters
            )
            
            functions.append(function)
            
        return functions

    def _convert_gigachat_function_call_to_openai(self, function_call):
        """Convert GigaChat function call to OpenAI tool call format"""
        if not function_call:
            return None
            
        return {
            "id": str(uuid.uuid4()),
            "type": "function",
            "function": {
                "name": function_call.name,
                "arguments": function_call.arguments
            }
        }

    async def chat_completion(self, conversation: List[Dict[str, Any]], stream: bool = False, tools = [], max_tokens: int = 500, params: GenerationParams = PRECISE_PARAMS, mm_processor_kwargs={}, extra={}) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        if not self.client:
            raise RuntimeError("GigaChat client not initialized")
        
        # Convert conversation to GigaChat format
        try:
            gigachat_messages = []
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")
                name = msg.get("name")
                
                # Map roles to GigaChat format
                if role == "system":
                    gigachat_role = MessagesRole.SYSTEM
                elif role == "user":
                    gigachat_role = MessagesRole.USER
                elif role == "assistant":
                    gigachat_role = MessagesRole.ASSISTANT
                elif role == "tool":
                    gigachat_role = MessagesRole.FUNCTION
                else:
                    gigachat_role = MessagesRole.USER
                
                # Handle attachments if present (in content array format)
                attachment_ids = None
                if isinstance(content, list):
                    text_parts = []
                    attachment_ids = []
                    
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                image_url = part.get("image_url", {})
                                if isinstance(image_url, dict) and "url" in image_url:
                                    url = image_url["url"]
                                    # Check if it's a base64 data URL
                                    if url.startswith("data:image/"):
                                        format_type = url.split(";")[0].split("/")[1]
                                        base64_data = url.split(",")[1]
                                        file_id = await self._upload_file(base64_data, f"image.{format_type}")
                                        attachment_ids.append(file_id.id_)
                                    # Handle regular URLs (http/https)
                                    elif url.startswith(("http://", "https://")):
                                        # Try to determine extension from URL or default to jpg
                                        extension = os.path.splitext(url)[1].lstrip(".") or "jpg"
                                        temp_path = await self._download_file(url, extension)
                                        file_id = await self.client.aupload_file(open(temp_path, "rb"))
                                        attachment_ids.append(file_id.id_)

                    # Join text parts as the content
                    content = "\n".join(text_parts) if text_parts else ""
                
                # Create message with attachments if any
                gigachat_message = Messages(role=gigachat_role, content=content)
                
                # Add name if provided
                if name:
                    gigachat_message.name = name
                    
                # Handle tool calls in assistant messages
                if role == "assistant" and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls and len(tool_calls) > 0:
                        # GigaChat only supports one function call per message
                        tool_call = tool_calls[0]
                        if tool_call.get("type") == "function":
                            function_data = tool_call.get("function", {})
                            arguments_str = function_data.get("arguments", "{}")
                            gigachat_message.function_call = FunctionCall(
                                name=function_data.get("name", ""),
                                arguments={}
                            )
                
                # Handle tool messages
                #if role == "tool":
                #    gigachat_message.tool_call_id = msg.get("tool_call_id")
                    
                if attachment_ids:
                    gigachat_message.attachments = attachment_ids
                    
                gigachat_messages.append(gigachat_message)
            
            # Convert OpenAI tools to GigaChat functions
            functions = self._convert_openai_tools_to_gigachat_functions(tools) if tools else None
            
            # Create payload
            payload = Chat(
                messages=gigachat_messages,
                max_tokens=max_tokens,
                functions=functions if functions else None
            )
            
            if not stream:
                # Non-streaming implementation
                response = await self.client.achat(payload)
                
                # Convert to OpenAI format
                openai_response = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content or ""
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }],
                    "id": str(uuid.uuid4()),
                    "created": int(response.created) if hasattr(response, "created") else int(os.path.getctime(__file__)),
                    "model": self.model_config['alias'],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                        "completion_tokens": response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0,
                        "total_tokens": response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 0
                    }
                }
                
                # Handle function/tool calls if present
                if hasattr(response.choices[0].message, "function_call") and response.choices[0].message.function_call:
                    func_call = response.choices[0].message.function_call
                    tool_call = {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": func_call.name,
                            "arguments": json.dumps(func_call.arguments) if isinstance(func_call.arguments, dict) else func_call.arguments
                        }
                    }
                    openai_response["choices"][0]["message"]["content"] = None
                    openai_response["choices"][0]["message"]["tool_calls"] = [tool_call]
                
                # Clean up temp files
                self._cleanup_temp_files()
                
                return openai_response
            else:
                # Streaming implementation
                async def response_generator():
                    try:
                        # Track if we've seen the function call
                        function_call_sent = False
                        
                        # Stream response
                        async for chunk in self.client.astream(payload):
                            delta = {}
                            
                            # Process content
                            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                delta["content"] = chunk.choices[0].delta.content
                            
                            # Process function call if present
                            if hasattr(chunk.choices[0].delta, 'function_call') and chunk.choices[0].delta.function_call and not function_call_sent:
                                # Convert to OpenAI format
                                tool_call = self._convert_gigachat_function_call_to_openai(chunk.choices[0].delta.function_call)
                                if tool_call:
                                    delta["tool_calls"] = [tool_call]
                                    function_call_sent = True  # Only send once for streaming
                            
                            yield {
                                "choices": [
                                    {
                                        "delta": delta,
                                        "finish_reason": chunk.choices[0].finish_reason if hasattr(chunk.choices[0], 'finish_reason') else None
                                    }
                                ]
                            }
                    finally:
                        # Clean up temp files after completion finishes
                        self._cleanup_temp_files()
                
                return response_generator()
        except Exception as e:
            self._cleanup_temp_files()
            raise e
    
    async def _get_pid(self) -> Optional[int]:
        return None