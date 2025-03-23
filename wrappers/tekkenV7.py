from typing import Any, Dict, List, AsyncIterator, Optional, Union
import json
import re
import time
import uuid
import asyncio
from backends.base import Backend
from backends.generation_params import PRECISE_PARAMS, GenerationParams


from mistral_common.protocol.instruct.messages import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    Tool,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


class TekkenV7:
    
    def __init__(self, backend: Backend):
        self.backend = backend
    
    def _replace_special_chars(self, text: str) -> str:
        """Replace '▁' with two spaces in the text."""
        return text.replace('▁', '  ')
    
    def _parse_tool_calls(self, tool_calls_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from the text format to dictionary objects."""
        # Clean up the text and replace the special characters
        clean_text = self._replace_special_chars(tool_calls_text)
        
        # Extract the JSON array from the [TOOL_CALLS] marker
        match = re.search(r'\[TOOL_CALLS\](.*?)(?:\n\n|$)', clean_text, re.DOTALL)
        if not match:
            return []
            
        try:
            # Parse the JSON array
            tool_calls_json = json.loads(match.group(1))
            
            # If it's a single object, wrap it in a list
            if isinstance(tool_calls_json, dict):
                tool_calls_json = [tool_calls_json]
                
            # Create ToolCall dictionaries
            result = []
            for idx, call in enumerate(tool_calls_json):
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                
                # Ensure arguments is a JSON string
                arguments = call.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, indent=2)
                
                result.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": arguments
                    }
                })
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            return []
    
    def _split_content_and_tool_calls(self, text: str) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Split the completion text into content and tool calls."""
        # Replace special characters
        text = self._replace_special_chars(text)
        
        # Check if there are tool calls
        if "[TOOL_CALLS]" in text:
            # Split the text at the tool calls marker
            content_parts = text.split("[TOOL_CALLS]", 1)
            content = content_parts[0].strip() if content_parts[0].strip() else None
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(text)
            return content, tool_calls
        else:
            # No tool calls, just content
            return text.strip() if text.strip() else None, []
    
    async def chat_completion(self, conversation: List[Dict[str, Any]], params: GenerationParams = PRECISE_PARAMS, stream: bool = False) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        tokenizer = MistralTokenizer.v7()
        
        messages = []
        for msg in conversation:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(UserMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AssistantMessage(content=msg["content"]))
            elif msg["role"] == "tool":
                messages.append(ToolMessage(content=msg["content"]))
        
        # Tokenize a list of messages
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                tools=[
                    Tool(
                        function=Function(
                            name="get_current_weather",
                            description="Get the current weather",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "format": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                        "description": "The temperature unit to use. Infer this from the user's location.",
                                    },
                                },
                                "required": ["location", "format"],
                            },
                        )
                    )
                ],
                messages=messages,
            )
        )
        text = tokenized.text
        print("Formatted prompt: " + str(text))
        
        if stream:
            # Return the async generator directly
            return self._create_stream_iterator(text)
        else:
            return await self._non_stream_chat_completion(text)
    
    async def _non_stream_chat_completion(self, prompt: str) -> Dict[str, Any]:
        """Handle non-streaming completion requests."""
        response = await self.backend.completion(prompt, stream=False, max_tokens=500)
        
        # Get the completion text from the response
        completion_text = self._replace_special_chars(response.choices[0].text)
        
        # Parse content and tool calls
        content, tool_calls = self._split_content_and_tool_calls(completion_text)
        
        # Determine finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        
        # Create the response JSON
        return {
            "id": f"chatcmpl_{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o-mini",  # Placeholder, use actual model name if available
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls if tool_calls else None
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Placeholder, use actual token count if available
                "completion_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            }
        }
    
    async def _create_stream_iterator(self, prompt: str) -> AsyncIterator[Dict[str, Any]]:
        """Create and return an async iterator for streaming responses."""
        async def stream_generator():
            stream = await self.backend.completion(prompt, stream=True, max_tokens=500)
            
            accumulated_text = ""
            content_complete = False
            tool_calls_sent = False
            tool_calls_text = ""
            
            # Generate a consistent ID for all chunks in this stream
            chat_id = f"chatcmpl_{uuid.uuid4().hex[:8]}"
            
            async for event in stream:
                chunk = self._replace_special_chars(event.choices[0].text)
                accumulated_text += chunk
                
                # Check if we have reached the tool calls marker
                if not content_complete and "[TOOL_CALLS]" in accumulated_text:
                    # Send all text before the tool calls marker as content
                    content_parts = accumulated_text.split("[TOOL_CALLS]", 1)
                    content_text = content_parts[0].strip()
                    
                    # Only yield content if there's actually content
                    if content_text:
                        yield {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "gpt-4o-mini",  # Placeholder
                            "choices": [
                                {
                                    "delta": {
                                        "content": content_text,
                                        "function_call": None,
                                        "role": "assistant",
                                        "tool_calls": None
                                    },
                                    "finish_reason": None,
                                    "index": 0
                                }
                            ]
                        }
                    
                    # Mark content as complete
                    content_complete = True
                    
                    # Start collecting tool calls
                    tool_calls_text = "[TOOL_CALLS]" + content_parts[1]
                elif not content_complete:
                    # Still collecting content
                    yield {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "gpt-4o-mini",  # Placeholder
                        "choices": [
                            {
                                "delta": {
                                    "content": chunk,
                                    "function_call": None,
                                    "role": "assistant",
                                    "tool_calls": None
                                },
                                "finish_reason": None,
                                "index": 0
                            }
                        ]
                    }
                else:
                    # Continue collecting tool calls text
                    tool_calls_text += chunk
                    
                    # Try to parse tool calls to see if they're complete
                    tool_calls = self._parse_tool_calls(tool_calls_text)
                    
                    # If we have valid tool calls and haven't sent them yet
                    if tool_calls and not tool_calls_sent:
                        yield {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "gpt-4o-mini",  # Placeholder
                            "choices": [
                                {
                                    "delta": {
                                        "content": "",
                                        "function_call": None,
                                        "role": "assistant",
                                        "tool_calls": tool_calls
                                    },
                                    "finish_reason": None,
                                    "index": 0
                                }
                            ]
                        }
                        tool_calls_sent = True
            
            # Final chunk to signal completion
            finish_reason = "tool_calls" if tool_calls_sent else "stop"
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4o-mini",  # Placeholder
                "choices": [
                    {
                        "delta": {
                            "content": None,
                            "function_call": None,
                            "role": None,
                            "tool_calls": None
                        },
                        "finish_reason": finish_reason,
                        "index": 0
                    }
                ]
            }
        
        # Return the async generator
        return stream_generator()