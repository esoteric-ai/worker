from typing import Any, Dict, List, AsyncIterator, Optional
import json
import re
import uuid
import time
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
    
    async def chat_completion(self, 
                             conversation: List[Dict[str, Any]], 
                             stream: bool = False,
                             params: GenerationParams = PRECISE_PARAMS) -> Any:
        """
        Process chat completion with support for streaming and tool calls.
        
        Args:
            conversation: List of conversation messages
            stream: Whether to stream the response
            params: Generation parameters
        
        Returns:
            Chat completion response in OpenAI format or AsyncIterator of chunks if streaming
        """
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

        # Count the number of tokens
        print("Formatted prompt: " + str(text))
        
        if stream:
            return self._stream_chat_completion(text, params)
        else:
            return await self._non_stream_chat_completion(text, params)
    
    async def _non_stream_chat_completion(self, 
                                         text: str, 
                                         params: GenerationParams) -> Dict[str, Any]:
        """
        Handle non-streaming chat completion.
        
        Args:
            text: The formatted prompt text
            params: Generation parameters
        
        Returns:
            Chat completion response in OpenAI format
        """
        max_tokens = 500
        response = await self.backend.completion(text, False, max_tokens)
        
        # Parse the completion text
        completion_text = response.choices[0].text
        
        # Check if there's a tool call in the response
        tool_call_match = re.search(r'(\[TOOL_CALLS\]▁)(.*?)(\[/TOOL_CALLS\])?', completion_text, re.DOTALL)
        
        chat_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "tekken-v7",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                    },
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0,
                "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else 0,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            }
        }
        
        if tool_call_match:
            # Extract the content before the tool call
            content_before_tool = completion_text[:tool_call_match.start()].strip()
            chat_response["choices"][0]["message"]["content"] = content_before_tool if content_before_tool else None
            
            # Extract and parse the tool call
            tool_call_text = tool_call_match.group(2).strip()
            try:
                tool_calls = json.loads(tool_call_text)
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]
                
                formatted_tool_calls = []
                for i, call in enumerate(tool_calls):
                    formatted_tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:6]}",
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(call.get("arguments", {}), indent=2)
                        }
                    })
                
                chat_response["choices"][0]["message"]["tool_calls"] = formatted_tool_calls
                chat_response["choices"][0]["finish_reason"] = "tool_calls"
                
                # If there's content after the tool call
                if tool_call_match.group(3) and tool_call_match.end() < len(completion_text):
                    content_after_tool = completion_text[tool_call_match.end():].strip()
                    if content_after_tool:
                        if chat_response["choices"][0]["message"]["content"] is None:
                            chat_response["choices"][0]["message"]["content"] = content_after_tool
                        else:
                            chat_response["choices"][0]["message"]["content"] += "\n" + content_after_tool
            except json.JSONDecodeError:
                # If we can't parse the tool call, treat it as regular content
                chat_response["choices"][0]["message"]["content"] = completion_text
                chat_response["choices"][0]["finish_reason"] = "stop"
        else:
            # No tool call, just regular content
            chat_response["choices"][0]["message"]["content"] = completion_text
            chat_response["choices"][0]["finish_reason"] = "stop"
        
        return chat_response
    
    async def _stream_chat_completion(self, 
                                    text: str, 
                                    params: GenerationParams) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat completion with support for tool calls.
        
        Args:
            text: The formatted prompt text
            params: Generation parameters
        
        Returns:
            AsyncIterator of chat completion chunks
        """
        max_tokens = 500
        stream = await self.backend.completion(text, True, max_tokens)
        
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        
        buffer = ""
        tool_call_buffer = ""
        in_tool_call = False
        tool_call_sent = False
        
        async for event in stream:
            chunk = event.choices[0].text
            
            if not in_tool_call:
                # Check if this chunk contains the start of a tool call
                tool_call_start = chunk.find("[TOOL_CALLS]▁")
                
                if tool_call_start >= 0:
                    # Send the content before tool_call as a regular content chunk
                    pre_tool_content = buffer + chunk[:tool_call_start]
                    if pre_tool_content:
                        yield {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "tekken-v7",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": pre_tool_content,
                                        "function_call": None,
                                        "tool_calls": None
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                    
                    # Switch to tool call mode
                    in_tool_call = True
                    tool_call_buffer = chunk[tool_call_start + len("[TOOL_CALLS]▁"):]
                    buffer = ""
                else:
                    # Regular content - add to buffer and send
                    buffer += chunk
                    if buffer:
                        yield {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "tekken-v7",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": buffer,
                                        "function_call": None,
                                        "tool_calls": None
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        buffer = ""
            else:
                # In tool call mode
                tool_call_end = chunk.find("[/TOOL_CALLS]")
                
                if tool_call_end >= 0:
                    # Tool call is complete
                    tool_call_buffer += chunk[:tool_call_end]
                    
                    try:
                        # Parse and send the tool call
                        tool_calls_data = json.loads(tool_call_buffer)
                        if not isinstance(tool_calls_data, list):
                            tool_calls_data = [tool_calls_data]
                        
                        formatted_tool_calls = []
                        for i, call in enumerate(tool_calls_data):
                            formatted_tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:6]}",
                                "type": "function",
                                "function": {
                                    "name": call.get("name", ""),
                                    "arguments": json.dumps(call.get("arguments", {}), indent=2)
                                }
                            })
                        
                        yield {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "tekken-v7",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "",
                                        "function_call": None,
                                        "tool_calls": formatted_tool_calls
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        tool_call_sent = True
                        
                        # Switch back to regular content mode for any remaining content
                        in_tool_call = False
                        buffer = chunk[tool_call_end + len("[/TOOL_CALLS]"):]
                        
                        # If there's content after the tool call, send it
                        if buffer:
                            yield {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "tekken-v7",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": buffer,
                                            "function_call": None,
                                            "tool_calls": None
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            buffer = ""
                    except json.JSONDecodeError:
                        # If we can't parse the tool call, treat it as regular content
                        in_tool_call = False
                        buffer = f"[TOOL_CALLS]▁{tool_call_buffer}{chunk}"
                        yield {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "tekken-v7",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": buffer,
                                        "function_call": None,
                                        "tool_calls": None
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        buffer = ""
                else:
                    # Still collecting the tool call
                    tool_call_buffer += chunk
        
        # End of stream, check if we have an uncompleted tool call
        if in_tool_call and not tool_call_sent:
            try:
                # Try to parse as JSON even without closing tag
                tool_calls_data = json.loads(tool_call_buffer)
                if not isinstance(tool_calls_data, list):
                    tool_calls_data = [tool_calls_data]
                
                formatted_tool_calls = []
                for i, call in enumerate(tool_calls_data):
                    formatted_tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:6]}",
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(call.get("arguments", {}), indent=2)
                        }
                    })
                
                yield {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "tekken-v7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "",
                                "function_call": None,
                                "tool_calls": formatted_tool_calls
                            },
                            "finish_reason": None
                        }
                    ]
                }
            except json.JSONDecodeError:
                # Failed to parse, send as regular content
                if tool_call_buffer:
                    yield {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "tekken-v7",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": f"[TOOL_CALLS]▁{tool_call_buffer}",
                                    "function_call": None,
                                    "tool_calls": None
                                },
                                "finish_reason": None
                            }
                        ]
                    }
        
        # Send the final chunk with finish_reason
        finish_reason = "tool_calls" if tool_call_sent else "stop"
        yield {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "tekken-v7",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }
            ]
        }