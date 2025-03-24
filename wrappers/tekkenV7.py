from typing import Any, Dict, List, AsyncIterator, Literal, Optional, Union
import json
import re
import time
import uuid
import asyncio
from backends.base import Backend, ModelConfig
from backends.generation_params import PRECISE_PARAMS, GenerationParams


from mistral_common.protocol.instruct.messages import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function, Tool, ToolCall, FunctionCall
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
    
    async def chat_completion(self, conversation: List[Dict[str, Any]], stream: bool = False, tools = [], max_tokens: int = 500, params: GenerationParams = PRECISE_PARAMS) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        tokenizer = MistralTokenizer.v7()

        messages = []
        for msg in conversation:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(UserMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Handle assistant messages with potential tool_calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    tool_calls = []
                    for tool_call in msg["tool_calls"]:
                        tool_calls.append(
                            ToolCall(
                                id=tool_call["id"],
                                function=FunctionCall(
                                    name=tool_call["function"]["name"],
                                    arguments=tool_call["function"]["arguments"]
                                )
                            )
                        )
                    messages.append(AssistantMessage(
                        content=msg.get("content"),
                        tool_calls=tool_calls
                    ))
                else:
                    messages.append(AssistantMessage(content=msg.get("content")))
            elif msg["role"] == "tool":
                # Handle tool messages with tool_call_id
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError("tool_call_id is required for tool messages")

                messages.append(ToolMessage(
                    tool_call_id=tool_call_id,
                    name=msg.get("name"),  # Add name if available
                    content=msg["content"]
                ))

        # Convert the passed tools to the correct format
        formatted_tools = []
        for tool in tools:
            if "function" in tool:
                function = tool["function"]
                formatted_tools.append(
                    Tool(
                        function=Function(
                            name=function["name"],
                            description=function.get("description", ""),
                            parameters=function.get("parameters", {})
                        )
                    )
                )

        # Tokenize a list of messages with the provided tools
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                tools=formatted_tools,
                messages=messages,
                model=params.get("model") # Add model from params or default
            )
        )
        text = tokenized.text
        print("Formatted prompt: " + str(text))

        if stream:
            # Return the async generator directly
            return await self._create_stream_iterator(text, max_tokens, params)
        else:
            return await self._non_stream_chat_completion(text, max_tokens, params)
    
    async def _non_stream_chat_completion(self, prompt: str, max_tokens, params: GenerationParams) -> Dict[str, Any]:
        """Handle non-streaming completion requests."""
        response = await self.backend.completion(prompt, stream=False, max_tokens=max_tokens, params=params)

        # Get the completion text and check finish reason
        completion_text = self._replace_special_chars(response.choices[0].text)
        finish_reason = response.choices[0].finish_reason

        # If finish reason is length, return text without trying to parse tool calls
        if finish_reason == "length":
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
                            "content": completion_text,
                            "tool_calls": None
                        },
                        "logprobs": None,
                        "finish_reason": "length"
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
    
    async def _create_stream_iterator(self, prompt: str, max_tokens, params: GenerationParams) -> AsyncIterator[Dict[str, Any]]:
        """Create and return an async iterator for streaming responses."""
        async def stream_generator():
            stream = await self.backend.completion(prompt, stream=True, max_tokens=max_tokens, params=params)

            accumulated_text = ""
            content_complete = False
            tool_calls_sent = False
            tool_calls_text = ""

            async for event in stream:
                # Check if finish reason is 'length'
                if event.choices[0].finish_reason == "length":
                    # Immediately return with finish reason 'length' and stop streaming
                    yield {
                        "model": "gpt-4o-mini",  # Placeholder
                        "choices": [
                            {
                                "delta": {
                                    "content": self._replace_special_chars(event.choices[0].text),
                                    "function_call": None,
                                    "role": None,
                                    "tool_calls": None
                                },
                                "finish_reason": "length",
                                "index": 0
                            }
                        ]
                    }
                    return  # Stop generating more chunks

                chunk = self._replace_special_chars(event.choices[0].text)
                accumulated_text += chunk

                # Check if we have reached the tool calls marker
                if not content_complete and "[TOOL_CALLS]" in accumulated_text:
                    # Send all text before the tool calls marker as content
                    content_parts = accumulated_text.split("[TOOL_CALLS]", 1)

                    # Mark content as complete
                    content_complete = True

                    # Start collecting tool calls
                    tool_calls_text = "[TOOL_CALLS]" + content_parts[1]
                elif not content_complete:
                    # Still collecting content
                    yield {
                        "created": int(time.time()),
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
    
    async def benchmark_model(self, model: ModelConfig) -> ModelConfig:
        await self.backend.benchmark_model(model)
    
    async def get_type(self) -> Literal["Managed", "Instant"]:
        print("Returning backend type passthrough")
        await self.backend.get_type()

    async def load_model(self, model: ModelConfig) -> None:
        await self.backend.load_model(model)

    async def unload_model(self) -> None:
        await self.backend.unload_model()
    
    async def completion(self, prompt: str,  stream: bool = False, params: GenerationParams = PRECISE_PARAMS) -> Union[str, AsyncIterator]:
        await self.backend.completion(prompt, stream, params)
    
    async def _get_pid(self) ->  Optional[int]:
        await self.backend._get_pid()