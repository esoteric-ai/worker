from typing import Any, Dict, List
from backends.base import Backend
from backends.generation_params import PRECISE_PARAMS, GenerationParams


from mistral_common.protocol.instruct.messages import (
    UserMessage,
    AssistantMessage,
    SystemMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    Tool,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


class TekkenV7:
    
    def __init__(self, backend :Backend):
        self.backend = backend
    
    
    async def chat_completion(self, conversation: List[Dict[str, Any]], params: GenerationParams = PRECISE_PARAMS) -> str:
        tokenizer = MistralTokenizer.v7()
        
        messages = []
        for msg in conversation:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(UserMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AssistantMessage(content=msg["content"]))
        
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
        
        stream = await self.backend.completion(text, True, 500)
        
        result = ""
        
        async for event in stream:
            print("CHUNK RECEIVED")
            print(event.choices[0].text)
            result += event.choices[0].text
        
        print(result)
        
        return result