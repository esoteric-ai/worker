# backends/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Backend(ABC):
    """
    Abstract base class for any LLM backend.
    """

    @abstractmethod
    async def load_model(self) -> str:
        """
        Load or retrieve the current model. 
        Return the model name or identifier.
        """
        pass

    @abstractmethod
    async def chat_completion(self, conversation: List[Dict[str, Any]]) -> str:
        """
        Given a conversation (list of messages), produce a completion text.
        Return the model's generated text.
        """
        pass
