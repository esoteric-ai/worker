from typing import Any, Dict, TypedDict

class GenerationParams(TypedDict):
    temperature: float = 1.0
    max_tokens: int = 700
    
    top_p: float = 0.95 
    top_k: int = 40
    min_p: float = 0.05
    
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

# Actual default values as dictionaries
DEFAULT_PARAMS: Dict[str, Any] = {
    "temperature": 1.0,
    "max_tokens": 700,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

PRECISE_PARAMS: Dict[str, Any] = {
    "temperature": 0.1,
    "max_tokens": 700,
    "top_p": 1.0,
    "top_k": 1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}