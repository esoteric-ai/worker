from typing import TypedDict

class GenerationParams(TypedDict):
    temperature: float = 1.0
    max_tokens: int = 700
    
    top_p: float = 0.95 
    top_k: int = 40
    min_p: float = 0.05
    
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class PreciseParams(GenerationParams):
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: int = 1
    min_p: float = 0.0