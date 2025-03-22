# backends/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Any, Optional, TypedDict, TYPE_CHECKING

from backends.generation_params import GenerationParams, PreciseParams

if TYPE_CHECKING:
    from backends.benchmark import benchmark_model_implementation

class ModelLoadConfig(TypedDict):
    num_gpu_layers: int = 0
    gpu_split: List[int] = [1]

class ModelPerformanceMetrics(TypedDict):
    parallel_requests: int = 1
    ram_requirement: int = 8000
    vram_requirement: List[int] = [8000]
    benchmark_results: Dict[str, Any] = {}

class ModelConfig(TypedDict):
    alias: str = None
    backend: str = None
    quant: str = None
    context_length: int = 8192
    
    api_name: str

    load_options: ModelLoadConfig
    performance_metrics: Optional[ModelPerformanceMetrics] = None
    
    
class Backend(ABC):
    
    @abstractmethod
    async def benchmark_model(self, model: ModelConfig) -> ModelConfig:
        return await benchmark_model_implementation(self, model)
    
    @abstractmethod
    async def get_type(self) -> Literal["Managed", "Instant"]:
        pass
    
    @abstractmethod
    async def benchmark_model(self, model: ModelConfig) -> ModelConfig:
        pass

    @abstractmethod
    async def load_model(self, model: ModelConfig) -> None:
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        pass
    
    @abstractmethod
    async def completion(self, prompt: str, params: GenerationParams = PreciseParams) -> str:
        pass

    @abstractmethod
    async def chat_completion(self, conversation: List[Dict[str, Any]], params: GenerationParams = PreciseParams) -> str:
        pass
    
    @abstractmethod
    async def _get_pid(self) ->  Optional[int]:
        pass
