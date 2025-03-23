import asyncio
from typing import Dict, Any, List
from backends.tabby import TabbyBackend, TabbyBackendConfig
from backends.base import ModelConfig, ModelLoadConfig, ModelPerformanceMetrics

class TestTabbyBackend():
    def setUp(self):
        self.config = TabbyBackendConfig(
            base_url="http://127.0.0.1:5000",
            api_key="test_key",
            run_path="/path/to/tabby",
            run_arguments="",
            environment={"CUDA_VISIBLE_DEVICES": "0"}
        )
        self.backend = TabbyBackend(self.config)
        
    def tearDown(self):
        if self.backend.running:
            asyncio.run(self.backend.stop())
    
    async def test_start(self):
        model_config = ModelConfig(
            alias="test-model",
            backend="tabby",
            quant="4.0bpw",
            context_length=8192,
            api_name="test/model",
            load_options=ModelLoadConfig(
                num_gpu_layers=999,
                gpu_split=[1]
            )
        )
        
        self.backend.load_model(model_config)
        
        stream = await self.backend.completion("A quick brown fox", True, 300)
        
        for event in stream:
            print(event)
        
        
        self.backend.unload_model()
    
    def test_benchmark_model(self):
        
        
        asyncio.run(self.test_start())
        

if __name__ == "__main__":
    test = TestTabbyBackend()
    test.setUp()
    test.test_benchmark_model()
    test.tearDown()