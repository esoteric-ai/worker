# backends/tabby.py
import asyncio
import os
import subprocess
import sys
from typing import List, Dict, Any, Optional, TypedDict

from backends.base import Backend, ModelConfig
from backends.generation_params import GenerationParams, PreciseParams

from openai import AsyncOpenAI
import httpx

class TabbyBackendConfig(TypedDict):
    base_url: str = "http://127.0.0.1/"
    api_key: Optional[str] = None
    run_path: str = None
    run_arguments: str = None
    environment: Optional[Dict[str, str]] = None
    

class TabbyBackend(Backend):
    def __init__(
        self,
        config: TabbyBackendConfig,
    ):
        self.config = config
        
        self.openai_client = AsyncOpenAI(
            api_key=self.config.get("api_key") or "-",
            base_url=f"{self.config.get('base_url')}/v1",
        )
        
        self.running = False
        self.pid = None
        self.active_model = None

    async def _get_pid(self) -> Optional[int]:
        return self.pid

    async def get_type(self) -> str:
        return "Managed"
    
    
    
    async def load_model(self, model: ModelConfig) -> None:
        if not self.running:
            await self.start()

        api_name = model.get("api_name")
        if not api_name:
            raise ValueError("Model API name is required")

        load_options = model.get("load_options", {})
        context_length = model.get("context_length", 8192)

        request_body = {
            "model_name": api_name,
            "max_seq_len": context_length,
        }

        if "gpu_split" in load_options:
            request_body["gpu_split"] = load_options["gpu_split"]

        try:
            headers = {"Content-Type": "application/json"}
            if self.config.get("api_key"):
                headers["Authorization"] = f"Bearer {self.config.get('api_key')}"
                headers["X-Admin-Key"] = self.config.get("api_key")

            async with httpx.AsyncClient(timeout=None) as client:
                base_url = self.config.get("base_url").rstrip("/")
                response = await client.post(
                    f"{base_url}/v1/model/load",
                    json=request_body,
                    headers=headers
                )
                response.raise_for_status()

            print(f"[TabbyBackend] Successfully loaded model: {api_name}")

            self.active_model = model
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model {api_name}: {str(e)}")
    
    async def unload_model(self) -> None:
        if not self.running:
            print("[TabbyBackend] Backend is not running, no model to unload")
            return

        try:
            headers = {"Content-Type": "application/json"}
            if self.config.get("api_key"):
                headers["Authorization"] = f"Bearer {self.config.get('api_key')}"
                headers["X-Admin-Key"] = self.config.get("api_key")

            async with httpx.AsyncClient(timeout=None) as client:
                base_url = self.config.get("base_url").rstrip("/")
                response = await client.post(
                    f"{base_url}/v1/model/unload",
                    headers=headers
                )
                response.raise_for_status()

            print(f"[TabbyBackend] Successfully unloaded model: {self.active_model.get("api_name")}")

            self.active_model = None

        except Exception as e:
            raise RuntimeError(f"Failed to unload model: {str(e)}")

    async def start(self):
        if self.running:
            return
        
        run_path = self.config.get("run_path")
        run_arguments = self.config.get("run_arguments", "")
        environment = self.config.get("environment", {})
        
        if not run_path:
            raise ValueError("run_path is required")
        
        try:
            cmd = [run_path] + (run_arguments.split() if run_arguments else [])
            
            env = os.environ.copy()
            if environment:
                env.update(environment)
            
            self.process = subprocess.Popen(
                cmd,
                # stdout=subprocess.PIPE, 
                # stderr=subprocess.PIPE,
                # shell=True,
                # text=True
                env=env
            )
            
            print(f"[TabbyBackend] Started backend process with PID {self.process.pid}")
            
            if self.process.poll() is not None:
                raise RuntimeError(f"Failed to start process: {self.process.stderr.read()}")
                
            self.running = True
        except Exception as e:
            self.process = None
            raise RuntimeError(f"Failed to start Tabby backend: {str(e)}")
    
    async def stop(self):
        if not self.running:
            return
        
        try:
            if sys.platform == "win32":
                self.process.terminate()
            else:
                import signal
                self.process.send_signal(signal.SIGTERM)
                
            # Wait for a short time for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        lambda: self.process.wait()
                    ), 
                    timeout=2.5
                )
            except asyncio.TimeoutError:
                self.process.kill()
            
                
            print(f"[TabbyBackend] Process stopped")
            self.process = None
            self.running = False
        except Exception as e:
            print(f"[TabbyBackend] Error stopping process: {str(e)}")
            self.process = None
            self.running = False


    async def chat_completion(self, conversation: List[Dict[str, Any]], params: GenerationParams = PreciseParams) -> str:
        """
        Use the shared OpenAI client for chat completions with generation parameters.
        The API call is made using the aliased API model name.
        """
        if not self.active_model:
            raise ValueError("No active model loaded. Call load_model() first.")

        request_body = {
            "model": self.active_model.get("api_name"),
            "messages": conversation,
        }

        extra_body = {
            "temperature": params.get("temperature", 0.1),
            "max_tokens": params.get("max_tokens", 700),
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", 1),
            "min_p": params.get("min_p", 0.0),
            "repetition_penalty": params.get("repetition_penalty", 1.0),
            "frequency_penalty": params.get("frequency_penalty", 0.0),
            "presence_penalty": params.get("presence_penalty", 0.0),
            
            # Not implemented yet parameters
            "penalty_range": -1,
            "top_a": 0.0,
            "temp_last": False,
            "typical": 1.0,
            "tfs": 1.0,
            "logit_bias": None,
            "mirostat_mode": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
        }
        
        extra_body = {k: v for k, v in extra_body.items() if v is not None}

        response = await self.openai_client.chat.completions.create(
            **request_body,
            extra_body=extra_body,
        )

        if not response.choices:
            return ""
        
        return response.choices[0].message.content

    async def completion(self, prompt: str, params: GenerationParams = PreciseParams) -> str:
        return ""