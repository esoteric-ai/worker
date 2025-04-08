import asyncio
import os
import subprocess
import sys
from typing import AsyncIterator, List, Dict, Any, Literal, Optional, TypedDict, Union

from backends.base import Backend, ModelConfig
from backends.generation_params import GenerationParams, PRECISE_PARAMS

from openai import AsyncOpenAI
import httpx

class VllmBackendConfig(TypedDict):
    base_url: str = "http://127.0.0.1:8000"
    api_key: Optional[str] = None

class VllmBackend(Backend):
    def __init__(
        self,
        config: VllmBackendConfig,
    ):
        self.config = config
        
        self.openai_client = AsyncOpenAI(
            api_key=self.config.get("api_key") or "-",
            base_url=f"{self.config.get('base_url')}/v1",
        )
        
        self.running = False
        self.process = None
        self.active_model = None

    async def _get_pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    async def get_type(self) -> Literal["Managed", "Instant"]:
        print("Returning backend type: Managed")
        return "Managed"
    
    async def load_model(self, model: ModelConfig) -> None:
        if self.running:
            await self.unload_model()
        
        api_name = model.get("api_name")
        if not api_name:
            raise ValueError("Model API name is required")
            
        load_options = model.get("load_options", {})
        run_path = load_options.get("run_path")
        run_arguments = load_options.get("run_arguments", "")
        environment = load_options.get("environment", {})
        
        if not run_path:
            raise ValueError("run_path is required in load_options")
        
        try:
            cmd = [run_path] + (run_arguments.split() if run_arguments else [])
            print(cmd)
            kwargs = {'env': {**environment}} if environment else {'env': os.environ}
            
            # On Unix systems, create a new process group
            if sys.platform != "win32":
                import os
                kwargs['preexec_fn'] = os.setsid
            
            self.process = subprocess.Popen(
                cmd,
                
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                
                **kwargs
            )
            
            print(f"[VllmBackend] Started VLLM process with PID {self.process.pid}")
            
            # Wait 30 seconds for the model to load
            print(f"[VllmBackend] Waiting 30 seconds for model {api_name} to load...")
            await asyncio.sleep(30)
            
            if self.process.poll() is not None:
                raise RuntimeError(f"Failed to start VLLM process")
                
            self.running = True
            self.active_model = model
            print(f"[VllmBackend] Successfully loaded model: {api_name}")
            
        except Exception as e:
            self.process = None
            raise RuntimeError(f"Failed to start VLLM backend: {str(e)}")
    
    async def unload_model(self) -> None:
        if not self.running:
            print("[VllmBackend] Backend is not running, no model to unload")
            return
            
        if not self.active_model:
            print("[VllmBackend] No active model to unload")
            return

        api_name = self.active_model.get("api_name")
        try:
            if sys.platform == "win32":
                # On Windows, terminate the process
                self.process.terminate()
            else:
                # On Unix, kill the entire process group
                import signal
                import os
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except OSError:
                    # Fallback to just the process if getting the process group fails
                    self.process.send_signal(signal.SIGTERM)
                    
            # Wait for a short time for graceful shutdown
            try:
                # Run the blocking wait() in a thread to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.process.wait),
                    timeout=2.5
                )
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown fails
                if sys.platform == "win32":
                    self.process.kill()
                else:
                    try:
                        pgid = os.getpgid(self.process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        self.process.kill()
                    
            print(f"[VllmBackend] Process stopped")
            self.process = None
            self.running = False
            self.active_model = None
            
        except Exception as e:
            print(f"[VllmBackend] Error stopping process: {str(e)}")
            self.process = None
            self.running = False
            self.active_model = None

    async def start(self):
        # No-op since VLLM is started in load_model
        pass
    
    async def stop(self):
        # Unload the model if it's loaded
        await self.unload_model()

    async def chat_completion(
        self, 
        conversation: List[Dict[str, Any]], 
        stream: bool = False, 
        tools = [],
        max_tokens: int = 500, 
        params: GenerationParams = PRECISE_PARAMS
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Use the shared OpenAI client for chat completions with generation parameters.
        The API call is made using the OpenAI-compatible VLLM API.
        """
        if not self.active_model:
            raise ValueError("No active model loaded. Call load_model() first.")

        request_body = {
            "model": self.active_model.get("api_name"),
            "messages": conversation,
            "stream": stream,
        }

        extra_body = {
            "temperature": params.get("temperature", 0.1),
            "max_tokens": max_tokens,
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", 1),
            "repetition_penalty": params.get("repetition_penalty", 1.0),
            "frequency_penalty": params.get("frequency_penalty", 0.0),
            "presence_penalty": params.get("presence_penalty", 0.0),
        }
        try:
            if stream:
                async def response_generator():
                    stream_response = await self.openai_client.chat.completions.create(
                        **request_body,
                        extra_body=extra_body,
                    )

                    async for chunk in stream_response:
                        yield chunk.model_dump()

                return response_generator()
            else:
                response = await self.openai_client.chat.completions.create(
                    **request_body,
                    extra_body=extra_body,
                )

                return response.model_dump()
            
        except Exception as e:
            print(f"ERROR in chat_completion: {type(e).__name__}: {str(e)}")
            raise

    async def completion(
        self, 
        prompt: str, 
        stream: bool = False, 
        max_tokens: int = 100, 
        params: GenerationParams = PRECISE_PARAMS
    ) -> Union[str, AsyncIterator]:
        if not self.active_model:
            raise ValueError("No active model loaded. Call load_model() first.")

        request_body = {
            "model": self.active_model.get("api_name"),
            "prompt": prompt,
            "stream": stream,
            "max_tokens": max_tokens
        }

        extra_body = {
            "temperature": params.get("temperature", 0.1),
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", 1),
            "repetition_penalty": params.get("repetition_penalty", 1.0),
            "frequency_penalty": params.get("frequency_penalty", 0.0),
            "presence_penalty": params.get("presence_penalty", 0.0),
        }

        try:
            response = await self.openai_client.completions.create(
                **request_body,
                extra_body=extra_body,
            )

            if stream:
                async def response_generator():
                    async for chunk in response:
                        yield chunk.model_dump()

                return response_generator()
            else:
                return response.model_dump()

        except Exception as e:
            print(f"ERROR in completion: {type(e).__name__}: {str(e)}")
            raise