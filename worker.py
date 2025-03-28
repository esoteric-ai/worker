import asyncio
import json
import uuid
import sys
import signal
from typing import AsyncIterator, List, Dict, Any, Optional, TypedDict, Union

import torch

import websockets
import httpx

from backends.base import Backend, ModelConfig, ModelLoadConfig, ModelPerformanceMetrics
from backends.generation_params import PRECISE_PARAMS
from backends.tabby import TabbyBackend, TabbyBackendConfig
from backends.ollama import OllamaBackend, OllamaBackendConfig
from wrappers.tekkenV7 import TekkenV7

class BackendInstance:
    """Represents a loaded backend instance with its model"""
    def __init__(self, backend: Backend, model_config: Dict[str, Any], 
                 wrapper_name: Optional[str] = None, real_backend: Optional[Backend] = None):
        self.backend = backend
        self.real_backend = real_backend
        self.model_config = model_config
        self.wrapper_name = wrapper_name
        self.gpu_allocation = model_config.get("performance_metrics", {}).get("vram_requirement", [])
        self.loaded_on_gpus = [i for i, vram in enumerate(self.gpu_allocation) if vram > 0]

    @property
    def model_alias(self) -> str:
        return self.model_config.get("alias", "unknown")



class WorkerClient:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.worker_name: str = cfg.get("worker_name", "my_worker")
        self.server_base_url: str = cfg.get("server_base_url", "http://localhost:8000")
        
        self.model_configs: List[ModelConfig] = cfg.get("model_configs", [])
        
         # Backend configurations
        self.backend_configs = {
            "TabbyAPI": TabbyBackendConfig(
                base_url=cfg.get("tabby_api_url", "http://127.0.0.1"),
                api_key=cfg.get("tabby_api_key"),
                run_path=cfg.get("tabby_run_path"),
                run_arguments=cfg.get("tabby_run_arguments"),
                environment=cfg.get("tabby_environment")
            ),
            "Ollama": OllamaBackendConfig(
                base_url=cfg.get("ollama_api_url", "http://127.0.0.1"),
                api_key=cfg.get("ollama_api_key"),
                run_path=cfg.get("ollama_run_path"),
                run_arguments=cfg.get("ollama_run_arguments"),
                environment=cfg.get("ollama_environment")
            )
        }
        
        # Track loaded models and backends
        self.loaded_backends: Dict[str, BackendInstance] = {}  # key: unique instance ID
        
        # Add tracking for active tasks per model
        self.model_active_tasks: Dict[str, int] = {}  # key: backend_instance_id, value: active task count

        self.worker_uid: Optional[str] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.pending_requests: Dict[str, asyncio.Future] = {}

        self.batch_size: int = cfg.get("batch_size", 20)
        self.buffer_size = self.batch_size

        # Remove async primitives initialization from __init__
        self.task_queue: Optional[asyncio.Queue] = None
        self.completed_queue: Optional[asyncio.Queue] = None
        self.tasks_available_event: Optional[asyncio.Event] = None
        self.ws_connected: Optional[asyncio.Event] = None

        self.processing_tasks = set()
        self.reconnect_schedule = [5] * 10 + [60] * 10 + [3600] * 24

    def _create_backend_instance(self, backend_type: str, model_config: Dict[str, Any]) -> BackendInstance:
        """Create a new backend instance based on type and potential wrapper"""
        wrapper_name = model_config.get("wrapper")
        
        if backend_type == "TabbyAPI":
            real_backend = TabbyBackend(self.backend_configs["TabbyAPI"])
            if wrapper_name == "TekkenV7":
                backend = TekkenV7(real_backend)
            else:
                backend = real_backend
                real_backend = None
        elif backend_type == "Ollama":
            real_backend = OllamaBackend(self.backend_configs["Ollama"])
            if wrapper_name == "TekkenV7":
                backend = TekkenV7(real_backend)
            else:
                backend = real_backend
                real_backend = None
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
            
        return BackendInstance(
            backend=backend,
            real_backend=real_backend,
            model_config=model_config,
            wrapper_name=wrapper_name
        )

    async def get_available_vram(self) -> List[int]:
        """Get available VRAM on each GPU in MB"""
        available_vram = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_vram = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
                used_vram = torch.cuda.memory_reserved(i) // (1024 * 1024)
                available_vram.append(total_vram - used_vram)
        return available_vram
    
    async def classify_models(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify models into hot (can be loaded alongside current models)
        and cold (require unloading current models)
        """
        available_vram = await self.get_available_vram()
        
        # Create a map of GPU index -> used VRAM by currently loaded models
        gpu_usage = {i: 0 for i in range(len(available_vram))}
        for backend_instance in self.loaded_backends.values():
            for gpu_idx, vram_needed in enumerate(backend_instance.gpu_allocation):
                if gpu_idx < len(gpu_usage) and vram_needed > 0:
                    gpu_usage[gpu_idx] += vram_needed
        
        hot_models = []
        cold_models = []
        
        # Check each model config if it can be loaded alongside current models
        for model_config in self.model_configs:
            vram_requirements = model_config.get("performance_metrics", {}).get("vram_requirement", [])
            
            # Extend vram_requirements if it's shorter than available_vram
            vram_requirements = vram_requirements + [0] * (len(available_vram) - len(vram_requirements))
            
            # Check if model is already loaded
            model_already_loaded = any(
                backend.model_alias == model_config.get("alias")
                for backend in self.loaded_backends.values()
            )
            
            can_load_parallel = True
            for gpu_idx, vram_needed in enumerate(vram_requirements):
                if gpu_idx >= len(available_vram):
                    break
                    
                if vram_needed > 0 and (available_vram[gpu_idx] - gpu_usage[gpu_idx]) < vram_needed:
                    can_load_parallel = False
                    break
            
            if can_load_parallel or model_already_loaded:
                hot_models.append(model_config)
            elif not model_already_loaded:
                cold_models.append(model_config)
                
        return hot_models, cold_models
    
    async def load_model_for_task(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Load the model required for this task if not already loaded.
        First checks if any model from task["models"] is already loaded.
        If not, loads the first compatible model from the list.
        Returns the backend instance ID for the model.
        """
        model_aliases = task.get("models", [])
        if not model_aliases:
            # Fall back to legacy model_alias field if present
            legacy_alias = task.get("model_alias")
            if legacy_alias:
                model_aliases = [legacy_alias]
            else:
                return None

        # First pass: check if any model is already loaded
        for model_alias in model_aliases:
            for instance_id, backend_instance in self.loaded_backends.items():
                if backend_instance.model_alias == model_alias:
                    print(f"[Worker] Using already loaded model: {model_alias}")
                    return instance_id

        # Second pass: try to load one of the models
        for model_alias in model_aliases:
            # Find model config
            model_config = None
            for config in self.model_configs:
                if config.get("alias") == model_alias:
                    model_config = config
                    break

            if not model_config:
                print(f"[Worker] Model {model_alias} not found in configurations")
                continue  # Try next model

            # Check if we need to unload models to make space
            vram_requirements = model_config.get("performance_metrics", {}).get("vram_requirement", [])
            available_vram = await self.get_available_vram()

            # Extend vram_requirements if it's shorter than available_vram
            vram_requirements = vram_requirements + [0] * (len(available_vram) - len(vram_requirements))

            # Check if we need to unload models
            gpus_to_free = []
            for gpu_idx, vram_needed in enumerate(vram_requirements):
                if gpu_idx >= len(available_vram):
                    break

                if vram_needed > 0 and available_vram[gpu_idx] < vram_needed:
                    gpus_to_free.append(gpu_idx)

            # Unload models that use GPUs we need to free
            if gpus_to_free:
                backends_to_unload = []
                for instance_id, backend_instance in self.loaded_backends.items():
                    for gpu_idx in gpus_to_free:
                        if gpu_idx in backend_instance.loaded_on_gpus:
                            backends_to_unload.append(instance_id)
                            break

                for instance_id in backends_to_unload:
                    await self.unload_backend(instance_id)

            # Load the model
            backend_type = model_config.get("backend", "TabbyAPI")
            backend_instance = self._create_backend_instance(backend_type, model_config)

            print(f"[Worker] Loading model: {model_config.get('alias')}")
            await backend_instance.backend.load_model(model_config)
            print(f"[Worker] Model loaded: {model_config.get('alias')}")

            # Generate unique instance ID
            instance_id = f"{model_alias}_{uuid.uuid4().hex[:8]}"
            self.loaded_backends[instance_id] = backend_instance

            return instance_id

        # If we get here, none of the models could be loaded
        print(f"[Worker] Could not load any model from {model_aliases}")
        return None
                    
    async def unload_backend(self, instance_id: str) -> bool:
        """Unload a backend instance by its ID"""
        if instance_id not in self.loaded_backends:
            return False
            
        backend_instance = self.loaded_backends[instance_id]
        try:
            if backend_instance.real_backend:
                await backend_instance.real_backend.unload_model()
            else:
                await backend_instance.backend.unload_model()
            print(f"[Worker] Unloaded model: {backend_instance.model_alias}")
            del self.loaded_backends[instance_id]
            # Remove from active tasks tracking
            if instance_id in self.model_active_tasks:
                del self.model_active_tasks[instance_id]
            return True
        except Exception as e:
            print(f"[Worker] Error unloading model {backend_instance.model_alias}: {str(e)}")
            return False

    async def producer_loop(self):
        """
        Periodically fetch new tasks (up to 'buffer_size')
        if our local queue has room, otherwise wait.
        """
        while True:
            fetched_any = False
            current_size = self.task_queue.qsize()
            if current_size < self.buffer_size:
                num_to_fetch = self.buffer_size - current_size
                tasks = await self.request_tasks(num_to_fetch)
                if tasks:
                    fetched_any = True
                    for task in tasks:
                        print("[Producer] Adding task to local queue:", task.get("id"))
                        await self.task_queue.put(task)
                    print(f"[Producer] Fetched {len(tasks)} tasks. Queue size: {self.task_queue.qsize()}")
            if not fetched_any:
                try:
                    await asyncio.wait_for(self.tasks_available_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    self.tasks_available_event.clear()

    async def get_suitable_model_for_task(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Find a suitable model for the task that is already loaded.
        Returns backend_instance_id if found, None otherwise.
        """
        model_aliases = task.get("models", [])

        # Check if any model is already loaded
        for model_alias in model_aliases:
            for instance_id, backend_instance in self.loaded_backends.items():
                if backend_instance.model_alias == model_alias:
                    return instance_id
                    
        return None

    async def consumer_loop(self):
        """
        Consumer that processes tasks respecting per-model parallel request limits.
        """
        while True:
            # Clean up completed tasks
            self.processing_tasks = {t for t in self.processing_tasks if not t.done()}

            # Check if we can process more tasks
            if len(self.processing_tasks) < self.batch_size:
                if self.task_queue.empty():
                    await asyncio.sleep(0.1)
                    continue

                # Create a temporary queue to hold skipped tasks
                temp_queue = asyncio.Queue()
                processed_any = False
                tasks_to_process = min(self.task_queue.qsize(), self.batch_size - len(self.processing_tasks))

                for _ in range(tasks_to_process):
                    if self.task_queue.empty():
                        break

                    task_data = await self.task_queue.get()

                    # First try to find a suitable model that's already loaded
                    model_id = await self.get_suitable_model_for_task(task_data)

                    # If no suitable model is loaded, try to load one
                    if not model_id:
                        model_id = await self.load_model_for_task(task_data)

                    if model_id:
                        # Check if the model has capacity for more tasks
                        model_config = self.loaded_backends[model_id].model_config
                        model_parallel_limit = model_config.get("performance_metrics", {}).get("parallel_requests", 1)
                        current_model_tasks = self.model_active_tasks.get(model_id, 0)

                        if current_model_tasks < model_parallel_limit:
                            # We can process this task
                            task = asyncio.create_task(self.process_one_task_wrapper(task_data))
                            task.job_name = task_data.get("job_name")
                            self.processing_tasks.add(task)
                            processed_any = True
                        else:
                            # Model is at capacity, put back in the temporary queue
                            await temp_queue.put(task_data)
                    else:
                        # No suitable model could be loaded
                        task_data["error"] = "No suitable model could be found or loaded for this task"
                        await self.completed_queue.put(task_data)

                    self.task_queue.task_done()

                # Put any skipped tasks back into the main queue
                while not temp_queue.empty():
                    await self.task_queue.put(await temp_queue.get())

                # If we couldn't process any tasks, wait a bit
                if not processed_any:
                    await asyncio.sleep(0.1)
            else:
                # We're at global batch limit
                await asyncio.sleep(0.1)

    async def process_one_task_wrapper(self, task_data: Dict[str, Any]):
        """
        Wrapper to process a task and put it in the completed queue.
        """
        # Get the model for this task before processing
        backend_instance_id = None
        
        try:
            # Load model and get instance ID
            backend_instance_id = await self.load_model_for_task(task_data)
            
            if backend_instance_id:
                # Increment active task count for this model
                self.model_active_tasks[backend_instance_id] = self.model_active_tasks.get(backend_instance_id, 0) + 1
            
            # For streaming tasks, we don't put them in the completed queue 
            # as they are sent chunk by chunk
            if task_data.get("stream", False):
                await self.process_one_task(task_data)
            else:
                processed_task = await self.process_one_task(task_data)
                if processed_task:
                    await self.completed_queue.put(processed_task)
        except Exception as e:
            task_data_with_error = task_data.copy()
            task_data_with_error["error"] = str(e)
            
            # For streaming tasks, send an error event
            if task_data.get("stream", False):
                await self.send_stream_event(task_data_with_error["id"], {
                    "event": "error",
                    "data": str(e)
                })
                # Send stream end event
                await self.send_stream_event(task_data_with_error["id"], {
                    "event": "done"
                })
            else:
                await self.completed_queue.put(task_data_with_error)
            
            print(f"Error processing task: {e}")
        finally:
            # Decrement active task count for this model when done
            if backend_instance_id:
                self.model_active_tasks[backend_instance_id] = max(0, self.model_active_tasks.get(backend_instance_id, 1) - 1)

    async def submit_loop(self):
        """
        Submit completed tasks in batches.
        """
        batch = []
        flush_interval = 5

        while True:
            try:
                task = await asyncio.wait_for(self.completed_queue.get(), timeout=flush_interval)
                batch.append(task)

                if len(batch) >= self.batch_size:
                    await self.submit_completed_tasks(batch)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self.submit_completed_tasks(batch)
                    batch = []

    async def run(self):
        """
        Main entry point.
        """
        
        # Initialize async primitives that were only declared in __init__
        self.task_queue = asyncio.Queue()
        self.completed_queue = asyncio.Queue()
        self.tasks_available_event = asyncio.Event()
        self.ws_connected = asyncio.Event()
        
        

        await self.register_with_server()
        if not self.worker_uid:
            print("[Worker] Failed to register, exiting.")
            return

        # Build the websocket URL (using "ws" instead of "http").
        ws_url = f"{self.server_base_url.replace('http', 'ws')}/worker/ws/{self.worker_uid}"
        print(f"[Worker] Will connect to WebSocket: {ws_url}")

        # Start background tasks.
        producer_task = asyncio.create_task(self.producer_loop())
        consumer_task = asyncio.create_task(self.consumer_loop())
        submit_task = asyncio.create_task(self.submit_loop())

        reconnect_attempt = 0
        while True:
            try:
                print(f"[Worker] Connecting to websocket: {ws_url}")
                async with websockets.connect(ws_url, max_size=None, ping_timeout=300) as ws:
                    self.websocket = ws
                    self.ws_connected.set()
                    print("[Worker] WebSocket connection established.")
                    reconnect_attempt = 0  # Reset on successful connection

                    # Run the listener loop on this connection.
                    await self.listen_forever()
            except Exception as e:
                print("[Worker] WebSocket connection error:", e)
            finally:
                # On disconnect, clear the connection event and cancel pending requests.
                self.ws_connected.clear()
                self._cancel_pending_requests_due_to_disconnect()
                
            # Determine the delay before the next reconnection attempt.
            delay = self.reconnect_schedule[min(reconnect_attempt, len(self.reconnect_schedule) - 1)]
            print(f"[Worker] Disconnected. Reconnecting in {delay} seconds...")
            await asyncio.sleep(delay)
            reconnect_attempt += 1

    async def register_with_server(self):
        """
        POST /worker/register to get a worker_uid from the server.
        """
        supported_backend_types = ["TabbyAPI", "Ollama"] 
        
        body = {
            "name": self.worker_name,
            "backend_types": supported_backend_types,
            "supported_models": self.model_configs
        }
        url = f"{self.server_base_url}/worker/register"
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(url, json=body)
                r.raise_for_status()
                data = r.json()
                self.worker_uid = data["worker_uid"]
                print(f"[Worker] Registered successfully. worker_uid={self.worker_uid}")
            except Exception as e:
                print("[Worker] Failed to register:", e)
                self.worker_uid = None

    async def listen_forever(self):
        """
        Continuously read messages from the server WebSocket
        and handle them.
        """
        while True:
            try:
                raw_msg = await self.websocket.recv()
            except websockets.ConnectionClosed:
                print("[Worker] WebSocket connection closed.")
                return
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                print("[Worker] Received non-JSON from server, ignoring.")
                continue
            await self.handle_server_message(data)

    async def cancel_tasks_for_job(self, job_name: str):
        """
        Cancel any tasks that are in progress (or queued locally) for the given job.
        """
        print("TASK CANCEL MESSAGE RECEIVED")
        # Cancel in-progress tasks.
        tasks_to_cancel = [t for t in self.processing_tasks if getattr(t, "job_name", None) == job_name]
        for t in tasks_to_cancel:
            t.cancel()
        # Also remove tasks from the local task_queue.
        new_queue = asyncio.Queue()
        while not self.task_queue.empty():
            task_data = self.task_queue.get_nowait()
            if task_data.get("job_name") != job_name:
                await new_queue.put(task_data)
        self.task_queue = new_queue
        print(f"[Worker] Cancelled tasks for job '{job_name}'.")

    async def handle_server_message(self, data: Dict[str, Any]):
        """
        Handle messages from the server.
        """
        request_id = data.get("request_id")
        action = data.get("action")

        if action == "cancel_tasks":
            job_name = data.get("job_name")
            print(f"[Worker] Received cancel_tasks for job '{job_name}'. Cancelling tasks...")
            await self.cancel_tasks_for_job(job_name)
            return

        # If this is a response to a pending request, resolve its future.
        if request_id and request_id in self.pending_requests:
            fut = self.pending_requests.pop(request_id)
            fut.set_result(data)
            return

        # Otherwise, handle push notifications.
        if action == "tasks_available":
            print("[Worker] Tasks available, notifying producer.")
            self.tasks_available_event.set()
        else:
            print("[Worker] Unhandled message from server:", data)

    async def request_tasks(self, number: int) -> List[Dict[str, Any]]:
        """
        Ask the server for up to 'number' tasks, including hot/cold model information.
        """
        hot_models, cold_models = await self.classify_models()
        
        request_id = str(uuid.uuid4())
        request = {
            "action": "request_tasks",
            "request_id": request_id,
            "number": number,
            "hot_models": hot_models,
            "cold_models": cold_models
        }
        try:
            response = await self.send_request(request)
        except Exception as e:
            print("[Worker] Failed to request tasks:", e)
            return []
        tasks = response.get("tasks", [])
        return tasks

    async def submit_completed_tasks(self, tasks: List[Dict[str, Any]]):
        """
        Submit a batch of completed tasks to the server.
        """
        if not tasks:
            return

        request_id = str(uuid.uuid4())
        request = {
            "action": "submit_completed_tasks",
            "request_id": request_id,
            "tasks": tasks
        }
        try:
            response = await self.send_request(request)
        except Exception as e:
            print("[Worker] Failed to submit completed tasks:", e)
            # Re-queue the tasks so they aren't lost.
            for task in tasks:
                await self.completed_queue.put(task)
            return

        ack = response.get("ack", False)
        print(f"[Worker] Server ack for completed tasks: {ack} (Count: {len(tasks)})")

    async def send_stream_event_without_ack(self, task_id: str, event: Dict[str, Any]):
        """
        Send a streaming event for a specific task to the server without waiting for acknowledgment.
        """
        request_id = str(uuid.uuid4())
        request = {
            "action": "stream_event",
            "request_id": request_id,
            "task_id": task_id,
            "event": event
        }
        try:
            # Wait until the websocket is connected.
            await self.ws_connected.wait()
            await self.websocket.send(json.dumps(request))
            return True
        except Exception as e:
            print(f"[Worker] Failed to send stream event for task {task_id}: {e}")
            return False

    async def send_stream_event(self, task_id: str, event: Dict[str, Any]):
        """
        Send a streaming event for a specific task to the server.
        """
        request_id = str(uuid.uuid4())
        request = {
            "action": "stream_event",
            "request_id": request_id,
            "task_id": task_id,
            "event": event
        }
        try:
            response = await self.send_request(request)
            return response.get("ack", False)
        except Exception as e:
            print(f"[Worker] Failed to send stream event for task {task_id}: {e}")
            return False

    async def send_request(self, msg: dict) -> dict:
        """
        Send a JSON request to the server WebSocket with a request_id,
        and wait for the matching response.
        """
        # Wait until the websocket is connected.
        await self.ws_connected.wait()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        request_id = msg["request_id"]
        self.pending_requests[request_id] = fut

        try:
            await self.websocket.send(json.dumps(msg))
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            raise e

        return await fut

    def _cancel_pending_requests_due_to_disconnect(self):
        """
        When the websocket disconnects, cancel all pending request futures so that they
        do not hang forever.
        """
        for req_id, fut in self.pending_requests.items():
            if not fut.done():
                fut.set_exception(ConnectionError("WebSocket disconnected"))
        self.pending_requests.clear()

    async def process_streamed_response(self, task_id: str, stream_iter: AsyncIterator[Dict[str, Any]]):
        """
        Process a streamed response from the backend and send chunks to the server.
        """
        async for chunk in stream_iter:
            # Send the chunk as a stream event without waiting for ack
            await self.send_stream_event_without_ack(task_id, {
                "event": "chunk",
                "data": chunk
            })
        
        # Send stream end event and wait for acknowledgment
        await self.send_stream_event(task_id, {
            "event": "done"
        })

    async def process_one_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single task via the backend.
        """
        conversation = task.get("conversation", [])
        task_id = task.get("id")
        
        # Extract additional parameters from the task
        stream = task.get("stream", False)
        max_tokens = task.get("max_tokens", 500)
        tools = task.get("tools", [])
        params = task.get("params", PRECISE_PARAMS)
        print(task)
        # Load the model required for this task
        backend_instance_id = await self.load_model_for_task(task)
        if not backend_instance_id:
            error_msg = f"Could not load model for task: {task_id}, models: {task.get('models', ['unknown'])}"
            print(f"[Worker] {error_msg}")
            
            if stream:
                await self.send_stream_event(task_id, {
                    "event": "error",
                    "data": error_msg
                })
                await self.send_stream_event(task_id, {
                    "event": "done"
                })
                return None
            else:
                task["error"] = error_msg
                task["worker_name"] = self.worker_name
                return task
        
        # Get the backend instance
        backend_instance = self.loaded_backends[backend_instance_id]
        backend = backend_instance.backend
        
        if stream:
            # Process streaming response
            stream_iter = await backend.chat_completion(
                conversation, 
                stream=True,
                max_tokens=max_tokens,
                params=params,
                tools=tools
            )
            # Handle streaming response in a separate method
            await self.process_streamed_response(task_id, stream_iter)
            # For streaming tasks, we return None as we've already sent the response
            return None
        else:
            # Process non-streaming response
            response = await backend.chat_completion(
                conversation, 
                stream=False,
                max_tokens=max_tokens,
                params=params,
                tools=tools
            )
            task["response"] = response
            task["worker_name"] = self.worker_name
            return task
    
    async def shutdown(self):
        """Gracefully shut down the worker and unload models"""
        print("\n[Worker] Shutting down, unloading models...")
        try:
            # Unload all active backends
            for instance_id in list(self.loaded_backends.keys()):
                await self.unload_backend(instance_id)
        except Exception as e:
            print(f"[Worker] Error during shutdown: {str(e)}")
        
        # Cancel all running tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()
                
        print("[Worker] Shutdown complete")


def main():
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    worker = WorkerClient(config_path)
    
    asyncio.run(worker.run())
    
    ## Set up signal handlers for graceful shutdown
    #loop = asyncio.get_event_loop()
    #
    ## Define shutdown handler
    #async def shutdown_handler(sig):
    #    print(f"\n[Worker] Received signal {sig.name}, shutting down...")
    #    await worker.shutdown()
    #    
    #    loop.stop()
    #
    ## Register signal handlers
    #for sig in (signal.SIGINT, signal.SIGTERM):
    #    loop.add_signal_handler(
    #        sig,
    #        lambda s=sig: asyncio.create_task(shutdown_handler(s))
    #    )
    #
    #try:
    #    loop.run_until_complete(worker.run())
    #except KeyboardInterrupt:
    #    # This is a fallback in case the signal handler doesn't catch it
    #    print("\n[Worker] Keyboard interrupt received, shutting down...")
    #    loop.run_until_complete(worker.shutdown())
    #finally:
    #    loop.close()


if __name__ == "__main__":
    main()