import asyncio
import json
import uuid
import sys
from typing import List, Dict, Any, Optional

import websockets
import httpx

from backends.base import Backend
from backends.tabby import TabbyBackend
from backends.ollama import OllamaBackend


class WorkerClient:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.worker_name: str = cfg.get("worker_name", "my_worker")
        self.server_base_url: str = cfg.get("server_base_url", "http://localhost:8000")
        self.backend_type: str = cfg.get("backend", "TabbyAPI")
        self.supported_models: List[str] = cfg.get("models", ["llama8b"])
        model_aliases = cfg.get("model_aliases", {})
        if self.backend_type == "TabbyAPI":
            tabby_api_url = cfg.get("tabby_api_url", "http://127.0.0.1")
            tabby_api_key = cfg.get("tabby_api_key", "")
            self.backend: Backend = TabbyBackend(tabby_api_url, tabby_api_key, model_aliases, supported_models=self.supported_models)
        elif self.backend_type == "OllamaAPI":
            ollama_api_url = cfg.get("ollama_api_url", "http://localhost:11434")
            
            self.backend: Backend = OllamaBackend(ollama_api_url, model_aliases, supported_models=self.supported_models)
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

        self.worker_uid: Optional[str] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.pending_requests: Dict[str, asyncio.Future] = {}

        self.batch_size: int = cfg.get("batch_size", 20)
        self.buffer_size = self.batch_size

        # Queue for inbound tasks
        self.task_queue = asyncio.Queue()

        # Queue for completed tasks
        self.completed_queue = asyncio.Queue()

        # Event used by the server to tell us new tasks are available
        self.tasks_available_event = asyncio.Event()

        self.processing_tasks = set()

        # This event is set only when the websocket connection is active.
        self.ws_connected = asyncio.Event()

        # Reconnection schedule:
        # 10 attempts every 5 seconds, 10 attempts every 60 seconds, then every 3600 seconds (1 hour) for 24 attempts.
        self.reconnect_schedule = [5] * 10 + [60] * 10 + [3600] * 24

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

    async def consumer_loop(self):
        """
        Consumer that processes tasks concurrently up to batch_size limit.
        """
        while True:
            # Clean up completed tasks.
            self.processing_tasks = {t for t in self.processing_tasks if not t.done()}

            # If we have room for more tasks, process them.
            while len(self.processing_tasks) < self.batch_size:
                try:
                    task_data = self.task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                # Create and start the task. Attach the job name to the task.
                task = asyncio.create_task(self.process_one_task_wrapper(task_data))
                task.job_name = task_data.get("job_name")
                self.processing_tasks.add(task)
                self.task_queue.task_done()

            await asyncio.sleep(0.1)

    async def process_one_task_wrapper(self, task_data: Dict[str, Any]):
        """
        Wrapper to process a task and put it in the completed queue.
        """
        try:
            processed_task = await self.process_one_task(task_data)
            await self.completed_queue.put(processed_task)
        except Exception as e:
            task_data_with_error = task_data
            task_data_with_error["error"] = str(e)
            await self.completed_queue.put(task_data_with_error)
            print(f"Error processing task: {e}")

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
        
        The persistent producer/consumer/submit loops are started once.
        Then a reconnect loop is entered that connects to the websocket server,
        starts the listen_forever loop, and if disconnected, waits for a delay
        (using the schedule) before trying to reconnect.
        """
        active_model = await self.backend.load_model()
        print(f"[Worker] Active model from backend: {active_model}")

        await self.register_with_server()
        if not self.worker_uid:
            print("[Worker] Failed to register, exiting.")
            return

        # Build the websocket URL (using "ws" instead of "http").
        ws_url = f"{self.server_base_url.replace('http', 'ws')}/worker/ws/{self.worker_uid}"
        print(f"[Worker] Will connect to WebSocket: {ws_url}")

        # Start the persistent background tasks (which use send_request).
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

                    # Run the listener loop on this connection. (It will return if the connection closes.)
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
        active_model = getattr(self.backend, "active_model", "unknown_model")
        body = {
            "name": self.worker_name,
            "supported_models": self.supported_models,
            "active_model": active_model
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
        Ask the server for up to 'number' tasks.
        """
        request_id = str(uuid.uuid4())
        request = {
            "action": "request_tasks",
            "request_id": request_id,
            "number": number
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
            # Re-queue the tasks so they aren’t lost.
            for task in tasks:
                await self.completed_queue.put(task)
            return

        ack = response.get("ack", False)
        print(f"[Worker] Server ack for completed tasks: {ack} (Count: {len(tasks)})")

    async def send_request(self, msg: dict) -> dict:
        """
        Send a JSON request to the server WebSocket with a request_id,
        and wait for the matching response.
        
        This method first waits until the connection is active.
        If the send fails (or if the connection is lost before a response is received)
        then the waiting future is cancelled so that the higher–level logic (e.g. re–queuing)
        can take over.
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

    async def process_one_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single task via the backend.
        """
        conversation = task.get("conversation", [])
        text = await self.backend.chat_completion(conversation)
        task["text"] = text
        task["worker_name"] = self.worker_name
        return task


def main():
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    worker = WorkerClient(config_path)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
