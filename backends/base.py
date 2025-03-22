# backends/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Any, Optional, TypedDict
import asyncio
import time
import psutil
import torch
import gc
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from backends.generation_params import GenerationParams, PreciseParams

# Initialize Rich console
console = Console()

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
    async def benchmark_model(self, model: ModelConfig) -> ModelConfig:
        """
        Implementation of the benchmark_model method that can be used by all backends.
        """
        # Print benchmark header
        model_name = model.get('name', 'Unknown Model')
        console.print(Panel.fit(f"[bold cyan]Benchmarking Model:[/] [yellow]{model_name}[/]", 
                            border_style="blue", padding=(1, 2)))
        
        # Initialize performance metrics if not present
        if model.get('performance_metrics') is None:
            model['performance_metrics'] = ModelPerformanceMetrics()
        
        # Load the model with progress indication
        console.print("[bold green]► Loading model...[/]")
        start_load = time.time()
        await self.load_model(model)
        load_time = time.time() - start_load
        console.print(f"[green]✓ Model loaded in[/] [bold]{load_time:.2f}s[/]")
        
        # Get process for memory measurements
        process = psutil.Process()
        process_pid = await self._get_pid() or process.pid
        process = psutil.Process(process_pid)
        
        # Function to measure RAM and VRAM
        def measure_memory():
            # Measure RAM
            ram_used = process.memory_info().rss // (1024 * 1024)  # Convert to MB
            
            # Measure VRAM for each GPU
            vram_used = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    used = (total - free) // (1024 * 1024)  # Convert to MB
                    vram_used.append(used)
            
            return ram_used, vram_used
        
        # Create a test prompt asking for a 500-word story
        conversation = [
            {"role": "system", "content": "You are a creative assistant."},
            {"role": "user", "content": "Write a 500-word story about an adventure in space."}
        ]
        
        console.print("[bold]📋 Test prompt:[/] Write a 500-word story about an adventure in space.")
        
        # Function to wait after error
        async def wait_after_error():
            for remaining in range(60, 0, -1):
                console.print(f"[yellow]Waiting after error: {remaining}s remaining.[/] Press Enter to skip wait...", end="\r")
                try:
                    # Set a very short timeout to make it non-blocking but still check for input
                    user_input = await asyncio.wait_for(
                        asyncio.to_thread(input, ""), 
                        timeout=1.0
                    )
                    if user_input.strip() == "":
                        console.print("\n[bold green]Wait skipped by user input.[/]")
                        break
                except asyncio.TimeoutError:
                    # No input within the timeout, continue waiting
                    pass
            console.print("[green]Resuming benchmark...[/]")
        
        # Function to test a specific number of parallel requests
        async def test_parallel_requests(count):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure initial memory usage
            init_ram, init_vram = measure_memory()
            
            console.print(f"[cyan]Testing [bold]{count}[/] parallel requests...[/]")
            start_time = time.time()
            
            # Create and run parallel requests
            tasks = []
            for _ in range(count):
                tasks.append(self.chat_completion(conversation.copy()))
            
            try:
                # Use a timeout to prevent hanging
                await asyncio.gather(*tasks, return_exceptions=True)
                success = True
            except Exception as e:
                console.print(f"[bold red]Error with {count} parallel requests: {str(e)}[/]")
                success = False
                # Wait after error occurs
                await wait_after_error()
            
            end_time = time.time()
            
            # Measure final memory usage
            final_ram, final_vram = measure_memory()
            
            # Calculate throughput (requests per second)
            elapsed_time = end_time - start_time
            throughput = count / elapsed_time if elapsed_time > 0 and success else 0
            
            # Calculate peak memory usage
            peak_ram = max(final_ram, init_ram)
            peak_vram = [max(final, init) for final, init in zip(final_vram, init_vram)] if final_vram else []
            
            # Print result for this test
            status = "[green]✓ SUCCESS[/]" if success else "[red]✗ FAILED[/]"
            console.print(f"  {status} - [bold]{count}[/] parallel requests: "
                        f"[cyan]{throughput:.2f}[/] req/s, "
                        f"[magenta]{peak_ram}[/] MB RAM, "
                        f"VRAM: {', '.join([f'[yellow]{v}[/] MB' for v in peak_vram])}")
            
            return {
                "success": success,
                "throughput": throughput,
                "time_per_request": elapsed_time / count if count > 0 and success else 0,
                "peak_ram_mb": peak_ram,
                "peak_vram_mb": peak_vram,
            }
        
        # Binary search to find optimal parallel request count
        console.print("\n[bold yellow]🔍 Finding optimal parallelism level...[/]")
        
        # Start with testing powers of 2
        counts_to_test = [1, 2, 4, 8, 16, 32]
        results = {}
        
        # Test powers of 2 to find approximate range
        best_count = 1
        best_throughput = 0
        max_successful_count = 0  # Track the maximum successful count
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Testing request counts...", total=len(counts_to_test))
            
            for count in counts_to_test:
                # Skip if count is higher than a previously failed count
                if max_successful_count > 0 and count > max_successful_count:
                    console.print(f"[yellow]Skipping {count} parallel requests (higher than previous failure)[/]")
                    progress.update(task, advance=1)
                    continue
                    
                result = await test_parallel_requests(count)
                results[count] = result
                progress.update(task, advance=1)
                
                if result["success"]:
                    max_successful_count = count
                    if result["throughput"] > best_throughput:
                        best_throughput = result["throughput"]
                        best_count = count
                else:
                    # If we failed, no need to test higher counts
                    console.print(f"[yellow]Reached limit at {count} parallel requests[/]")
                    break
        
        console.print("\n[bold yellow]🔍 Refining search for optimal parallelism...[/]")
        
        # Refine the search between the last successful count and the next power of 2
        if best_count > 1:
            lower_bound = best_count // 2
            upper_bound = best_count
            
            # If best_count is our highest successful test, try to find the actual upper limit
            if (best_count == max_successful_count and 
                best_count == counts_to_test[-1] or 
                (best_count in results and results[best_count]["success"])):
                
                next_count = best_count * 2
                # Only test if we haven't already hit a failure
                if max_successful_count == 0 or next_count <= max_successful_count:
                    console.print(f"[cyan]Testing if we can go higher: {next_count} parallel requests[/]")
                    result = await test_parallel_requests(next_count)
                    results[next_count] = result
                    
                    if result["success"]:
                        max_successful_count = next_count
                        if result["throughput"] > best_throughput:
                            best_throughput = result["throughput"]
                            best_count = next_count
                        lower_bound = best_count // 2
                        upper_bound = best_count * 2
                    else:
                        upper_bound = next_count
        
            # Binary search within the range
            console.print(f"[cyan]Binary search between {lower_bound} and {upper_bound}[/]")
            while upper_bound - lower_bound > 1:
                mid = (lower_bound + upper_bound) // 2
                
                # Skip if mid is higher than a previously failed count
                if max_successful_count > 0 and mid > max_successful_count:
                    console.print(f"[yellow]Skipping {mid} (higher than previous failure limit of {max_successful_count})[/]")
                    upper_bound = mid
                    continue
                    
                if mid not in results:
                    result = await test_parallel_requests(mid)
                    results[mid] = result
                
                if results[mid]["success"]:
                    max_successful_count = max(max_successful_count, mid)
                    if results[mid]["throughput"] > best_throughput:
                        best_throughput = results[mid]["throughput"]
                        best_count = mid
                    lower_bound = mid
                else:
                    upper_bound = mid
        
        # Get final data for the best configuration
        best_result = results[best_count]
        
        # Update model config with benchmark results
        model["performance_metrics"]["parallel_requests"] = best_count
        model["performance_metrics"]["ram_requirement"] = best_result["peak_ram_mb"]
        model["performance_metrics"]["vram_requirement"] = best_result["peak_vram_mb"]
        model["performance_metrics"]["benchmark_results"] = {
            "throughput": best_result["throughput"],
            "time_per_request": best_result["time_per_request"],
            "tested_configs": results
        }
        
        # Create a summary table of all tested configurations
        console.print("\n[bold green]📊 Benchmark Results Summary[/]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parallel Requests")
        table.add_column("Status")
        table.add_column("Throughput (req/s)")
        table.add_column("Time per req (s)")
        table.add_column("RAM (MB)")
        table.add_column("VRAM (MB)")
        
        for count, res in sorted(results.items()):
            status = "✅" if res["success"] else "❌"
            vram_str = ", ".join([str(v) for v in res["peak_vram_mb"]]) if res["peak_vram_mb"] else "N/A"
            
            # Highlight the best configuration
            if count == best_count:
                table.add_row(
                    f"[bold yellow]{count}[/]",
                    f"[green]{status}[/]",
                    f"[bold cyan]{res['throughput']:.2f}[/]",
                    f"[bold]{res['time_per_request']:.2f}[/]",
                    f"[bold]{res['peak_ram_mb']}[/]",
                    f"[bold]{vram_str}[/]"
                )
            else:
                table.add_row(
                    str(count),
                    f"[green]{status}[/]" if res["success"] else f"[red]{status}[/]",
                    f"{res['throughput']:.2f}",
                    f"{res['time_per_request']:.2f}",
                    str(res['peak_ram_mb']),
                    vram_str
                )
        
        console.print(table)
        
        # Display optimal configuration
        optimal_text = Text()
        optimal_text.append("🏆 Optimal Configuration\n", style="bold yellow")
        optimal_text.append(f"Parallel Requests: ", style="dim")
        optimal_text.append(f"{best_count}\n", style="bold cyan")
        optimal_text.append(f"Throughput: ", style="dim")
        optimal_text.append(f"{best_result['throughput']:.2f} requests/second\n", style="bold green")
        optimal_text.append(f"Time per request: ", style="dim")
        optimal_text.append(f"{best_result['time_per_request']:.2f} seconds\n", style="bold")
        optimal_text.append(f"RAM Usage: ", style="dim")
        optimal_text.append(f"{best_result['peak_ram_mb']} MB\n", style="bold magenta")
        if best_result["peak_vram_mb"]:
            optimal_text.append(f"VRAM Usage: ", style="dim")
            optimal_text.append(", ".join([f"{v} MB" for v in best_result["peak_vram_mb"]]), style="bold yellow")
        
        console.print(Panel(optimal_text, border_style="green", padding=(1, 2)))
        
        # Unload the model
        console.print("[bold green]► Unloading model...[/]")
        await self.unload_model()
        console.print("[green]✓ Model unloaded[/]")
        
        return model
    
    @abstractmethod
    async def get_type(self) -> Literal["Managed", "Instant"]:
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