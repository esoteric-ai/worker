import asyncio
import os
import sys
import base64
from pathlib import Path
from rich.console import Console
from typing import Dict

# Import our GigaChat backend
from backends.gigachat import GigaChatBackend, GigaChatBackendConfig
from backends.base import ModelConfig

# Initialize Rich console for nicer output
console = Console()

def encode_image_to_base64(image_path):
    """Read an image file and encode it to base64"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

async def test_gigachat():
    # Get API key from environment variable
    api_key = '=='
    if not api_key:
        console.print("[bold red]Error:[/bold red] GIGACHAT_API_KEY environment variable is not set")
        return
    
    # Create backend config with the API key
    config = {"api_key": api_key}
    
    # Initialize the backend
    backend = GigaChatBackend(config)
    
    try:
        # Load the model (minimal configuration)
        model_config = ModelConfig(
            alias="gigachat",
            backend="gigachat",
            api_name="GigaChat-Pro"
        )
        await backend.load_model(model_config)
        
        # Get the current directory and image path
        image_path = Path(os.path.dirname(os.path.abspath(__file__))) / "image.jpg"
        
        if not image_path.exists():
            console.print(f"[bold yellow]Warning:[/bold yellow] Image not found at {image_path}")
            console.print("[bold yellow]Continuing with text-only conversation...[/bold yellow]")
            
            # Create a sample text-only conversation
            conversation = [
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": "Tell me briefly about quantum computing."}
            ]
        else:
            # Encode the image to base64
            console.print(f"[green]Loading image from {image_path}[/green]")
            img_base64 = encode_image_to_base64(image_path)
            
            # Create a sample conversation with image
            conversation = [
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image? Describe it briefly."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]}
            ]
        
        # Display what we're sending
        console.print("[bold green]Starting chat completion with streaming...[/bold green]")
        if isinstance(conversation[1]["content"], list):
            console.print("[bold blue]User:[/bold blue] What's in this image? (Image attached)")
        else:
            console.print("[bold blue]User:[/bold blue] Tell me briefly about quantum computing.")
        
        console.print("[bold purple]Assistant:[/bold purple] ", end="")
        
        tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        
        conversation = [
            {"role": "system", "content": "You are a helpful and concise assistant that can call tools."},
            {"role": "user", "content": "Hello, please fetch weather in Moscow currently."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "641f0647-076b-4eb0-83d0-755ac526aa76", 
                 "type": "function", 
                 "function": {
                     "name": "get_current_weather", 
                     "arguments": '{"location": "Moscow"}'
                 }}
            ]},
            {"role": "tool", "tool_call_id": "641f0647-076b-4eb0-83d0-755ac526aa76", "content": '{"temperature": 15, "unit": "celsius", "description": "Partly cloudy"}'}
        ]
        
        # Process streaming response
        response_text = ""
        # async for chunk in await backend.chat_completion(conversation, tools=tools, stream=True, max_tokens=500):
        #     content = chunk["choices"][0]["delta"].get("content", "")
        #     tool_calls = chunk["choices"][0]["delta"].get("tool_calls", [])
        #     
        #     if tool_calls:
        #         print(tool_calls) 
        #     
        #     if content:
        #         print(content, end="")
        #         response_text += content
        #         
        #     # Check if we're done
        #     finish_reason = chunk["choices"][0].get("finish_reason")
        #     if finish_reason:
        #         console.print(f"\n\n[dim]Finished: {finish_reason}[/dim]")
        
        print(await backend.chat_completion(conversation, tools=tools, stream=False, max_tokens=500))
        
        console.print("\n[bold green]Chat completion completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
    
    finally:
        # Always unload the model to clean up resources
        await backend.unload_model()
        console.print("[dim]Model unloaded.[/dim]")

def fix_asyncio_event_loop_policy():
    """Fix for Event Loop is Closed error on Windows"""
    if sys.platform.startswith('win'):
        # On Windows, use the SelectEventLoop to avoid ProactorEventLoop issues
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    # Set up instructions
    console.print("[bold]GigaChat Backend Test[/bold]")
    console.print("Testing with image attachment functionality.\n")
    
    # Fix for Windows asyncio issues
    fix_asyncio_event_loop_policy()
    
    # Run the async test
    asyncio.run(test_gigachat())