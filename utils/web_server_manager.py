"""
Web Server Manager for DurgasAI.

This module provides utilities for managing web server inference instances:
- Server lifecycle management (start, stop, restart)
- Configuration management and validation
- Process monitoring and health checks
- Performance metrics collection
- Integration with Streamlit session state
"""

import asyncio
import json
import time
import subprocess
import threading
import requests
import os
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import psutil

from .logger import debug, info, warning, error, log_user_action
from .config import Config


class WebServerConfig:
    """Configuration class for web server instances."""
    
    def __init__(
        self,
        server_type: str = "basic",
        model: str = "google-bert/bert-base-uncased",
        task: str = "fill-mask",
        host: str = "0.0.0.0",
        port: int = 8000,
        max_batch_size: int = 4,
        batch_timeout: float = 0.1,
        max_queue_size: int = 50,
        **kwargs
    ):
        """
        Initialize web server configuration.
        
        Args:
            server_type: Type of server (basic, advanced, multi_task)
            model: HuggingFace model identifier
            task: Inference task type
            host: Server host address
            port: Server port number
            max_batch_size: Maximum batch size for batching
            batch_timeout: Timeout for batch collection (seconds)
            max_queue_size: Maximum queue size
            **kwargs: Additional configuration parameters
        """
        self.server_type = server_type
        self.model = model
        self.task = task
        self.host = host
        self.port = port
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size
        self.additional_config = kwargs
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.server_type not in ["basic", "advanced", "multi_task"]:
            raise ValueError(f"Invalid server type: {self.server_type}")
        
        if not self.model or not isinstance(self.model, str):
            raise ValueError("Model must be a non-empty string")
        
        if not self.task or not isinstance(self.task, str):
            raise ValueError("Task must be a non-empty string")
        
        if not (1000 <= self.port <= 65535):
            raise ValueError("Port must be between 1000 and 65535")
        
        if not (1 <= self.max_batch_size <= 32):
            raise ValueError("Max batch size must be between 1 and 32")
        
        if not (0.001 <= self.batch_timeout <= 10.0):
            raise ValueError("Batch timeout must be between 0.001 and 10.0 seconds")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'server_type': self.server_type,
            'model': self.model,
            'task': self.task,
            'host': self.host,
            'port': self.port,
            'max_batch_size': self.max_batch_size,
            'batch_timeout': self.batch_timeout,
            'max_queue_size': self.max_queue_size,
            **self.additional_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WebServerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def get_server_id(self) -> str:
        """Generate unique server ID."""
        return f"{self.model}:{self.task}:{self.port}"


class WebServerInstance:
    """Represents a running web server instance."""
    
    def __init__(self, config: WebServerConfig, process: subprocess.Popen):
        """
        Initialize web server instance.
        
        Args:
            config: Server configuration
            process: Server process handle
        """
        self.config = config
        self.process = process
        self.start_time = datetime.now()
        self.status = "starting"
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'last_health_check': None
        }
        
        debug(f"WebServerInstance created: {self.config.get_server_id()}", "web_server_manager")
    
    def get_server_id(self) -> str:
        """Get server ID."""
        return self.config.get_server_id()
    
    def is_running(self) -> bool:
        """Check if server process is running."""
        if not self.process:
            return False
        
        return self.process.poll() is None
    
    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update server statistics."""
        self.stats.update(stats)
        self.stats['last_update'] = datetime.now()
    
    def terminate(self, timeout: int = 5):
        """Terminate server process."""
        if not self.process:
            return
        
        try:
            # Try graceful termination first
            self.process.terminate()
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            self.process.kill()
            self.process.wait(timeout=2)
        
        self.status = "stopped"
        debug(f"WebServerInstance terminated: {self.get_server_id()}", "web_server_manager")


class WebServerManager:
    """
    Manager for web server inference instances.
    
    This class provides:
    - Server lifecycle management
    - Configuration management
    - Health monitoring
    - Performance tracking
    - Process management
    """
    
    def __init__(self):
        """
        Initialize the web server manager with comprehensive setup.
        
        This constructor sets up the web server management system including:
        - Server instance tracking and lifecycle management
        - Server template loading for different deployment types
        - Temporary directory setup for server files
        - Performance monitoring initialization
        
        Components Initialized:
        - Server registry for tracking active deployments
        - Template system for pre-configured server types
        - Temporary file management for server scripts
        - Logging integration for debugging and monitoring
        """
        debug("Starting WebServerManager initialization", "web_server_manager")
        
        # Initialize server registry for tracking active servers
        self.servers: Dict[str, WebServerInstance] = {}
        debug("Server instance registry initialized", "web_server_manager")
        
        # Load server templates for different deployment configurations
        debug("Loading server configuration templates", "web_server_manager")
        self.server_templates = self._load_server_templates()
        template_count = len(self.server_templates)
        debug(f"Loaded {template_count} server templates", "web_server_manager")
        
        # Setup temporary directory for server files
        self.temp_dir = Path("output/temp/servers")
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            debug(f"Created temporary directory: {self.temp_dir}", "web_server_manager")
        else:
            debug(f"Using existing temporary directory: {self.temp_dir}", "web_server_manager")
        
        # Log available templates for debugging
        template_names = list(self.server_templates.keys())
        debug(f"Available server templates: {', '.join(template_names)}", "web_server_manager")
        
        info("WebServerManager initialization completed successfully", "web_server_manager",
             template_count=template_count,
             temp_dir=str(self.temp_dir))
    
    def _load_server_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load server configuration templates."""
        return {
            "basic_fill_mask": {
                "name": "Basic Fill-Mask Server",
                "description": "Simple BERT-based fill-mask inference server",
                "config": WebServerConfig(
                    server_type="basic",
                    model="google-bert/bert-base-uncased",
                    task="fill-mask",
                    port=8000
                )
            },
            "advanced_text_generation": {
                "name": "Advanced Text Generation Server",
                "description": "High-performance text generation with batching",
                "config": WebServerConfig(
                    server_type="advanced",
                    model="microsoft/DialoGPT-medium",
                    task="text-generation",
                    port=8001,
                    max_batch_size=4,
                    batch_timeout=0.1
                )
            },
            "sentiment_analysis": {
                "name": "Sentiment Analysis Server",
                "description": "Fast sentiment analysis with batching",
                "config": WebServerConfig(
                    server_type="advanced",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    task="sentiment-analysis",
                    port=8002,
                    max_batch_size=8,
                    batch_timeout=0.05
                )
            }
        }
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available server templates."""
        return self.server_templates
    
    def create_server(self, config: WebServerConfig) -> str:
        """
        Create and start a new web server instance.
        
        Args:
            config: Server configuration
            
        Returns:
            Server ID
            
        Raises:
            ValueError: If server already exists or configuration is invalid
            RuntimeError: If server fails to start
        """
        debug("Starting web server creation process", "web_server",
              server_type=config.server_type,
              model=config.model,
              host=config.host,
              port=config.port)
        
        server_id = config.get_server_id()
        debug(f"Generated server ID: {server_id}", "web_server")
        
        # Check if server already exists
        if server_id in self.servers:
            warning(f"Server {server_id} already exists", "web_server",
                   existing_servers=list(self.servers.keys()))
            raise ValueError(f"Server {server_id} already exists")
        
        # Check if port is available
        debug(f"Checking port availability: {config.host}:{config.port}", "web_server")
        if not self._is_port_available(config.host, config.port):
            error(f"Port {config.port} is already in use", "web_server",
                  host=config.host, port=config.port)
            raise ValueError(f"Port {config.port} is already in use")
        
        debug("Port is available, proceeding with server creation", "web_server")
        
        try:
            # Generate server code based on configuration
            debug("Generating server code", "web_server", server_type=config.server_type)
            server_code = self._generate_server_code(config)
            debug("Server code generated successfully", "web_server",
                  code_length=len(server_code))
            
            # Save server code to temporary file for execution
            server_file = self.temp_dir / f"server_{config.port}.py"
            debug(f"Saving server code to: {server_file}", "web_server")
            
            with open(server_file, 'w') as f:
                f.write(server_code)
            debug("Server code saved successfully", "web_server", file_size=server_file.stat().st_size)
            
            # Start server process in background
            debug("Starting server process", "web_server")
            process = self._start_server_process(config, server_file)
            debug("Server process started", "web_server", process_id=process.pid)
            
            # Create server instance for tracking
            debug("Creating server instance for tracking", "web_server")
            server_instance = WebServerInstance(config, process)
            self.servers[server_id] = server_instance
            debug("Server instance created and registered", "web_server",
                  total_servers=len(self.servers))
            
            # Wait for server to start and perform health check
            debug("Waiting for server startup and performing health check", "web_server")
            time.sleep(2)
            if self._check_server_health(config.host, config.port):
                server_instance.status = "running"
                info(f"Server started successfully: {server_id}", "web_server_manager")
                log_user_action("server_started", server_id=server_id, config=config.to_dict())
            else:
                server_instance.status = "failed"
                warning(f"Server health check failed: {server_id}", "web_server_manager")
            
            return server_id
            
        except Exception as e:
            error(f"Failed to create server {server_id}: {str(e)}", "web_server_manager", e)
            # Cleanup on failure
            if server_id in self.servers:
                self.stop_server(server_id)
            raise RuntimeError(f"Failed to start server: {str(e)}")
    
    def stop_server(self, server_id: str) -> bool:
        """
        Stop a web server instance.
        
        Args:
            server_id: Server ID to stop
            
        Returns:
            True if stopped successfully, False otherwise
        """
        if server_id not in self.servers:
            warning(f"Server not found: {server_id}", "web_server_manager")
            return False
        
        try:
            server_instance = self.servers[server_id]
            server_instance.terminate()
            
            # Remove from active servers
            del self.servers[server_id]
            
            info(f"Server stopped successfully: {server_id}", "web_server_manager")
            log_user_action("server_stopped", server_id=server_id)
            
            return True
            
        except Exception as e:
            error(f"Failed to stop server {server_id}: {str(e)}", "web_server_manager", e)
            return False
    
    def restart_server(self, server_id: str) -> bool:
        """
        Restart a web server instance.
        
        Args:
            server_id: Server ID to restart
            
        Returns:
            True if restarted successfully, False otherwise
        """
        if server_id not in self.servers:
            warning(f"Server not found: {server_id}", "web_server_manager")
            return False
        
        try:
            # Get configuration before stopping
            config = self.servers[server_id].config
            
            # Stop server
            self.stop_server(server_id)
            
            # Wait a moment
            time.sleep(1)
            
            # Start server again
            new_server_id = self.create_server(config)
            
            info(f"Server restarted successfully: {server_id}", "web_server_manager")
            return True
            
        except Exception as e:
            error(f"Failed to restart server {server_id}: {str(e)}", "web_server_manager", e)
            return False
    
    def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """
        Get server status and statistics.
        
        Args:
            server_id: Server ID
            
        Returns:
            Server status dictionary or None if not found
        """
        if server_id not in self.servers:
            return None
        
        server_instance = self.servers[server_id]
        
        return {
            'server_id': server_id,
            'status': server_instance.status,
            'config': server_instance.config.to_dict(),
            'uptime': server_instance.get_uptime(),
            'is_running': server_instance.is_running(),
            'stats': server_instance.stats,
            'start_time': server_instance.start_time.isoformat()
        }
    
    def get_all_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers."""
        return {
            server_id: self.get_server_status(server_id)
            for server_id in self.servers.keys()
        }
    
    def health_check_server(self, server_id: str) -> bool:
        """
        Perform health check on a server.
        
        Args:
            server_id: Server ID
            
        Returns:
            True if healthy, False otherwise
        """
        if server_id not in self.servers:
            return False
        
        server_instance = self.servers[server_id]
        config = server_instance.config
        
        is_healthy = self._check_server_health(config.host, config.port)
        server_instance.stats['last_health_check'] = datetime.now().isoformat()
        
        if is_healthy:
            server_instance.status = "running"
        else:
            server_instance.status = "unhealthy"
        
        return is_healthy
    
    def update_server_stats(self, server_id: str) -> Dict[str, Any]:
        """
        Update server statistics by fetching from the server.
        
        Args:
            server_id: Server ID
            
        Returns:
            Updated statistics dictionary
        """
        if server_id not in self.servers:
            return {}
        
        server_instance = self.servers[server_id]
        config = server_instance.config
        
        try:
            # Try to get stats from server
            if config.server_type in ['advanced', 'multi_task']:
                stats_url = f"http://{config.host}:{config.port}/stats"
                response = requests.get(stats_url, timeout=5)
                
                if response.status_code == 200:
                    server_stats = response.json()
                    server_instance.update_stats(server_stats.get('server_stats', {}))
            
            return server_instance.stats
            
        except Exception as e:
            debug(f"Failed to update stats for {server_id}: {str(e)}", "web_server_manager")
            return server_instance.stats
    
    def cleanup_all_servers(self):
        """Stop and cleanup all running servers."""
        server_ids = list(self.servers.keys())
        for server_id in server_ids:
            self.stop_server(server_id)
        
        info("All servers cleaned up", "web_server_manager")
    
    def _is_port_available(self, host: str, port: int) -> bool:
        """Check if a port is available."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception:
            return False
    
    def _check_server_health(self, host: str, port: int) -> bool:
        """Check server health via HTTP request."""
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _start_server_process(self, config: WebServerConfig, server_file: Path) -> subprocess.Popen:
        """Start server process using uvicorn."""
        # Convert path to module format
        module_path = str(server_file.relative_to(Path.cwd())).replace('/', '.').replace('\\', '.').replace('.py', '')
        
        cmd = [
            "python", "-m", "uvicorn",
            f"{module_path}:app",
            "--host", config.host,
            "--port", str(config.port),
            "--log-level", "info"
        ]
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path.cwd())
        )
        
        return process
    
    def _generate_server_code(self, config: WebServerConfig) -> str:
        """Generate server code based on configuration."""
        if config.server_type == "basic":
            return self._generate_basic_server_code(config)
        elif config.server_type == "advanced":
            return self._generate_advanced_server_code(config)
        elif config.server_type == "multi_task":
            return self._generate_multi_task_server_code(config)
        else:
            raise ValueError(f"Unknown server type: {config.server_type}")
    
    def _generate_basic_server_code(self, config: WebServerConfig) -> str:
        """Generate basic server code."""
        return f'''
import asyncio
import json
import time
from datetime import datetime
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from transformers import pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline variable
pipe = None

async def homepage(request):
    """Handle inference requests."""
    try:
        payload = await request.body()
        text_input = payload.decode("utf-8")
        
        if not text_input or len(text_input.strip()) == 0:
            return JSONResponse({{"error": "Empty input provided"}}, status_code=400)
        
        response_queue = asyncio.Queue()
        await request.app.model_queue.put((text_input, response_queue))
        output = await response_queue.get()
        
        return JSONResponse(output)
        
    except Exception as e:
        logger.error(f"Error in homepage handler: {{str(e)}}")
        return JSONResponse({{"error": f"Internal server error: {{str(e)}}""}}, status_code=500)

async def server_loop(queue):
    """Main inference loop."""
    global pipe
    try:
        logger.info("Loading pipeline: {config.task} with model {config.model}")
        pipe = pipeline(task="{config.task}", model="{config.model}")
        logger.info("Pipeline loaded successfully")
        
        while True:
            try:
                text_input, response_queue = await queue.get()
                logger.info(f"Processing request: {{text_input[:50]}}...")
                
                start_time = time.time()
                result = pipe(text_input)
                inference_time = time.time() - start_time
                
                response = {{
                    "result": result,
                    "metadata": {{
                        "model": "{config.model}",
                        "task": "{config.task}",
                        "inference_time": round(inference_time, 4),
                        "timestamp": datetime.now().isoformat()
                    }}
                }}
                
                await response_queue.put(response)
                logger.info(f"Request processed in {{inference_time:.4f}}s")
                
            except Exception as e:
                logger.error(f"Error processing request: {{str(e)}}")
                error_response = {{
                    "error": str(e),
                    "metadata": {{
                        "model": "{config.model}",
                        "task": "{config.task}",
                        "timestamp": datetime.now().isoformat()
                    }}
                }}
                await response_queue.put(error_response)
                
    except Exception as e:
        logger.error(f"Critical error in server loop: {{str(e)}}")
        raise

async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({{
        "status": "healthy",
        "model": "{config.model}",
        "task": "{config.task}",
        "timestamp": datetime.now().isoformat()
    }})

async def startup_event():
    """Initialize the server on startup."""
    logger.info("Starting inference server...")
    queue = asyncio.Queue()
    app.model_queue = queue
    asyncio.create_task(server_loop(queue))
    logger.info("Inference server started successfully")

# Create application
routes = [
    Route("/", homepage, methods=["POST"]),
    Route("/health", health_check, methods=["GET"]),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
]

app = Starlette(routes=routes, middleware=middleware)
app.model_queue = None

@app.on_event("startup")
async def startup():
    await startup_event()
'''
    
    def _generate_advanced_server_code(self, config: WebServerConfig) -> str:
        """Generate advanced server code with batching."""
        return f'''
import asyncio
import json
import time
from datetime import datetime
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from transformers import pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_BATCH_SIZE = {config.max_batch_size}
BATCH_TIMEOUT = {config.batch_timeout}
MAX_QUEUE_SIZE = {config.max_queue_size}

# Global variables
pipe = None
is_overloaded = False
stats = {{
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "batch_count": 0,
    "average_batch_size": 0
}}

async def homepage(request):
    """Handle inference requests with circuit breaking."""
    global stats
    
    try:
        if is_overloaded:
            return JSONResponse({{"error": "Server is overloaded, please try again later"}}, status_code=503)
        
        if request.app.model_queue.qsize() >= MAX_QUEUE_SIZE:
            return JSONResponse({{"error": "Queue is full, please try again later"}}, status_code=503)
        
        payload = await request.body()
        text_input = payload.decode("utf-8")
        
        if not text_input or len(text_input.strip()) == 0:
            return JSONResponse({{"error": "Empty input provided"}}, status_code=400)
        
        response_queue = asyncio.Queue()
        request_id = f"req_{{int(time.time() * 1000)}}_{{id(request)}}"
        
        request_obj = {{
            "id": request_id,
            "input": text_input,
            "response_queue": response_queue,
            "timestamp": time.time()
        }}
        
        await request.app.model_queue.put(request_obj)
        
        try:
            output = await asyncio.wait_for(response_queue.get(), timeout=30.0)
            stats["successful_requests"] += 1
            return JSONResponse(output)
        except asyncio.TimeoutError:
            stats["failed_requests"] += 1
            return JSONResponse({{"error": "Request timeout"}}, status_code=504)
        
    except Exception as e:
        logger.error(f"Error in homepage handler: {{str(e)}}")
        stats["failed_requests"] += 1
        return JSONResponse({{"error": f"Internal server error: {{str(e)}}""}}, status_code=500)
    finally:
        stats["total_requests"] += 1

async def server_loop_with_batching(queue):
    """Advanced inference loop with dynamic batching."""
    global pipe, is_overloaded, stats
    
    try:
        logger.info("Loading pipeline: {config.task} with model {config.model}")
        pipe = pipeline(task="{config.task}", model="{config.model}")
        logger.info("Pipeline loaded successfully")
        
        while True:
            try:
                batch_requests = []
                batch_inputs = []
                
                # Get first request (blocking)
                first_request = await queue.get()
                batch_requests.append(first_request)
                batch_inputs.append(first_request["input"])
                
                # Collect additional requests with timeout
                start_time = time.time()
                while (len(batch_requests) < MAX_BATCH_SIZE and 
                       (time.time() - start_time) < BATCH_TIMEOUT):
                    try:
                        remaining_timeout = BATCH_TIMEOUT - (time.time() - start_time)
                        if remaining_timeout <= 0:
                            break
                        
                        request = await asyncio.wait_for(queue.get(), timeout=remaining_timeout)
                        batch_requests.append(request)
                        batch_inputs.append(request["input"])
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                batch_size = len(batch_requests)
                logger.info(f"Processing batch of {{batch_size}} requests")
                
                inference_start = time.time()
                
                if batch_size == 1:
                    results = [pipe(batch_inputs[0])]
                else:
                    results = pipe(batch_inputs, batch_size=batch_size)
                
                inference_time = time.time() - inference_start
                
                # Send responses back
                for i, (request_obj, result) in enumerate(zip(batch_requests, results)):
                    response = {{
                        "result": result,
                        "metadata": {{
                            "request_id": request_obj["id"],
                            "model": "{config.model}",
                            "task": "{config.task}",
                            "batch_size": batch_size,
                            "batch_position": i,
                            "inference_time": round(inference_time, 4),
                            "queue_time": round(inference_start - request_obj["timestamp"], 4),
                            "timestamp": datetime.now().isoformat()
                        }}
                    }}
                    await request_obj["response_queue"].put(response)
                
                # Update statistics
                stats["batch_count"] += 1
                stats["average_batch_size"] = (
                    (stats["average_batch_size"] * (stats["batch_count"] - 1) + batch_size) 
                    / stats["batch_count"]
                )
                
                logger.info(f"Batch processed: {{batch_size}} requests in {{inference_time:.4f}}s")
                
            except Exception as e:
                logger.error(f"Error processing batch: {{str(e)}}")
                
                for request_obj in batch_requests:
                    error_response = {{
                        "error": str(e),
                        "metadata": {{
                            "request_id": request_obj["id"],
                            "model": "{config.model}",
                            "task": "{config.task}",
                            "timestamp": datetime.now().isoformat()
                        }}
                    }}
                    await request_obj["response_queue"].put(error_response)
                    
    except Exception as e:
        logger.error(f"Critical error in server loop: {{str(e)}}")
        is_overloaded = True
        raise

async def stats_endpoint(request):
    """Return server performance statistics."""
    queue_size = request.app.model_queue.qsize() if request.app.model_queue else 0
    
    return JSONResponse({{
        "server_stats": stats,
        "server_config": {{
            "model": "{config.model}",
            "task": "{config.task}",
            "max_batch_size": MAX_BATCH_SIZE,
            "batch_timeout": BATCH_TIMEOUT,
            "max_queue_size": MAX_QUEUE_SIZE
        }},
        "current_state": {{
            "queue_size": queue_size,
            "is_overloaded": is_overloaded
        }},
        "timestamp": datetime.now().isoformat()
    }})

async def health_check(request):
    """Enhanced health check endpoint."""
    queue_size = request.app.model_queue.qsize() if request.app.model_queue else 0
    
    return JSONResponse({{
        "status": "overloaded" if is_overloaded else "healthy",
        "model": "{config.model}",
        "task": "{config.task}",
        "queue_size": queue_size,
        "max_queue_size": MAX_QUEUE_SIZE,
        "total_requests": stats["total_requests"],
        "timestamp": datetime.now().isoformat()
    }})

async def startup_event():
    """Initialize the advanced server on startup."""
    logger.info("Starting advanced inference server...")
    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    app.model_queue = queue
    asyncio.create_task(server_loop_with_batching(queue))
    logger.info("Advanced inference server started successfully")

# Create application
routes = [
    Route("/", homepage, methods=["POST"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/stats", stats_endpoint, methods=["GET"]),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
]

app = Starlette(routes=routes, middleware=middleware)
app.model_queue = None

@app.on_event("startup")
async def startup():
    await startup_event()
'''
    
    def _generate_multi_task_server_code(self, config: WebServerConfig) -> str:
        """Generate multi-task server code."""
        # For simplicity, use advanced server code
        # In a real implementation, this would support multiple models
        return self._generate_advanced_server_code(config)


# Global instance for easy access
web_server_manager = WebServerManager()
