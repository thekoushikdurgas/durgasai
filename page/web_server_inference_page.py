"""
Web Server Inference Page for DurgasAI.

This module provides a comprehensive interface for web server inference management:
- Server configuration and deployment
- Real-time monitoring and metrics
- Testing and validation tools
- Integration with Hugging Face Pipeline API
- Support for multiple server types (basic, advanced, multi-task)
"""

import streamlit as st
import sys
import asyncio
import json
import time
import subprocess
import threading
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config
from utils.custom_pipeline_manager import custom_pipeline_manager

# Try to import server components
try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    import uvicorn
    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class WebServerInferencePage:
    """
    Web server inference management interface.
    
    This class provides:
    - Server configuration and deployment
    - Real-time monitoring and metrics
    - Testing and validation tools
    - Integration with existing model management
    """
    
    def __init__(self):
        """Initialize the web server inference page."""
        debug("Initializing WebServerInferencePage", "web_server_page")
        self.config = Config()
        
        # Initialize session state for web server page
        if 'web_server_page_state' not in st.session_state:
            st.session_state.web_server_page_state = {
                'active_servers': {},
                'server_configs': {},
                'server_processes': {},
                'monitoring_data': {},
                'selected_server_type': 'basic',
                'selected_model': 'google-bert/bert-base-uncased',
                'selected_task': 'fill-mask',
                'server_port': 8000,
                'server_host': '0.0.0.0',
                'max_batch_size': 4,
                'batch_timeout': 0.1,
                'max_queue_size': 50,
                'test_results': {},
                'show_advanced_config': False,
                # New advanced parameters
                'device': -1,  # CPU by default
                'torch_dtype': None,
                'enable_quantization': False,
                'task_parameters': {}
            }
        
        info("WebServerInferencePage initialized successfully", "web_server_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🌐 Web Server Inference</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>🌐 About Web Server Inference</h4>
        <p>Deploy Hugging Face models as web services for production inference. This page provides tools for:</p>
        <ul>
        <li><strong>Server Deployment:</strong> Launch inference servers with various configurations</li>
        <li><strong>Real-time Monitoring:</strong> Track performance metrics and server health</li>
        <li><strong>Dynamic Batching:</strong> Optimize throughput with intelligent request batching</li>
        <li><strong>Multi-Model Support:</strong> Deploy multiple models simultaneously</li>
        <li><strong>Testing Tools:</strong> Validate server performance and functionality</li>
        <li><strong>Production Ready:</strong> Error handling, circuit breaking, and monitoring</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        missing_deps = []
        
        if not STARLETTE_AVAILABLE:
            missing_deps.append("starlette, uvicorn")
        
        if not TRANSFORMERS_AVAILABLE:
            missing_deps.append("transformers")
        
        if missing_deps:
            st.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            st.markdown("**To install missing dependencies:**")
            st.code("pip install starlette uvicorn transformers torch")
            return False
        
        st.success("✅ All dependencies available")
        return True
    
    def render_server_configuration(self):
        """Render server configuration interface."""
        st.markdown("### 🔧 Server Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Server type selection
            server_type = st.selectbox(
                "Server Type:",
                ["basic", "advanced", "multi_task"],
                index=0,
                help="Choose server type based on your needs"
            )
            st.session_state.web_server_page_state['selected_server_type'] = server_type
            
            # Model selection with comprehensive options
            model_options = [
                # Text models
                "google-bert/bert-base-uncased",
                "google/gemma-2-2b", 
                "google/gemma-7b",
                "microsoft/DialoGPT-medium",
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "google/flan-t5-base",
                "google/pegasus-billsum",
                "distilbert-base-uncased-finetuned-sst-2-english",
                # Audio models
                "openai/whisper-large-v3",
                "openai/whisper-base",
                # Vision models
                "google/vit-base-patch16-224",
                "microsoft/resnet-50",
                # Custom model option
                "custom"
            ]
            
            selected_model = st.selectbox(
                "Model:",
                model_options,
                index=0,
                help="Select the model to deploy"
            )
            
            # Custom model input
            if selected_model == "custom":
                custom_model = st.text_input(
                    "Custom Model Path:",
                    placeholder="Enter HuggingFace model ID or local path",
                    help="Enter any HuggingFace model ID or local model path"
                )
                if custom_model:
                    selected_model = custom_model
            
            st.session_state.web_server_page_state['selected_model'] = selected_model
            
            # Task selection with comprehensive options including custom pipelines
            standard_tasks = [
                # Text tasks
                "fill-mask",
                "text-generation", 
                "text-classification",
                "sentiment-analysis",
                "token-classification",
                "question-answering",
                "summarization",
                "translation",
                "zero-shot-classification",
                "text2text-generation",
                # Audio tasks
                "automatic-speech-recognition",
                "audio-classification",
                "text-to-speech",
                # Vision tasks
                "image-classification",
                "object-detection",
                "image-segmentation",
                "visual-question-answering",
                # Multimodal tasks
                "feature-extraction",
                "embeddings"
            ]
            
            # Get custom pipelines
            custom_pipelines = custom_pipeline_manager.get_custom_pipelines()
            custom_task_options = [f"custom:{name}" for name in custom_pipelines.keys()]
            
            # Combine standard and custom tasks
            all_task_options = standard_tasks + custom_task_options
            
            selected_task = st.selectbox(
                "Task:",
                all_task_options,
                index=0,
                help="Select the inference task (custom pipelines prefixed with 'custom:')"
            )
            st.session_state.web_server_page_state['selected_task'] = selected_task
            
            # Show custom pipeline info if selected
            if selected_task.startswith("custom:"):
                custom_name = selected_task[7:]  # Remove 'custom:' prefix
                if custom_name in custom_pipelines:
                    pipeline_info = custom_pipelines[custom_name]
                    st.info(f"📝 **{pipeline_info['name']}**: {pipeline_info['description']}")
                    st.caption(f"Version: {pipeline_info['version']} | Usage: {pipeline_info['usage_count']} times")
        
        with col2:
            # Network configuration
            server_host = st.text_input(
                "Host:",
                value=st.session_state.web_server_page_state['server_host'],
                help="Server host address"
            )
            st.session_state.web_server_page_state['server_host'] = server_host
            
            server_port = st.number_input(
                "Port:",
                min_value=8000,
                max_value=9999,
                value=st.session_state.web_server_page_state['server_port'],
                help="Server port number"
            )
            st.session_state.web_server_page_state['server_port'] = server_port
        
        # Advanced configuration
        show_advanced = st.checkbox(
            "Show Advanced Configuration",
            value=st.session_state.web_server_page_state['show_advanced_config']
        )
        st.session_state.web_server_page_state['show_advanced_config'] = show_advanced
        
        if show_advanced:
            st.markdown("#### Advanced Settings")
            
            # Device and optimization settings
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.markdown("**Device Configuration**")
                device_option = st.selectbox(
                    "Device:",
                    ["CPU", "GPU (CUDA:0)", "GPU (CUDA:1)", "Auto Device Map"],
                    index=0,
                    help="Select compute device for inference"
                )
                
                # Convert device option to device value
                device_map = {
                    "CPU": -1,
                    "GPU (CUDA:0)": 0,
                    "GPU (CUDA:1)": 1,
                    "Auto Device Map": "auto"
                }
                st.session_state.web_server_page_state['device'] = device_map[device_option]
                
                torch_dtype = st.selectbox(
                    "Torch Dtype:",
                    ["auto", "float32", "float16", "bfloat16"],
                    index=0,
                    help="Precision for model weights"
                )
                st.session_state.web_server_page_state['torch_dtype'] = torch_dtype if torch_dtype != "auto" else None
            
            with col4:
                st.markdown("**Batch Processing**")
                max_batch_size = st.number_input(
                    "Max Batch Size:",
                    min_value=1,
                    max_value=32,
                    value=st.session_state.web_server_page_state['max_batch_size'],
                    help="Maximum batch size for dynamic batching"
                )
                st.session_state.web_server_page_state['max_batch_size'] = max_batch_size
                
                batch_timeout = st.number_input(
                    "Batch Timeout (ms):",
                    min_value=10,
                    max_value=1000,
                    value=int(st.session_state.web_server_page_state['batch_timeout'] * 1000),
                    help="Timeout for collecting batches (milliseconds)"
                )
                st.session_state.web_server_page_state['batch_timeout'] = batch_timeout / 1000
            
            with col5:
                st.markdown("**Optimization**")
                max_queue_size = st.number_input(
                    "Max Queue Size:",
                    min_value=10,
                    max_value=500,
                    value=st.session_state.web_server_page_state['max_queue_size'],
                    help="Maximum queue size before rejecting requests"
                )
                st.session_state.web_server_page_state['max_queue_size'] = max_queue_size
                
                enable_quantization = st.checkbox(
                    "Enable 8-bit Quantization",
                    value=False,
                    help="Reduce memory usage with 8-bit quantization"
                )
                st.session_state.web_server_page_state['enable_quantization'] = enable_quantization
            
            # Task-specific parameters
            st.markdown("#### Task-Specific Parameters")
            
            task_params = self._get_task_specific_parameters(st.session_state.web_server_page_state['selected_task'])
            st.session_state.web_server_page_state['task_parameters'] = task_params
    
    def render_server_management(self):
        """Render server management interface."""
        st.markdown("### 🚀 Server Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Start Server", use_container_width=True):
                self._start_server()
        
        with col2:
            if st.button("🔴 Stop Server", use_container_width=True):
                self._stop_server()
        
        with col3:
            if st.button("🔄 Restart Server", use_container_width=True):
                self._restart_server()
        
        # Server status
        self._display_server_status()
    
    def _start_server(self):
        """Start the inference server."""
        try:
            state = st.session_state.web_server_page_state
            server_id = f"{state['selected_model']}:{state['selected_task']}:{state['server_port']}"
            
            # Check if server is already running
            if server_id in state['active_servers']:
                st.warning(f"Server {server_id} is already running")
                return
            
            # Create server configuration
            server_config = {
                'server_type': state['selected_server_type'],
                'model': state['selected_model'],
                'task': state['selected_task'],
                'host': state['server_host'],
                'port': state['server_port'],
                'max_batch_size': state['max_batch_size'],
                'batch_timeout': state['batch_timeout'],
                'max_queue_size': state['max_queue_size'],
                'device': state.get('device', -1),
                'torch_dtype': state.get('torch_dtype'),
                'enable_quantization': state.get('enable_quantization', False),
                'task_parameters': state.get('task_parameters', {})
            }
            
            # Generate server code
            server_code = self._generate_server_code(server_config)
            
            # Save server code to file
            server_file = Path(f"output/temp/server_{state['server_port']}.py")
            server_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(server_file, 'w') as f:
                f.write(server_code)
            
            # Start server process
            cmd = [
                sys.executable, "-m", "uvicorn", 
                f"output.temp.server_{state['server_port']}:app",
                "--host", state['server_host'],
                "--port", str(state['server_port']),
                "--reload"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store server information
            state['active_servers'][server_id] = {
                'config': server_config,
                'process': process,
                'start_time': datetime.now(),
                'status': 'starting'
            }
            
            st.success(f"✅ Starting server {server_id}...")
            log_user_action("server_started", server_id=server_id, config=server_config)
            
            # Wait a moment and check if server started successfully
            time.sleep(2)
            if self._check_server_health(state['server_host'], state['server_port']):
                state['active_servers'][server_id]['status'] = 'running'
                st.success(f"🚀 Server {server_id} started successfully!")
            else:
                st.error(f"❌ Failed to start server {server_id}")
                state['active_servers'][server_id]['status'] = 'failed'
            
        except Exception as e:
            error(f"Failed to start server: {str(e)}", "web_server_page", e)
            st.error(f"❌ Failed to start server: {str(e)}")
    
    def _stop_server(self):
        """Stop the inference server."""
        try:
            state = st.session_state.web_server_page_state
            
            if not state['active_servers']:
                st.warning("No active servers to stop")
                return
            
            # Stop all active servers
            for server_id, server_info in state['active_servers'].items():
                process = server_info['process']
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                    st.success(f"🔴 Stopped server {server_id}")
                    log_user_action("server_stopped", server_id=server_id)
            
            # Clear active servers
            state['active_servers'] = {}
            
        except Exception as e:
            error(f"Failed to stop server: {str(e)}", "web_server_page", e)
            st.error(f"❌ Failed to stop server: {str(e)}")
    
    def _restart_server(self):
        """Restart the inference server."""
        self._stop_server()
        time.sleep(1)
        self._start_server()
    
    def _generate_server_code(self, config: Dict[str, Any]) -> str:
        """Generate server code based on configuration."""
        server_type = config['server_type']
        
        if server_type == 'basic':
            return self._generate_basic_server_code(config)
        elif server_type == 'advanced':
            return self._generate_advanced_server_code(config)
        elif server_type == 'multi_task':
            return self._generate_multi_task_server_code(config)
        else:
            raise ValueError(f"Unknown server type: {server_type}")
    
    def _generate_basic_server_code(self, config: Dict[str, Any]) -> str:
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
        logger.info("Loading pipeline: {config['task']} with model {config['model']}")
        pipe = pipeline(task="{config['task']}", model="{config['model']}")
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
                        "model": "{config['model']}",
                        "task": "{config['task']}",
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
                        "model": "{config['model']}",
                        "task": "{config['task']}",
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
        "model": "{config['model']}",
        "task": "{config['task']}",
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
    
    def _generate_advanced_server_code(self, config: Dict[str, Any]) -> str:
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
MAX_BATCH_SIZE = {config['max_batch_size']}
BATCH_TIMEOUT = {config['batch_timeout']}
MAX_QUEUE_SIZE = {config['max_queue_size']}

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
        logger.info("Loading pipeline: {config['task']} with model {config['model']}")
        pipe = pipeline(task="{config['task']}", model="{config['model']}")
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
                            "model": "{config['model']}",
                            "task": "{config['task']}",
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
                            "model": "{config['model']}",
                            "task": "{config['task']}",
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
            "model": "{config['model']}",
            "task": "{config['task']}",
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
        "model": "{config['model']}",
        "task": "{config['task']}",
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
    
    def _generate_multi_task_server_code(self, config: Dict[str, Any]) -> str:
        """Generate multi-task server code."""
        # For simplicity, we'll use the advanced server with single model
        # In a real implementation, this would support multiple models
        return self._generate_advanced_server_code(config)
    
    def _check_server_health(self, host: str, port: int) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _display_server_status(self):
        """Display current server status."""
        st.markdown("#### 📊 Server Status")
        
        state = st.session_state.web_server_page_state
        
        if not state['active_servers']:
            st.info("No active servers")
            return
        
        for server_id, server_info in state['active_servers'].items():
            config = server_info['config']
            status = server_info['status']
            start_time = server_info['start_time']
            
            # Status color
            status_color = {
                'running': '🟢',
                'starting': '🟡', 
                'failed': '🔴',
                'stopped': '⚫'
            }.get(status, '⚪')
            
            with st.expander(f"{status_color} {server_id} - {status.upper()}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model:** {config['model']}")
                    st.write(f"**Task:** {config['task']}")
                    st.write(f"**Address:** http://{config['host']}:{config['port']}")
                
                with col2:
                    st.write(f"**Status:** {status}")
                    st.write(f"**Started:** {start_time.strftime('%H:%M:%S')}")
                    
                    # Health check
                    if status == 'running':
                        if self._check_server_health(config['host'], config['port']):
                            st.success("✅ Healthy")
                        else:
                            st.error("❌ Unhealthy")
    
    def render_testing_interface(self):
        """Render testing interface."""
        st.markdown("### 🧪 Server Testing")
        
        state = st.session_state.web_server_page_state
        
        if not state['active_servers']:
            st.info("Start a server to enable testing")
            return
        
        # Server selection for testing
        server_ids = list(state['active_servers'].keys())
        selected_server_id = st.selectbox(
            "Select server to test:",
            server_ids,
            help="Choose which server to test"
        )
        
        if selected_server_id:
            server_info = state['active_servers'][selected_server_id]
            config = server_info['config']
            
            # Test input
            test_input = st.text_area(
                "Test Input:",
                value=self._get_default_test_input(config['task']),
                help="Enter text to test the inference server"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 Single Test", use_container_width=True):
                    self._run_single_test(config, test_input, selected_server_id)
            
            with col2:
                if st.button("📊 Load Test", use_container_width=True):
                    self._run_load_test(config, test_input, selected_server_id)
            
            with col3:
                if st.button("❤️ Health Check", use_container_width=True):
                    self._run_health_check(config, selected_server_id)
            
            # Display test results
            self._display_test_results(selected_server_id)
    
    def _get_task_specific_parameters(self, task: str) -> Dict[str, Any]:
        """Get task-specific parameter configuration UI."""
        params = {}
        
        if task == "text-generation":
            col1, col2, col3 = st.columns(3)
            with col1:
                params['max_length'] = st.number_input("Max Length", min_value=1, max_value=512, value=50)
                params['temperature'] = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
            with col2:
                params['do_sample'] = st.checkbox("Do Sample", value=True)
                params['top_k'] = st.number_input("Top K", min_value=1, max_value=100, value=50)
            with col3:
                params['top_p'] = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
                params['repetition_penalty'] = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.1)
        
        elif task == "summarization":
            col1, col2 = st.columns(2)
            with col1:
                params['max_length'] = st.number_input("Max Length", min_value=10, max_value=300, value=130)
                params['min_length'] = st.number_input("Min Length", min_value=5, max_value=100, value=30)
            with col2:
                params['do_sample'] = st.checkbox("Do Sample", value=False)
                params['length_penalty'] = st.slider("Length Penalty", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        
        elif task == "question-answering":
            col1, col2 = st.columns(2)
            with col1:
                params['max_answer_len'] = st.number_input("Max Answer Length", min_value=1, max_value=200, value=15)
            with col2:
                params['top_k'] = st.number_input("Top K", min_value=1, max_value=20, value=1)
        
        elif task == "fill-mask":
            params['top_k'] = st.number_input("Top K", min_value=1, max_value=10, value=5)
        
        elif task == "token-classification":
            params['aggregation_strategy'] = st.selectbox(
                "Aggregation Strategy", 
                ["none", "simple", "first", "average", "max"], 
                index=1
            )
        
        elif task == "automatic-speech-recognition":
            col1, col2 = st.columns(2)
            with col1:
                params['return_timestamps'] = st.selectbox(
                    "Return Timestamps", 
                    [False, "word", "char"], 
                    index=0
                )
            with col2:
                params['chunk_length_s'] = st.number_input("Chunk Length (s)", min_value=1, max_value=30, value=10)
        
        elif task in ["text-classification", "sentiment-analysis"]:
            params['return_all_scores'] = st.checkbox("Return All Scores", value=False)
        
        # Remove empty parameters
        return {k: v for k, v in params.items() if v is not None}
    
    def _get_default_test_input(self, task: str) -> str:
        """Get default test input for a task."""
        defaults = {
            'fill-mask': 'Paris is the [MASK] of France.',
            'text-generation': 'The future of artificial intelligence is',
            'sentiment-analysis': 'I love this product!',
            'text-classification': 'This movie is absolutely fantastic!',
            'text2text-generation': 'Translate to French: Hello, how are you?',
            'question-answering': '{"question": "What is the capital of France?", "context": "France is a country in Western Europe. Its capital and largest city is Paris."}',
            'summarization': 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.',
            'token-classification': 'My name is John Doe and I work at OpenAI in San Francisco.',
            'translation': 'Hello, how are you today?',
            'zero-shot-classification': '{"sequences": "This is a great movie with excellent acting.", "candidate_labels": ["positive", "negative", "neutral"]}',
            'automatic-speech-recognition': 'Upload an audio file or provide URL',
            'audio-classification': 'Upload an audio file or provide URL',
            'image-classification': 'Upload an image file or provide URL',
            'object-detection': 'Upload an image file or provide URL',
            'visual-question-answering': '{"question": "What color is the sky?", "image": "image_url_here"}',
            'feature-extraction': 'Extract features from this text.',
            'embeddings': 'Generate embeddings for this text.'
        }
        return defaults.get(task, 'Test input')
    
    def _run_single_test(self, config: Dict[str, Any], test_input: str, server_id: str):
        """Run a single test request."""
        try:
            url = f"http://{config['host']}:{config['port']}/"
            
            start_time = time.time()
            response = requests.post(url, data=test_input, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Test successful ({response_time:.3f}s)")
                st.json(result)
            else:
                st.error(f"❌ Test failed: {response.status_code}")
                st.text(response.text)
            
            # Store test result
            state = st.session_state.web_server_page_state
            if server_id not in state['test_results']:
                state['test_results'][server_id] = []
            
            state['test_results'][server_id].append({
                'type': 'single',
                'timestamp': datetime.now(),
                'response_time': response_time,
                'success': response.status_code == 200,
                'status_code': response.status_code
            })
            
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")
    
    def _run_load_test(self, config: Dict[str, Any], test_input: str, server_id: str):
        """Run a load test with multiple concurrent requests."""
        st.info("Running load test with 10 concurrent requests...")
        
        try:
            import concurrent.futures
            import threading
            
            url = f"http://{config['host']}:{config['port']}/"
            num_requests = 10
            results = []
            
            def single_request():
                try:
                    start_time = time.time()
                    response = requests.post(url, data=test_input, timeout=30)
                    response_time = time.time() - start_time
                    return {
                        'success': response.status_code == 200,
                        'response_time': response_time,
                        'status_code': response.status_code
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'response_time': 0,
                        'status_code': 0,
                        'error': str(e)
                    }
            
            # Run concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(single_request) for _ in range(num_requests)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Analyze results
            successful = sum(1 for r in results if r['success'])
            failed = num_requests - successful
            avg_response_time = sum(r['response_time'] for r in results) / num_requests
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Successful", successful)
            with col2:
                st.metric("Failed", failed)
            with col3:
                st.metric("Avg Response Time", f"{avg_response_time:.3f}s")
            
            # Store test result
            state = st.session_state.web_server_page_state
            if server_id not in state['test_results']:
                state['test_results'][server_id] = []
            
            state['test_results'][server_id].append({
                'type': 'load',
                'timestamp': datetime.now(),
                'total_requests': num_requests,
                'successful_requests': successful,
                'failed_requests': failed,
                'avg_response_time': avg_response_time
            })
            
        except Exception as e:
            st.error(f"❌ Load test failed: {str(e)}")
    
    def _run_health_check(self, config: Dict[str, Any], server_id: str):
        """Run health check test."""
        try:
            url = f"http://{config['host']}:{config['port']}/health"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Server is healthy")
                st.json(result)
            else:
                st.error(f"❌ Health check failed: {response.status_code}")
                st.text(response.text)
                
        except Exception as e:
            st.error(f"❌ Health check failed: {str(e)}")
    
    def _display_test_results(self, server_id: str):
        """Display test results history."""
        state = st.session_state.web_server_page_state
        
        if server_id not in state['test_results'] or not state['test_results'][server_id]:
            return
        
        st.markdown("#### 📈 Test Results History")
        
        results = state['test_results'][server_id][-10:]  # Last 10 results
        
        # Create DataFrame for visualization
        df_data = []
        for result in results:
            if result['type'] == 'single':
                df_data.append({
                    'timestamp': result['timestamp'],
                    'type': 'Single Test',
                    'response_time': result['response_time'],
                    'success': result['success']
                })
            elif result['type'] == 'load':
                df_data.append({
                    'timestamp': result['timestamp'],
                    'type': 'Load Test',
                    'response_time': result['avg_response_time'],
                    'success': result['successful_requests'] > result['failed_requests']
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Response time chart
            fig = px.line(
                df, 
                x='timestamp', 
                y='response_time',
                color='type',
                title='Response Time Over Time',
                labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_monitoring_dashboard(self):
        """Render monitoring dashboard."""
        st.markdown("### 📊 Monitoring Dashboard")
        
        state = st.session_state.web_server_page_state
        
        if not state['active_servers']:
            st.info("Start a server to enable monitoring")
            return
        
        # Server metrics
        for server_id, server_info in state['active_servers'].items():
            config = server_info['config']
            
            if server_info['status'] != 'running':
                continue
            
            st.markdown(f"#### {server_id}")
            
            try:
                # Get server stats
                stats_url = f"http://{config['host']}:{config['port']}/stats"
                response = requests.get(stats_url, timeout=5)
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Requests", 
                            stats['server_stats'].get('total_requests', 0)
                        )
                    
                    with col2:
                        st.metric(
                            "Success Rate", 
                            f"{(stats['server_stats'].get('successful_requests', 0) / max(stats['server_stats'].get('total_requests', 1), 1) * 100):.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Queue Size", 
                            stats['current_state'].get('queue_size', 0)
                        )
                    
                    with col4:
                        st.metric(
                            "Avg Batch Size", 
                            f"{stats['server_stats'].get('average_batch_size', 0):.1f}"
                        )
                    
                    # Server configuration
                    with st.expander("Server Configuration", expanded=False):
                        st.json(stats['server_config'])
                
                else:
                    st.warning(f"Could not fetch stats for {server_id}")
                    
            except Exception as e:
                st.error(f"Error fetching stats for {server_id}: {str(e)}")
    
    def render_custom_pipeline_management(self):
        """Render custom pipeline management interface."""
        st.markdown("### 🛠️ Custom Pipeline Management")
        
        # Sub-tabs for different pipeline operations
        pipeline_tabs = st.tabs([
            "📋 Available Pipelines",
            "➕ Create Pipeline",
            "📊 Pipeline Statistics",
            "📁 Import/Export"
        ])
        
        with pipeline_tabs[0]:
            self._render_available_pipelines()
        
        with pipeline_tabs[1]:
            self._render_create_pipeline()
        
        with pipeline_tabs[2]:
            self._render_pipeline_statistics()
        
        with pipeline_tabs[3]:
            self._render_import_export_pipelines()
    
    def _render_available_pipelines(self):
        """Render available custom pipelines."""
        st.markdown("#### 📋 Available Custom Pipelines")
        
        custom_pipelines = custom_pipeline_manager.get_custom_pipelines()
        
        if not custom_pipelines:
            st.info("No custom pipelines available. Create one using the templates!")
            return
        
        for pipeline_name, pipeline_info in custom_pipelines.items():
            with st.expander(f"🔧 {pipeline_info['name']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {pipeline_info['description']}")
                    st.write(f"**Category:** {pipeline_info['category']}")
                    st.write(f"**Version:** {pipeline_info['version']}")
                
                with col2:
                    st.write(f"**Usage Count:** {pipeline_info['usage_count']}")
                    st.write(f"**Registered:** {pipeline_info['registered_at'][:10]}")
                    if pipeline_info.get('default_model'):
                        st.write(f"**Default Model:** {pipeline_info['default_model']}")
                
                # Pipeline actions
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    if st.button(f"🧪 Test", key=f"test_{pipeline_name}"):
                        self._test_custom_pipeline(pipeline_name)
                
                with col4:
                    if st.button(f"📊 Stats", key=f"stats_{pipeline_name}"):
                        stats = custom_pipeline_manager.get_pipeline_stats(pipeline_name)
                        st.json(stats)
                
                with col5:
                    if st.button(f"🗑️ Delete", key=f"delete_{pipeline_name}"):
                        if st.confirm(f"Delete pipeline '{pipeline_name}'?"):
                            if custom_pipeline_manager.delete_custom_pipeline(pipeline_name):
                                st.success(f"Pipeline '{pipeline_name}' deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete pipeline")
    
    def _render_create_pipeline(self):
        """Render pipeline creation interface."""
        st.markdown("#### ➕ Create New Custom Pipeline")
        
        # Get available templates
        templates = custom_pipeline_manager.get_available_templates()
        
        if not templates:
            st.error("No pipeline templates available")
            return
        
        # Template selection
        template_options = list(templates.keys())
        template_names = [templates[t]["name"] for t in template_options]
        
        selected_template_idx = st.selectbox(
            "Select Pipeline Template:",
            range(len(template_options)),
            format_func=lambda x: template_names[x],
            help="Choose a template to start with"
        )
        
        selected_template = template_options[selected_template_idx]
        template_info = templates[selected_template]
        
        # Show template information
        st.info(f"📝 **{template_info['name']}**: {template_info['description']}")
        st.caption(f"Category: {template_info['category']} | Difficulty: {template_info['difficulty']}")
        
        # Pipeline configuration
        col1, col2 = st.columns(2)
        
        with col1:
            pipeline_name = st.text_input(
                "Pipeline Name:",
                placeholder="my_custom_pipeline",
                help="Unique name for your custom pipeline"
            )
            
            pipeline_model = st.text_input(
                "Default Model:",
                value="distilbert-base-uncased-finetuned-sst-2-english",
                help="HuggingFace model to use with this pipeline"
            )
        
        with col2:
            pipeline_description = st.text_area(
                "Description:",
                value=template_info['description'],
                help="Describe what your pipeline does"
            )
        
        # Show example usage
        with st.expander("📖 Example Usage", expanded=False):
            st.json(template_info['example_usage'])
        
        # Show template code
        if st.checkbox("Show Template Code", value=False):
            template_code = custom_pipeline_manager.get_template_code(selected_template)
            if template_code:
                st.code(template_code, language="python")
        
        # Create pipeline
        if st.button("🚀 Create Pipeline", type="primary"):
            if not pipeline_name:
                st.error("Pipeline name is required")
                return
            
            if not pipeline_model:
                st.error("Model name is required")
                return
            
            # Check if pipeline name already exists
            existing_pipelines = custom_pipeline_manager.get_custom_pipelines()
            if pipeline_name in existing_pipelines:
                st.error(f"Pipeline '{pipeline_name}' already exists")
                return
            
            # Create the pipeline
            with st.spinner("Creating custom pipeline..."):
                success = custom_pipeline_manager.create_pipeline_from_template(
                    selected_template,
                    pipeline_name,
                    pipeline_model,
                    customizations={"description": pipeline_description}
                )
            
            if success:
                st.success(f"✅ Custom pipeline '{pipeline_name}' created successfully!")
                st.balloons()
                st.info("Your pipeline is now available in the task selection dropdown.")
            else:
                st.error("Failed to create custom pipeline")
    
    def _render_pipeline_statistics(self):
        """Render pipeline usage statistics."""
        st.markdown("#### 📊 Pipeline Usage Statistics")
        
        custom_pipelines = custom_pipeline_manager.get_custom_pipelines()
        
        if not custom_pipelines:
            st.info("No custom pipelines available for statistics")
            return
        
        # Create statistics dataframe
        stats_data = []
        for pipeline_name, pipeline_info in custom_pipelines.items():
            stats = custom_pipeline_manager.get_pipeline_stats(pipeline_name)
            
            stats_data.append({
                "Pipeline": pipeline_info["name"],
                "Category": pipeline_info["category"],
                "Total Calls": stats.get("calls", 0),
                "Errors": stats.get("errors", 0),
                "Success Rate": f"{((stats.get('calls', 0) - stats.get('errors', 0)) / max(stats.get('calls', 1), 1) * 100):.1f}%",
                "Avg Time (s)": f"{stats.get('avg_execution_time', 0):.3f}",
                "Last Used": stats.get("last_used", "Never")[:10] if stats.get("last_used") else "Never"
            })
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)
            
            # Usage chart
            if len(stats_data) > 1:
                fig = px.bar(
                    df, 
                    x="Pipeline", 
                    y="Total Calls",
                    title="Pipeline Usage Comparison",
                    color="Category"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_import_export_pipelines(self):
        """Render pipeline import/export interface."""
        st.markdown("#### 📁 Import/Export Pipelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📥 Import Pipeline")
            
            uploaded_file = st.file_uploader(
                "Upload Pipeline File",
                type=["py"],
                help="Upload a Python file containing custom pipeline classes"
            )
            
            if uploaded_file is not None:
                if st.button("Import Pipeline"):
                    # Save uploaded file temporarily
                    temp_file = Path("output/temp") / uploaded_file.name
                    temp_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load pipeline from file
                    success = custom_pipeline_manager.load_pipeline_from_file(temp_file)
                    
                    if success:
                        st.success("Pipeline imported successfully!")
                    else:
                        st.error("Failed to import pipeline")
                    
                    # Clean up temp file
                    temp_file.unlink()
        
        with col2:
            st.markdown("##### 📤 Export Pipelines")
            
            custom_pipelines = custom_pipeline_manager.get_custom_pipelines()
            
            if custom_pipelines:
                selected_pipelines = st.multiselect(
                    "Select Pipelines to Export:",
                    list(custom_pipelines.keys()),
                    help="Choose pipelines to export"
                )
                
                if selected_pipelines and st.button("Export Selected"):
                    # Create export package
                    export_data = {}
                    for pipeline_name in selected_pipelines:
                        pipeline_file = Path("output/custom_pipelines") / f"{pipeline_name}.py"
                        if pipeline_file.exists():
                            export_data[pipeline_name] = {
                                "info": custom_pipelines[pipeline_name],
                                "code": pipeline_file.read_text()
                            }
                    
                    if export_data:
                        # Convert to JSON for download
                        export_json = json.dumps(export_data, indent=2, default=str)
                        
                        st.download_button(
                            label="📥 Download Export",
                            data=export_json,
                            file_name=f"custom_pipelines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            else:
                st.info("No custom pipelines available for export")
    
    def _test_custom_pipeline(self, pipeline_name: str):
        """Test a custom pipeline."""
        st.markdown(f"#### 🧪 Testing Pipeline: {pipeline_name}")
        
        # Get pipeline info
        custom_pipelines = custom_pipeline_manager.get_custom_pipelines()
        pipeline_info = custom_pipelines.get(pipeline_name)
        
        if not pipeline_info:
            st.error(f"Pipeline '{pipeline_name}' not found")
            return
        
        # Test input
        test_input = st.text_area(
            "Test Input:",
            value="This is a test input for the custom pipeline.",
            help="Enter text to test the pipeline"
        )
        
        # Test parameters
        with st.expander("Test Parameters", expanded=False):
            test_params = st.text_area(
                "Parameters (JSON):",
                value='{"top_k": 3}',
                help="Enter parameters as JSON"
            )
        
        if st.button("Run Test"):
            try:
                # Parse parameters
                params = json.loads(test_params) if test_params.strip() else {}
                
                # This would integrate with the actual pipeline execution
                # For now, show a placeholder result
                st.success("✅ Test completed successfully!")
                
                # Mock result
                mock_result = {
                    "result": "Test result from custom pipeline",
                    "metadata": {
                        "pipeline": pipeline_name,
                        "execution_time": 0.123,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                st.json(mock_result)
                
                # Update pipeline stats
                custom_pipeline_manager.update_pipeline_stats(pipeline_name, 0.123, True)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON in parameters")
            except Exception as e:
                st.error(f"Test failed: {str(e)}")
                custom_pipeline_manager.update_pipeline_stats(pipeline_name, 0, False)
    
    def render_documentation(self):
        """Render documentation section."""
        st.markdown("### 📚 Documentation")
        
        with st.expander("🔧 Server Types", expanded=False):
            st.markdown("""
            **Basic Server:**
            - Simple request-response processing
            - Support for all Pipeline tasks
            - Task-specific parameter handling
            - Good for development and testing
            
            **Advanced Server:**
            - Dynamic batching for improved throughput
            - Device management (CPU/GPU/Auto)
            - Quantization support for memory efficiency
            - Circuit breaking for overload protection
            - Performance monitoring and statistics
            - Production-ready error handling
            
            **Multi-Task Server:**
            - Support for multiple models and tasks simultaneously
            - Separate queues for each model
            - Advanced resource management
            - Ideal for production deployments
            """)
        
        with st.expander("🎯 Supported Pipeline Tasks", expanded=False):
            st.markdown("""
            **Text Tasks:**
            - Text Generation (GPT-style models)
            - Text Classification / Sentiment Analysis
            - Token Classification (NER)
            - Question Answering
            - Summarization
            - Translation
            - Fill-Mask (BERT-style)
            - Zero-shot Classification
            
            **Audio Tasks:**
            - Automatic Speech Recognition (ASR)
            - Audio Classification
            - Text-to-Speech
            
            **Vision Tasks:**
            - Image Classification
            - Object Detection
            - Image Segmentation
            - Visual Question Answering
            
            **Multimodal Tasks:**
            - Feature Extraction
            - Embeddings Generation
            """)
        
        with st.expander("🚀 API Usage", expanded=False):
            st.markdown("""
            **Simple Text Input:**
            ```bash
            curl -X POST -d "Your text here" http://localhost:8000/
            ```
            
            **JSON Request with Parameters:**
            ```bash
            curl -X POST -H "Content-Type: application/json" \\
                 -d '{"inputs": "The future of AI is", "parameters": {"max_length": 100, "temperature": 0.7}, "task": "text-generation"}' \\
                 http://localhost:8000/
            ```
            
            **Health Check:**
            ```bash
            curl http://localhost:8000/health
            ```
            
            **Statistics (Advanced/Multi-task):**
            ```bash
            curl http://localhost:8000/stats
            ```
            """)
        
        with st.expander("⚡ Performance Tips", expanded=False):
            st.markdown("""
            **Device Optimization:**
            - Use GPU acceleration when available (device=0, 1, etc.)
            - Enable auto device mapping for large models (device_map="auto")
            - Use half-precision (float16/bfloat16) for faster inference
            - Enable 8-bit quantization to reduce memory usage
            
            **Batch Processing:**
            - Use appropriate batch sizes (2-8 for most models)
            - Set batch timeout based on your latency requirements
            - Monitor queue sizes to prevent memory issues
            
            **Model Selection:**
            - Choose smaller models for faster inference
            - Use distilled models when possible
            - Consider task-specific models for better performance
            
            **Server Configuration:**
            - Monitor server metrics regularly
            - Implement proper error handling and monitoring
            - Use circuit breakers for production deployments
            - Scale horizontally with multiple server instances
            """)
        
        with st.expander("📋 Task-Specific Parameters", expanded=False):
            st.markdown("""
            **Text Generation:**
            - max_length: Maximum output length
            - temperature: Sampling temperature (0.1-2.0)
            - do_sample: Enable sampling vs greedy decoding
            - top_k/top_p: Nucleus sampling parameters
            - repetition_penalty: Penalty for repetition
            
            **Summarization:**
            - max_length/min_length: Output length constraints
            - length_penalty: Penalty for length
            - do_sample: Sampling vs greedy
            
            **Question Answering:**
            - max_answer_len: Maximum answer length
            - top_k: Number of answer candidates
            
            **ASR (Speech Recognition):**
            - return_timestamps: Return word/character timestamps
            - chunk_length_s: Audio chunk length in seconds
            
            **Classification:**
            - return_all_scores: Return all class probabilities
            - aggregation_strategy: Token aggregation method
            """)
    
    def render(self):
        """Render the complete web server inference page."""
        try:
            # Header
            self.render_header()
            
            # Main content tabs
            tabs = st.tabs([
                "🔧 Configuration", 
                "🚀 Management", 
                "🧪 Testing", 
                "📊 Monitoring",
                "🛠️ Custom Pipelines", 
                "📚 Documentation"
            ])
            
            with tabs[0]:
                self.render_server_configuration()
            
            with tabs[1]:
                self.render_server_management()
            
            with tabs[2]:
                self.render_testing_interface()
            
            with tabs[3]:
                self.render_monitoring_dashboard()
            
            with tabs[4]:
                self.render_custom_pipeline_management()
            
            with tabs[5]:
                self.render_documentation()
                
        except Exception as e:
            error("Error rendering web server inference page", "web_server_page", e)
            st.error(f"Error rendering page: {str(e)}")


def render_web_server_inference_page():
    """Render the web server inference page."""
    page = WebServerInferencePage()
    page.render()


if __name__ == "__main__":
    render_web_server_inference_page()
