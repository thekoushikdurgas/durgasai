"""
ML Apps Page for DurgasAI.

This module provides a comprehensive interface for ML app management:
- Task selection and configuration
- Model selection and optimization
- Real-time ML app creation and deployment
- Performance monitoring and optimization
- Memory usage analysis
- Deployment management
- Integration with enhanced ML apps manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json
import pandas as pd
import webbrowser

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.ml_apps_manager import (
    ml_apps_manager, 
    MLAppConfig, 
    MLAppResult,
    MLAppInfo
)
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


class MLAppsPage:
    """
    Comprehensive ML app management interface.
    
    This class provides:
    - Task selection and configuration
    - Model selection and optimization
    - Real-time ML app creation and deployment
    - Performance monitoring and optimization
    - Memory usage analysis
    - Deployment management
    - Integration with enhanced ML apps manager
    """
    
    def __init__(self):
        """Initialize the ML apps page."""
        debug("Initializing MLAppsPage", "ml_apps_page")
        self.config = Config()
        
        # Initialize session state for ML apps page
        if 'ml_apps_page_state' not in st.session_state:
            st.session_state.ml_apps_page_state = {
                'selected_task': 'text-generation',
                'selected_model': 'google/gemma-2-2b',
                'selected_device': -1,
                'batch_size': 1,
                'torch_dtype': None,
                'load_in_8bit': False,
                'app_title': '',
                'app_description': '',
                'server_port': 7860,
                'share_app': False,
                'show_advanced': False,
                'performance_stats': None,
                'active_deployments': {},
                'ml_app_info': None
            }
        
        info("MLAppsPage initialized successfully", "ml_apps_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🚀 ML Apps Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>🚀 About Machine Learning Apps</h4>
        <p>Hugging Face ML Apps provide powerful integration between Transformers pipelines and Gradio, enabling rapid creation of interactive web interfaces for machine learning models. This page provides advanced tools for:</p>
        <ul>
        <li><strong>App Creation:</strong> Create ML apps from pipelines in minutes</li>
        <li><strong>Task Selection:</strong> Choose from text, vision, audio, and multimodal tasks</li>
        <li><strong>Model Configuration:</strong> Optimize models for specific tasks and hardware</li>
        <li><strong>Real-time Deployment:</strong> Deploy apps locally or share publicly</li>
        <li><strong>Performance Monitoring:</strong> Monitor app performance and usage</li>
        <li><strong>Hardware Optimization:</strong> GPU acceleration and memory optimization</li>
        <li><strong>Deployment Management:</strong> Manage multiple active deployments</li>
        <li><strong>App Caching:</strong> Intelligent caching for improved performance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_task_selection(self):
        """Render task selection interface."""
        st.markdown("### 🎯 Select Task")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Available tasks
            available_tasks = ml_apps_manager.get_available_tasks()
            
            # Group tasks by category
            task_categories = {
                "Text Tasks": [t for t in available_tasks if t in [
                    "text-generation", "text-classification", "question-answering", 
                    "summarization", "translation", "conversational", "fill-mask"
                ]],
                "Vision Tasks": [t for t in available_tasks if t in [
                    "image-classification", "object-detection", "image-segmentation",
                    "visual-question-answering"
                ]],
                "Audio Tasks": [t for t in available_tasks if t in [
                    "automatic-speech-recognition", "text-to-speech", "audio-classification"
                ]],
                "Feature Extraction": [t for t in available_tasks if t in [
                    "feature-extraction"
                ]]
            }
            
            selected_task = st.selectbox(
                "Choose a task:",
                options=available_tasks,
                index=available_tasks.index(st.session_state.ml_apps_page_state['selected_task']) 
                if st.session_state.ml_apps_page_state['selected_task'] in available_tasks else 0,
                help="Select a machine learning task for the ML app"
            )
        
        with col2:
            # Task info display
            st.markdown("**Task Information:**")
            if selected_task:
                task_descriptions = {
                    "text-generation": "Generate text from prompts",
                    "text-classification": "Classify text into categories",
                    "question-answering": "Answer questions from context",
                    "summarization": "Summarize long texts",
                    "translation": "Translate between languages",
                    "image-classification": "Classify images",
                    "object-detection": "Detect objects in images",
                    "automatic-speech-recognition": "Convert speech to text",
                    "feature-extraction": "Extract features/embeddings"
                }
                
                description = task_descriptions.get(selected_task, "Machine learning task")
                st.write(f"**Description:** {description}")
                
                # Show task category
                for category, tasks in task_categories.items():
                    if selected_task in tasks:
                        st.write(f"**Category:** {category}")
                        break
        
        # Update session state
        st.session_state.ml_apps_page_state['selected_task'] = selected_task
        
        return selected_task
    
    def render_model_selection(self):
        """Render model selection interface."""
        st.markdown("### 🤖 Select Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular models by task
            task_models = {
                "text-generation": [
                    "google/gemma-2-2b",
                    "gpt2",
                    "microsoft/DialoGPT-medium",
                    "facebook/blenderbot-400M-distill"
                ],
                "text-classification": [
                    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    "nlptown/bert-base-multilingual-uncased-sentiment",
                    "cardiffnlp/twitter-roberta-base-sentiment-latest"
                ],
                "question-answering": [
                    "distilbert/distilbert-base-cased-distilled-squad",
                    "deepset/roberta-base-squad2",
                    "bert-large-uncased-whole-word-masking-finetuned-squad"
                ],
                "image-classification": [
                    "google/vit-base-patch16-224",
                    "microsoft/resnet-50",
                    "facebook/convnext-tiny-224"
                ],
                "object-detection": [
                    "facebook/detr-resnet-50",
                    "microsoft/table-transformer-structure-recognition"
                ],
                "automatic-speech-recognition": [
                    "openai/whisper-tiny",
                    "facebook/wav2vec2-base-960h",
                    "jonatasgrosman/wav2vec2-large-xlsr-53-english"
                ]
            }
            
            selected_task = st.session_state.ml_apps_page_state['selected_task']
            available_models = task_models.get(selected_task, ["default"])
            
            selected_model = st.selectbox(
                "Choose a model:",
                options=available_models,
                index=0,
                help="Select a pre-trained model for the task"
            )
            
            # Custom model input
            custom_model = st.text_input(
                "Or enter custom model ID:",
                placeholder="e.g., google/gemma-2-2b",
                help="Enter any HuggingFace model ID"
            )
            
            if custom_model:
                selected_model = custom_model
        
        with col2:
            # Model info display
            st.markdown("**Model Information:**")
            if selected_model and selected_model != "default":
                try:
                    st.write(f"**Model ID:** {selected_model}")
                    st.write(f"**Task:** {selected_task}")
                    
                    # Show model size estimation
                    model_sizes = {
                        "tiny": "< 100M parameters",
                        "base": "100M - 500M parameters", 
                        "large": "500M - 1B parameters",
                        "xl": "1B+ parameters"
                    }
                    
                    size_estimate = "Unknown size"
                    for size_key in model_sizes:
                        if size_key in selected_model.lower():
                            size_estimate = model_sizes[size_key]
                            break
                    
                    st.write(f"**Estimated Size:** {size_estimate}")
                except Exception as e:
                    st.info(f"Model info unavailable: {e}")
        
        # Update session state
        st.session_state.ml_apps_page_state['selected_model'] = selected_model
        
        return selected_model
    
    def render_device_configuration(self):
        """Render device configuration interface."""
        st.markdown("### 🖥️ Device Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hardware Selection:**")
            
            # Device selection
            device_options = {
                "CPU": -1,
                "Auto-detect GPU": 0
            }
            
            # Add specific GPUs if available
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        device_options[f"GPU {i} ({torch.cuda.get_device_name(i)})"] = i
            except ImportError:
                pass
            
            selected_device_label = st.selectbox(
                "Select device:",
                options=list(device_options.keys()),
                index=list(device_options.values()).index(st.session_state.ml_apps_page_state['selected_device']) 
                if st.session_state.ml_apps_page_state['selected_device'] in device_options.values() else 0,
                help="Select hardware for ML app execution"
            )
            
            selected_device = device_options[selected_device_label]
        
        with col2:
            st.markdown("**Memory Optimization:**")
            
            # Batch size
            batch_size = st.number_input(
                "Batch Size:",
                min_value=1,
                max_value=32,
                value=st.session_state.ml_apps_page_state['batch_size'],
                help="Number of inputs to process simultaneously"
            )
            
            # Precision
            torch_dtype = st.selectbox(
                "Precision:",
                options=["None", "float16", "bfloat16", "float32"],
                index=["None", "float16", "bfloat16", "float32"].index(
                    st.session_state.ml_apps_page_state['torch_dtype'] or "None"
                ),
                help="Floating point precision (lower = less memory)"
            )
            
            torch_dtype = torch_dtype if torch_dtype != "None" else None
            
            # Quantization
            load_in_8bit = st.checkbox(
                "8-bit Quantization",
                value=st.session_state.ml_apps_page_state['load_in_8bit'],
                help="Reduce memory usage with 8-bit quantization"
            )
        
        # Update session state
        st.session_state.ml_apps_page_state.update({
            'selected_device': selected_device,
            'batch_size': batch_size,
            'torch_dtype': torch_dtype,
            'load_in_8bit': load_in_8bit
        })
        
        return selected_device, batch_size, torch_dtype, load_in_8bit
    
    def render_app_configuration(self):
        """Render app configuration interface."""
        st.markdown("### 🎨 App Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**App Details:**")
            
            # App title
            app_title = st.text_input(
                "App Title:",
                value=st.session_state.ml_apps_page_state.get('app_title', ''),
                placeholder="Enter a title for your ML app",
                help="Title displayed in the app interface"
            )
            
            # App description
            app_description = st.text_area(
                "App Description:",
                value=st.session_state.ml_apps_page_state.get('app_description', ''),
                placeholder="Enter a description for your ML app",
                help="Description displayed in the app interface"
            )
        
        with col2:
            st.markdown("**Deployment Options:**")
            
            # Server port
            server_port = st.number_input(
                "Server Port:",
                min_value=1024,
                max_value=65535,
                value=st.session_state.ml_apps_page_state.get('server_port', 7860),
                help="Port for the ML app server"
            )
            
            # Share option
            share_app = st.checkbox(
                "Share App Publicly",
                value=st.session_state.ml_apps_page_state.get('share_app', False),
                help="Create a public link to share the app"
            )
            
            # Auto-open browser
            auto_open = st.checkbox(
                "Auto-open in Browser",
                value=True,
                help="Automatically open the app in your browser when deployed"
            )
        
        # Update session state
        st.session_state.ml_apps_page_state.update({
            'app_title': app_title,
            'app_description': app_description,
            'server_port': server_port,
            'share_app': share_app
        })
        
        return app_title, app_description, server_port, share_app, auto_open
    
    def render_app_controls(self, selected_task: str, selected_model: str, 
                           selected_device: int, app_title: str, app_description: str,
                           server_port: int, share_app: bool, auto_open: bool):
        """Render app control buttons."""
        st.markdown("### 🚀 App Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Create & Deploy App", type="primary"):
                if selected_task and selected_model:
                    self.create_and_deploy_app(
                        selected_task, selected_model, selected_device,
                        app_title, app_description, server_port, share_app, auto_open
                    )
                else:
                    st.error("Please select task and model for app creation")
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = ml_apps_manager.get_performance_stats()
                st.session_state.ml_apps_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                ml_apps_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("ml_apps_cache_cleared")
        
        with col4:
            if st.button("🔄 Refresh"):
                st.rerun()
    
    def create_and_deploy_app(self, selected_task: str, selected_model: str, 
                             selected_device: int, app_title: str, app_description: str,
                             server_port: int, share_app: bool, auto_open: bool):
        """Create and deploy ML app."""
        with st.spinner("Creating and deploying ML app..."):
            try:
                # Create ML app configuration
                config = MLAppConfig(
                    task=selected_task,
                    model=selected_model,
                    device=selected_device,
                    batch_size=st.session_state.ml_apps_page_state['batch_size'],
                    torch_dtype=st.session_state.ml_apps_page_state['torch_dtype'],
                    load_in_8bit=st.session_state.ml_apps_page_state['load_in_8bit'],
                    title=app_title or f"{selected_task.title()} App",
                    description=app_description or f"A {selected_task} application",
                    server_port=server_port,
                    share=share_app
                )
                
                # Deploy the app
                result = ml_apps_manager.deploy_ml_app(config, auto_open=auto_open)
                
                if result and result.success:
                    st.session_state.ml_apps_page_state['active_deployments'] = ml_apps_manager.list_active_deployments()
                    st.success(f"✅ ML app deployed successfully!")
                    
                    if result.deployment_url:
                        st.info(f"🌐 App URL: {result.deployment_url}")
                        if st.button("🔗 Open App"):
                            try:
                                webbrowser.open(result.deployment_url)
                            except Exception as e:
                                st.error(f"Failed to open browser: {e}")
                    
                    log_user_action("ml_app_deployed", 
                                  task=selected_task, 
                                  model=selected_model,
                                  deployment_url=result.deployment_url)
                else:
                    error_msg = result.error if result else "Unknown error"
                    st.error(f"❌ ML app deployment failed: {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ ML app deployment failed: {str(e)}")
                error(f"ML app deployment failed", "ml_apps_page", e)
    
    def render_active_deployments(self):
        """Render active deployments management."""
        active_deployments = ml_apps_manager.list_active_deployments()
        
        if active_deployments:
            st.markdown("### 🌐 Active Deployments")
            
            for cache_key, deployment_info in active_deployments.items():
                with st.expander(f"App: {deployment_info['config'].task} ({deployment_info['config'].model})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Task:** {deployment_info['config'].task}")
                        st.write(f"**Model:** {deployment_info['config'].model}")
                        st.write(f"**Device:** {deployment_info['config'].device}")
                    
                    with col2:
                        st.write(f"**URL:** {deployment_info.get('deployment_url', 'Unknown')}")
                        st.write(f"**Started:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deployment_info['started_at']))}")
                    
                    with col3:
                        if st.button(f"🔗 Open", key=f"open_{cache_key}"):
                            try:
                                webbrowser.open(deployment_info['deployment_url'])
                            except Exception as e:
                                st.error(f"Failed to open: {e}")
                        
                        if st.button(f"⏹️ Stop", key=f"stop_{cache_key}"):
                            if ml_apps_manager.stop_deployment(cache_key):
                                st.success("Deployment stopped successfully")
                                st.rerun()
                            else:
                                st.error("Failed to stop deployment")
        else:
            st.info("No active deployments. Create and deploy an app to see it here.")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.ml_apps_page_state.get('performance_stats')
        
        if stats:
            st.markdown("### 📊 Performance Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total App Creations", stats['total_app_creations'])
                st.metric("Total Deployments", stats['total_deployments'])
                st.metric("Active Deployments", stats['active_deployments'])
                st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1f}%")
            
            with col2:
                st.metric("Avg Creation Time", f"{stats['average_creation_time']:.3f}s")
                st.metric("Avg Deployment Time", f"{stats['average_deployment_time']:.3f}s")
                st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
                st.metric("Deployment Success Rate", f"{stats['deployment_success_rate']:.1f}%")
            
            with col3:
                st.metric("Memory Cache Size", stats['memory_cache_size'])
                st.metric("Disk Cache Size", stats['disk_cache_size'])
                st.metric("Device Switches", stats['device_switches'])
                st.metric("Memory Optimizations", stats['memory_optimizations'])
                st.metric("Batch Optimizations", stats['batch_optimizations'])
                st.metric("Quantization Usage", stats['quantization_usage'])
    
    def render_cached_apps(self):
        """Render cached apps information."""
        cached_apps = ml_apps_manager.list_cached_ml_apps()
        
        if cached_apps:
            st.markdown("### 💾 Cached ML Apps")
            
            # Create app info table
            app_data = []
            for app_info in cached_apps:
                app_data.append({
                    'Task': app_info.task,
                    'Model': app_info.model,
                    'Device': str(app_info.device),
                    'Usage Count': app_info.usage_count,
                    'Memory Usage (MB)': f"{app_info.memory_usage:.1f}",
                    'Success Rate': f"{app_info.success_rate:.1%}",
                    'Deployment Status': app_info.deployment_status,
                    'Last Used': time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(app_info.last_used))
                })
            
            df = pd.DataFrame(app_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No ML apps cached yet. Create some apps to see them cached here.")
    
    def render_advanced_options(self):
        """Render advanced options."""
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Optimization Settings:**")
                use_optimization = st.checkbox("Enable Auto-Optimization", value=True)
                enable_caching = st.checkbox("Enable App Caching", value=True)
                auto_device_selection = st.checkbox("Auto Device Selection", value=True)
                
            with col2:
                st.markdown("**Monitoring Settings:**")
                show_detailed_stats = st.checkbox("Show Detailed Statistics", value=True)
                monitor_memory = st.checkbox("Monitor Memory Usage", value=True)
                track_performance = st.checkbox("Track Performance Metrics", value=True)
            
            # Store options
            st.session_state.ml_apps_page_state['advanced_options'] = {
                'use_optimization': use_optimization,
                'enable_caching': enable_caching,
                'auto_device_selection': auto_device_selection,
                'show_detailed_stats': show_detailed_stats,
                'monitor_memory': monitor_memory,
                'track_performance': track_performance
            }
    
    def render_gradio_status(self):
        """Render Gradio availability status."""
        if not GRADIO_AVAILABLE:
            st.error("""
            ⚠️ **Gradio not available**
            
            To use ML Apps functionality, please install Gradio:
            ```bash
            pip install gradio
            ```
            """)
            return False
        else:
            st.success("✅ Gradio is available for ML app creation")
            return True
    
    def render(self):
        """Render the complete ML apps page."""
        self.render_header()
        
        # Check Gradio availability
        if not self.render_gradio_status():
            return
        
        # Task selection
        selected_task = self.render_task_selection()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Device configuration
        selected_device, batch_size, torch_dtype, load_in_8bit = self.render_device_configuration()
        
        # App configuration
        app_title, app_description, server_port, share_app, auto_open = self.render_app_configuration()
        
        # App controls
        if selected_task and selected_model:
            self.render_app_controls(
                selected_task, selected_model, selected_device,
                app_title, app_description, server_port, share_app, auto_open
            )
        
        # Advanced options
        self.render_advanced_options()
        
        # Active deployments
        self.render_active_deployments()
        
        # Performance stats
        self.render_performance_stats()
        
        # Cached apps
        self.render_cached_apps()


def render_ml_apps_page():
    """Render function for the ML apps page."""
    try:
        page = MLAppsPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering ML apps page: {str(e)}")
        error(f"Failed to render ML apps page", "ml_apps_page", e)


if __name__ == "__main__":
    # For testing
    render_ml_apps_page()
