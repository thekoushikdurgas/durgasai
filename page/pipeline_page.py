"""
Pipeline Page for DurgasAI.

This module provides a comprehensive interface for pipeline management:
- Task selection and configuration
- Model selection and optimization
- Real-time pipeline testing
- Performance monitoring and optimization
- Memory usage analysis
- Batch processing optimization
- Integration with enhanced pipeline manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json
import pandas as pd

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline_manager import (
    pipeline_manager, 
    PipelineConfig, 
    PipelineResult,
    PipelineInfo
)
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PipelinePage:
    """
    Comprehensive pipeline management interface.
    
    This class provides:
    - Task selection and configuration
    - Model selection and optimization
    - Real-time pipeline testing
    - Performance monitoring and optimization
    - Memory usage analysis
    - Batch processing optimization
    - Integration with enhanced pipeline manager
    """
    
    def __init__(self):
        """Initialize the pipeline page."""
        debug("Initializing PipelinePage", "pipeline_page")
        self.config = Config()
        
        # Initialize session state for pipeline page
        if 'pipeline_page_state' not in st.session_state:
            st.session_state.pipeline_page_state = {
                'selected_task': 'text-generation',
                'selected_model': 'google/gemma-2-2b',
                'selected_device': -1,
                'batch_size': 1,
                'torch_dtype': None,
                'load_in_8bit': False,
                'test_inputs': ["The secret to baking a really good cake is "],
                'show_advanced': False,
                'performance_stats': None,
                'test_results': {},
                'pipeline_info': None
            }
        
        info("PipelinePage initialized successfully", "pipeline_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🔄 Pipeline Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>🔄 About Hugging Face Pipelines</h4>
        <p>Hugging Face Pipelines are simple but powerful inference APIs that provide ready-to-use solutions for a variety of machine learning tasks. This page provides advanced tools for:</p>
        <ul>
        <li><strong>Task Selection:</strong> Choose from text generation, classification, QA, vision, and audio tasks</li>
        <li><strong>Model Configuration:</strong> Optimize models for specific tasks and hardware</li>
        <li><strong>Real-time Testing:</strong> Test pipelines with your own inputs</li>
        <li><strong>Performance Monitoring:</strong> Monitor execution times, memory usage, and throughput</li>
        <li><strong>Hardware Optimization:</strong> GPU acceleration, quantization, and device management</li>
        <li><strong>Batch Processing:</strong> Optimize batch processing for maximum efficiency</li>
        <li><strong>Pipeline Caching:</strong> Intelligent caching for improved performance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_task_selection(self):
        """Render task selection interface."""
        st.markdown("### 🎯 Select Task")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Available tasks
            available_tasks = pipeline_manager.get_available_tasks()
            
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
                index=available_tasks.index(st.session_state.pipeline_page_state['selected_task']) 
                if st.session_state.pipeline_page_state['selected_task'] in available_tasks else 0,
                help="Select a machine learning task for the pipeline"
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
        st.session_state.pipeline_page_state['selected_task'] = selected_task
        
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
            
            selected_task = st.session_state.pipeline_page_state['selected_task']
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
                    if TRANSFORMERS_AVAILABLE:
                        # Try to get model info (this might be slow)
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
                    else:
                        st.info("Transformers not available")
                except Exception as e:
                    st.info(f"Model info unavailable: {e}")
        
        # Update session state
        st.session_state.pipeline_page_state['selected_model'] = selected_model
        
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
                index=list(device_options.values()).index(st.session_state.pipeline_page_state['selected_device']) 
                if st.session_state.pipeline_page_state['selected_device'] in device_options.values() else 0,
                help="Select hardware for pipeline execution"
            )
            
            selected_device = device_options[selected_device_label]
        
        with col2:
            st.markdown("**Memory Optimization:**")
            
            # Batch size
            batch_size = st.number_input(
                "Batch Size:",
                min_value=1,
                max_value=32,
                value=st.session_state.pipeline_page_state['batch_size'],
                help="Number of inputs to process simultaneously"
            )
            
            # Precision
            torch_dtype = st.selectbox(
                "Precision:",
                options=["None", "float16", "bfloat16", "float32"],
                index=["None", "float16", "bfloat16", "float32"].index(
                    st.session_state.pipeline_page_state['torch_dtype'] or "None"
                ),
                help="Floating point precision (lower = less memory)"
            )
            
            torch_dtype = torch_dtype if torch_dtype != "None" else None
            
            # Quantization
            load_in_8bit = st.checkbox(
                "8-bit Quantization",
                value=st.session_state.pipeline_page_state['load_in_8bit'],
                help="Reduce memory usage with 8-bit quantization"
            )
        
        # Update session state
        st.session_state.pipeline_page_state.update({
            'selected_device': selected_device,
            'batch_size': batch_size,
            'torch_dtype': torch_dtype,
            'load_in_8bit': load_in_8bit
        })
        
        return selected_device, batch_size, torch_dtype, load_in_8bit
    
    def render_input_interface(self):
        """Render input interface."""
        st.markdown("### 📝 Test Inputs")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Use Sample Inputs", "Enter Custom Inputs"],
                horizontal=True
            )
            
            if input_method == "Use Sample Inputs":
                # Sample inputs by task
                sample_inputs = {
                    "text-generation": [
                        "The secret to baking a really good cake is ",
                        "In the future, artificial intelligence will ",
                        "Once upon a time, there was a "
                    ],
                    "text-classification": [
                        "I love this new AI technology!",
                        "This product is terrible and overpriced.",
                        "The weather is nice today."
                    ],
                    "question-answering": {
                        "context": "Hugging Face is a company that provides natural language processing tools and models. They offer a wide range of pre-trained models and datasets for various AI tasks.",
                        "questions": [
                            "What does Hugging Face provide?",
                            "What kind of models do they offer?",
                            "What is the company called?"
                        ]
                    },
                    "image-classification": [
                        "Upload an image file to test image classification"
                    ]
                }
                
                selected_task = st.session_state.pipeline_page_state['selected_task']
                if selected_task in sample_inputs:
                    if selected_task == "question-answering":
                        st.text_area("Context:", value=sample_inputs[selected_task]["context"], height=100)
                        test_inputs = sample_inputs[selected_task]["questions"]
                    else:
                        test_inputs = sample_inputs[selected_task]
                else:
                    test_inputs = ["Sample input for testing"]
                
            else:
                # Custom inputs
                custom_inputs = st.text_area(
                    "Enter your inputs (one per line):",
                    value="\n".join(st.session_state.pipeline_page_state.get('test_inputs', [])),
                    height=150,
                    help="Enter multiple inputs, one per line"
                )
                test_inputs = [input.strip() for input in custom_inputs.split('\n') if input.strip()]
        
        with col2:
            # Input statistics
            if test_inputs:
                st.markdown("**Input Statistics:**")
                st.write(f"**Number of inputs:** {len(test_inputs)}")
                
                # Calculate input lengths
                input_lengths = [len(str(input).split()) if isinstance(input, str) else 0 for input in test_inputs]
                if input_lengths:
                    st.write(f"**Min words:** {min(input_lengths)}")
                    st.write(f"**Max words:** {max(input_lengths)}")
                    st.write(f"**Avg words:** {sum(input_lengths) / len(input_lengths):.1f}")
                
                # Show preview
                st.markdown("**Preview:**")
                for i, input_item in enumerate(test_inputs[:3]):
                    preview = str(input_item)[:50] + "..." if len(str(input_item)) > 50 else str(input_item)
                    st.write(f"{i+1}. {preview}")
                if len(test_inputs) > 3:
                    st.write(f"... and {len(test_inputs) - 3} more")
        
        # Update session state
        st.session_state.pipeline_page_state['test_inputs'] = test_inputs
        
        return test_inputs
    
    def render_test_controls(self, selected_task: str, selected_model: str, 
                           selected_device: int, test_inputs: List[str]):
        """Render test control buttons."""
        st.markdown("### 🧪 Test Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Test Pipeline", type="primary"):
                if selected_task and selected_model and test_inputs:
                    self.run_pipeline_test(selected_task, selected_model, selected_device, test_inputs)
                else:
                    st.error("Please select task, model, and enter inputs for testing")
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = pipeline_manager.get_performance_stats()
                st.session_state.pipeline_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                pipeline_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("pipeline_cache_cleared")
        
        with col4:
            if st.button("🔄 Refresh"):
                st.rerun()
    
    def run_pipeline_test(self, selected_task: str, selected_model: str, 
                         selected_device: int, test_inputs: List[str]):
        """Run pipeline test."""
        with st.spinner("Testing pipeline..."):
            try:
                # Create pipeline configuration
                config = PipelineConfig(
                    task=selected_task,
                    model=selected_model,
                    device=selected_device,
                    batch_size=st.session_state.pipeline_page_state['batch_size'],
                    torch_dtype=st.session_state.pipeline_page_state['torch_dtype'],
                    load_in_8bit=st.session_state.pipeline_page_state['load_in_8bit']
                )
                
                # Handle special cases for QA task
                if selected_task == "question-answering" and len(test_inputs) > 0:
                    # For QA, we need context and questions
                    context = st.session_state.pipeline_page_state.get('context', 
                        "Hugging Face is a company that provides natural language processing tools and models.")
                    
                    # Execute pipeline for each question
                    results = []
                    for question in test_inputs:
                        qa_input = {"question": question, "context": context}
                        result = pipeline_manager.execute_pipeline(config, qa_input)
                        if result:
                            results.append(result)
                else:
                    # Execute pipeline with inputs
                    result = pipeline_manager.execute_pipeline(config, test_inputs)
                    results = [result] if result else []
                
                if results and all(r.success for r in results):
                    st.session_state.pipeline_page_state['test_results'] = {
                        'config': config,
                        'results': results,
                        'inputs': test_inputs
                    }
                    st.success(f"✅ Pipeline test completed successfully")
                    log_user_action("pipeline_test_completed", 
                                  task=selected_task, 
                                  model=selected_model,
                                  num_inputs=len(test_inputs))
                else:
                    error_msg = results[0].error if results else "Unknown error"
                    st.error(f"❌ Pipeline test failed: {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ Pipeline test failed: {str(e)}")
                error(f"Pipeline test failed", "pipeline_page", e)
    
    def render_test_results(self):
        """Render test results."""
        test_results = st.session_state.pipeline_page_state.get('test_results', {})
        
        if not test_results:
            return
        
        results = test_results.get('results', [])
        config = test_results.get('config')
        inputs = test_results.get('inputs', [])
        
        if not results or not config:
            return
        
        st.markdown("### 📊 Test Results")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_processing_time = sum(r.processing_time for r in results)
        total_inputs = sum(r.input_count for r in results)
        
        with col1:
            st.metric("Total Processing Time", f"{total_processing_time:.3f}s")
            st.metric("Total Inputs", total_inputs)
        
        with col2:
            st.metric("Avg Time per Input", f"{total_processing_time/max(total_inputs, 1):.3f}s")
            st.metric("Success Rate", "100%" if all(r.success for r in results) else "Partial")
        
        with col3:
            if results[0].metadata:
                st.metric("Memory Usage", f"{results[0].metadata.get('memory_usage_mb', 0):.1f} MB")
                st.metric("Device Used", results[0].device_used or "Unknown")
        
        with col4:
            st.metric("Batch Size", config.batch_size or 1)
            st.metric("Model", config.model or "Default")
        
        # Detailed results
        if results and results[0].success:
            with st.expander("🔍 Detailed Results"):
                for i, (input_item, result) in enumerate(zip(inputs, results)):
                    st.markdown(f"**Input {i+1}:** {str(input_item)[:100]}...")
                    
                    if result.outputs:
                        if isinstance(result.outputs, list) and len(result.outputs) > 0:
                            output = result.outputs[0]
                            if isinstance(output, dict):
                                for key, value in output.items():
                                    if key == 'generated_text':
                                        st.write(f"**Generated:** {value}")
                                    elif key == 'label':
                                        st.write(f"**Label:** {value}")
                                    elif key == 'score':
                                        st.write(f"**Score:** {value:.3f}")
                                    elif key == 'answer':
                                        st.write(f"**Answer:** {value}")
                                    else:
                                        st.write(f"**{key}:** {value}")
                            else:
                                st.write(f"**Output:** {str(output)[:200]}...")
                        else:
                            st.write(f"**Output:** {str(result.outputs)[:200]}...")
                    
                    st.write(f"**Processing Time:** {result.processing_time:.3f}s")
                    st.write("---")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.pipeline_page_state.get('performance_stats')
        
        if stats:
            st.markdown("### 📊 Performance Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Executions", stats['total_executions'])
                st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1f}%")
                st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            
            with col2:
                st.metric("Avg Execution Time", f"{stats['average_execution_time']:.3f}s")
                st.metric("Total Inputs Processed", f"{stats['total_inputs_processed']:,}")
                st.metric("Memory Cache Size", stats['memory_cache_size'])
            
            with col3:
                st.metric("Device Switches", stats['device_switches'])
                st.metric("Memory Optimizations", stats['memory_optimizations'])
                st.metric("Batch Optimizations", stats['batch_optimizations'])
                st.metric("Quantization Usage", stats['quantization_usage'])
    
    def render_cached_pipelines(self):
        """Render cached pipelines information."""
        cached_pipelines = pipeline_manager.list_cached_pipelines()
        
        if cached_pipelines:
            st.markdown("### 💾 Cached Pipelines")
            
            # Create pipeline info table
            pipeline_data = []
            for pipeline_info in cached_pipelines:
                pipeline_data.append({
                    'Task': pipeline_info.task,
                    'Model': pipeline_info.model,
                    'Device': str(pipeline_info.device),
                    'Usage Count': pipeline_info.usage_count,
                    'Memory Usage (MB)': f"{pipeline_info.memory_usage:.1f}",
                    'Success Rate': f"{pipeline_info.success_rate:.1%}",
                    'Last Used': time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(pipeline_info.last_used))
                })
            
            df = pd.DataFrame(pipeline_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No pipelines cached yet. Run some tests to see cached pipelines here.")
    
    def render_advanced_options(self):
        """Render advanced options."""
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Optimization Settings:**")
                use_optimization = st.checkbox("Enable Auto-Optimization", value=True)
                enable_caching = st.checkbox("Enable Pipeline Caching", value=True)
                auto_device_selection = st.checkbox("Auto Device Selection", value=True)
                
            with col2:
                st.markdown("**Monitoring Settings:**")
                show_detailed_stats = st.checkbox("Show Detailed Statistics", value=True)
                monitor_memory = st.checkbox("Monitor Memory Usage", value=True)
                track_performance = st.checkbox("Track Performance Metrics", value=True)
            
            # Store options
            st.session_state.pipeline_page_state['advanced_options'] = {
                'use_optimization': use_optimization,
                'enable_caching': enable_caching,
                'auto_device_selection': auto_device_selection,
                'show_detailed_stats': show_detailed_stats,
                'monitor_memory': monitor_memory,
                'track_performance': track_performance
            }
    
    def render(self):
        """Render the complete pipeline page."""
        self.render_header()
        
        # Task selection
        selected_task = self.render_task_selection()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Device configuration
        selected_device, batch_size, torch_dtype, load_in_8bit = self.render_device_configuration()
        
        # Input interface
        test_inputs = self.render_input_interface()
        
        # Test controls
        if selected_task and selected_model and test_inputs:
            self.render_test_controls(selected_task, selected_model, selected_device, test_inputs)
        
        # Advanced options
        self.render_advanced_options()
        
        # Test results
        self.render_test_results()
        
        # Performance stats
        self.render_performance_stats()
        
        # Cached pipelines
        self.render_cached_pipelines()


def render_pipeline_page():
    """Render function for the pipeline page."""
    try:
        page = PipelinePage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering pipeline page: {str(e)}")
        error(f"Failed to render pipeline page", "pipeline_page", e)


if __name__ == "__main__":
    # For testing
    render_pipeline_page()
