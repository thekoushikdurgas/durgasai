"""
Auto Classes Management Page for DurgasAI.

This page provides a user interface for managing models using HuggingFace Auto Classes,
including model loading, configuration, and performance monitoring.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# try:
from utils.auto_classes_integration import DurgasAIAutoClassesIntegrator
from utils.model_manager import (
        EnhancedModelConfig, 
        TaskType, 
        ModelProvider,
        LoadingOptions,
        ModelCapabilities,
        ResourceRequirements
)
from utils.logger import info, debug, error, log_user_action
#     AUTO_CLASSES_AVAILABLE = True
# except ImportError as e:
#     AUTO_CLASSES_AVAILABLE = False
#     st.error(f"Auto Classes not available: {e}")
    
    # # Create dummy class for type hints when not available
    # class DurgasAIAutoClassesIntegrator:
    #     pass


def render_auto_classes_page():
    """Render the Auto Classes management page."""
    
    # if not AUTO_CLASSES_AVAILABLE:
    #     st.error("Auto Classes integration is not available. Please check dependencies.")
    #     return
    
    st.title("🤖 Auto Classes Model Management")
    st.markdown("Advanced model management using HuggingFace Auto Classes")
    
    # Initialize integrator
    if 'auto_classes_integrator' not in st.session_state:
        try:
            st.session_state.auto_classes_integrator = DurgasAIAutoClassesIntegrator(
                "config/model_config.json"
            )
            info("Auto Classes integrator initialized", "auto_classes_page")
        except Exception as e:
            st.error(f"Failed to initialize Auto Classes integrator: {e}")
            return
    
    integrator = st.session_state.auto_classes_integrator
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 System Status", 
        "🔧 Model Management", 
        "🧪 Model Testing", 
        "⚙️ Configuration", 
        "📈 Performance"
    ])
    
    with tab1:
        render_system_status(integrator)
    
    with tab2:
        render_model_management(integrator)
    
    with tab3:
        render_model_testing(integrator)
    
    with tab4:
        render_configuration_management(integrator)
    
    with tab5:
        render_performance_monitoring(integrator)


def render_system_status(integrator: DurgasAIAutoClassesIntegrator):
    """Render system status information."""
    st.header("System Status")
    
    # System validation
    validation = integrator.validate_system_compatibility()
    
    # Display compatibility status
    if validation['compatible']:
        st.success("✅ System is compatible with Auto Classes")
    else:
        st.error("❌ System compatibility issues detected")
    
    # Issues and recommendations
    if validation['issues']:
        st.subheader("Issues")
        for issue in validation['issues']:
            st.warning(f"⚠️ {issue}")
    
    if validation['recommendations']:
        st.subheader("Recommendations")
        for rec in validation['recommendations']:
            st.info(f"💡 {rec}")
    
    # System information
    st.subheader("System Information")
    system_info = integrator.adapter.enhanced_registry.get_system_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("CPU Cores", system_info['cpu_count'])
        st.metric("Total Memory", f"{system_info['memory_total_gb']:.1f} GB")
        st.metric("Available Memory", f"{system_info['memory_available_gb']:.1f} GB")
    
    with col2:
        st.metric("CUDA Available", "Yes" if system_info['cuda_available'] else "No")
        if system_info['cuda_available']:
            st.metric("CUDA Devices", system_info['cuda_device_count'])
        st.metric("Loaded Models", system_info['loaded_models_count'])
    
    # CUDA information
    if validation.get('cuda_info', {}).get('available'):
        st.subheader("CUDA Information")
        cuda_info = validation['cuda_info']
        st.json(cuda_info)


def render_model_management(integrator: DurgasAIAutoClassesIntegrator):
    """Render model management interface."""
    st.header("Model Management")
    
    # Available models
    models = integrator.get_available_models()
    
    if not models:
        st.warning("No models available. Please check configuration.")
        return
    
    # Model selection
    model_names = list(models.keys())
    selected_model = st.selectbox("Select Model", model_names)
    
    if selected_model:
        model_info = models[selected_model]
        
        # Display model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            st.write(f"**Name:** {selected_model}")
            st.write(f"**Model ID:** {model_info.get('model_id', 'N/A')}")
            st.write(f"**Task Type:** {model_info.get('task_type', 'N/A')}")
            st.write(f"**Provider:** {model_info.get('provider', 'N/A')}")
            st.write(f"**Loaded:** {'Yes' if model_info.get('loaded') else 'No'}")
        
        with col2:
            st.subheader("Capabilities")
            capabilities = integrator.get_model_capabilities(selected_model)
            if 'error' not in capabilities:
                for cap, value in capabilities.get('capabilities', {}).items():
                    st.write(f"**{cap.replace('_', ' ').title()}:** {'✅' if value else '❌'}")
        
        # Model actions
        st.subheader("Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Load Model", key=f"load_{selected_model}"):
                with st.spinner(f"Loading {selected_model}..."):
                    result = integrator.adapter.load_model_with_auto_classes(selected_model)
                    if result and result.success:
                        st.success(f"✅ Model loaded successfully!")
                        st.write(f"Auto Class: {result.auto_class_used}")
                        st.write(f"Loading Time: {result.loading_time:.2f}s")
                        st.write(f"Memory Usage: {result.memory_usage_mb:.1f}MB")
                        log_user_action("model_loaded", model=selected_model)
                    else:
                        st.error(f"❌ Failed to load model: {result.error if result else 'Unknown error'}")
        
        with col2:
            if st.button("Unload Model", key=f"unload_{selected_model}"):
                if integrator.adapter.enhanced_registry.unload_model(selected_model):
                    st.success("✅ Model unloaded successfully!")
                    log_user_action("model_unloaded", model=selected_model)
                else:
                    st.warning("Model was not loaded")
        
        with col3:
            if st.button("Refresh Info", key=f"refresh_{selected_model}"):
                st.rerun()
        
        # Detailed model information
        if model_info.get('loaded'):
            st.subheader("Detailed Information")
            detailed_info = integrator.adapter.get_model_info_enhanced(selected_model)
            
            # Performance metrics
            if 'loading_time' in detailed_info:
                st.metric("Loading Time", f"{detailed_info['loading_time']:.2f}s")
            if 'memory_usage_mb' in detailed_info:
                st.metric("Memory Usage", f"{detailed_info['memory_usage_mb']:.1f}MB")
            if 'parameters' in detailed_info:
                st.metric("Parameters", f"{detailed_info['parameters']:,}")


def render_model_testing(integrator: DurgasAIAutoClassesIntegrator):
    """Render model testing interface."""
    st.header("Model Testing")
    
    # Get loaded models
    loaded_models = integrator.adapter.enhanced_registry.list_loaded_models()
    
    if not loaded_models:
        st.warning("No models are currently loaded. Please load a model first.")
        return
    
    # Model selection for testing
    test_model = st.selectbox("Select Model to Test", loaded_models)
    
    if test_model:
        model_info = integrator.adapter.get_model_info_enhanced(test_model)
        task_type = model_info.get('task_type', 'unknown')
        
        st.subheader(f"Testing {test_model}")
        st.write(f"Task Type: {task_type}")
        
        # Input based on task type
        if task_type == "text-generation":
            render_text_generation_test(integrator, test_model)
        elif task_type == "seq2seq":
            render_seq2seq_test(integrator, test_model)
        elif task_type == "classification":
            render_classification_test(integrator, test_model)
        elif task_type == "image-text-to-text":
            render_multimodal_test(integrator, test_model)
        else:
            st.write("Testing interface not implemented for this task type yet.")


def render_text_generation_test(integrator: DurgasAIAutoClassesIntegrator, model_name: str):
    """Render text generation testing interface."""
    st.subheader("Text Generation Testing")
    
    # Input prompt
    prompt = st.text_area("Enter your prompt:", height=100, placeholder="Enter text to continue...")
    
    # Generation parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_tokens = st.slider("Max New Tokens", 10, 500, 100)
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    with col3:
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
    
    if st.button("Generate", key=f"generate_{model_name}"):
        if prompt.strip():
            with st.spinner("Generating response..."):
                try:
                    response = integrator.generate_response(
                        model_name,
                        prompt,
                        use_auto_classes=True,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    if response.success:
                        st.success("✅ Generation successful!")
                        st.subheader("Generated Response:")
                        st.write(response.content)
                        
                        # Display metadata
                        if response.metadata:
                            with st.expander("Generation Metadata"):
                                st.json(response.metadata)
                                
                        log_user_action("text_generation_test", 
                            model=model_name, 
                            prompt_length=len(prompt),
                            response_length=len(response.content))
                    else:
                        st.error(f"❌ Generation failed: {response.error}")
                        
                except Exception as e:
                    st.error(f"❌ Error during generation: {str(e)}")
        else:
            st.warning("Please enter a prompt")


def render_seq2seq_test(integrator: DurgasAIAutoClassesIntegrator, model_name: str):
    """Render seq2seq testing interface."""
    st.subheader("Sequence-to-Sequence Testing")
    
    # Common seq2seq tasks
    task_type = st.selectbox("Task Type", [
        "Translation", 
        "Summarization", 
        "Question Answering",
        "Custom"
    ])
    
    if task_type == "Translation":
        prompt = st.text_area("Text to translate:", placeholder="Enter text to translate...")
        target_lang = st.selectbox("Target Language", ["French", "Spanish", "German", "Italian"])
        full_prompt = f"Translate to {target_lang}: {prompt}"
    elif task_type == "Summarization":
        prompt = st.text_area("Text to summarize:", placeholder="Enter long text to summarize...")
        full_prompt = f"Summarize: {prompt}"
    elif task_type == "Question Answering":
        context = st.text_area("Context:", placeholder="Enter context...")
        question = st.text_input("Question:", placeholder="Enter question...")
        full_prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
    else:
        full_prompt = st.text_area("Custom prompt:", placeholder="Enter custom prompt...")
    
    if st.button("Process", key=f"seq2seq_{model_name}"):
        if full_prompt.strip():
            with st.spinner("Processing..."):
                try:
                    response = integrator.generate_response(
                        model_name,
                        full_prompt,
                        use_auto_classes=True,
                        max_new_tokens=200
                    )
                    
                    if response.success:
                        st.success("✅ Processing successful!")
                        st.subheader("Result:")
                        st.write(response.content)
                    else:
                        st.error(f"❌ Processing failed: {response.error}")
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
        else:
            st.warning("Please enter text to process")


def render_classification_test(integrator: DurgasAIAutoClassesIntegrator, model_name: str):
    """Render classification testing interface."""
    st.subheader("Text Classification Testing")
    
    # Input text
    text = st.text_area("Text to classify:", placeholder="Enter text for classification...")
    
    if st.button("Classify", key=f"classify_{model_name}"):
        if text.strip():
            with st.spinner("Classifying..."):
                try:
                    # For classification, we'll use the model directly
                    model_result = integrator.adapter.enhanced_registry.get_model(model_name)
                    
                    if model_result and model_result.success:
                        model = model_result.model
                        tokenizer = model_result.tokenizer
                        
                        # Tokenize and classify
                        inputs = tokenizer(text, return_tensors="pt", truncation=True)
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        
                        # Get top predictions
                        top_predictions = torch.topk(predictions, k=min(5, predictions.shape[-1]))
                        
                        st.success("✅ Classification successful!")
                        st.subheader("Predictions:")
                        
                        for i, (score, idx) in enumerate(zip(top_predictions.values[0], top_predictions.indices[0])):
                            st.write(f"{i+1}. Label {idx.item()}: {score.item():.3f}")
                    else:
                        st.error("Model not loaded")
                        
                except Exception as e:
                    st.error(f"❌ Error during classification: {str(e)}")
        else:
            st.warning("Please enter text to classify")


def render_multimodal_test(integrator: DurgasAIAutoClassesIntegrator, model_name: str):
    """Render multimodal testing interface."""
    st.subheader("Multimodal Testing")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=['png', 'jpg', 'jpeg'],
        key=f"image_{model_name}"
    )
    
    # Text prompt
    text_prompt = st.text_area(
        "Text prompt:", 
        placeholder="Describe what you want to know about the image..."
    )
    
    if st.button("Analyze", key=f"multimodal_{model_name}"):
        if uploaded_file and text_prompt.strip():
            with st.spinner("Analyzing image..."):
                try:
                    # Display uploaded image
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    
                    # Note: Full multimodal implementation would require
                    # proper image processing and model inference
                    st.info("🚧 Multimodal inference is partially implemented.")
                    st.write(f"Would analyze image with prompt: {text_prompt}")
                    
                    log_user_action("multimodal_test", model=model_name)
                    
                except Exception as e:
                    st.error(f"❌ Error during multimodal analysis: {str(e)}")
        else:
            st.warning("Please upload an image and enter a text prompt")


def render_configuration_management(integrator: DurgasAIAutoClassesIntegrator):
    """Render configuration management interface."""
    st.header("Configuration Management")
    
    # Current configuration display
    st.subheader("Current Models")
    models = integrator.get_available_models()
    
    for model_name, model_info in models.items():
        with st.expander(f"📋 {model_name}"):
            st.json(model_info)
    
    # Add new model configuration
    st.subheader("Add New Model")
    
    with st.form("add_model_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_model_id = st.text_input("Model ID", placeholder="e.g., microsoft/DialoGPT-small")
            new_display_name = st.text_input("Display Name", placeholder="e.g., DialoGPT Small")
            new_description = st.text_area("Description", placeholder="Model description...")
        
        with col2:
            new_task_type = st.selectbox("Task Type", [t.value for t in TaskType])
            new_provider = st.selectbox("Provider", [p.value for p in ModelProvider])
            new_max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=512)
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
                top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
            with col2:
                gpu_required = st.checkbox("GPU Required")
                min_memory = st.number_input("Min Memory (GB)", min_value=0.5, max_value=64.0, value=2.0)
        
        if st.form_submit_button("Add Model"):
            if new_model_id and new_display_name:
                try:
                    # Create enhanced configuration
                    new_config = EnhancedModelConfig(
                        model_id=new_model_id,
                        display_name=new_display_name,
                        provider=ModelProvider(new_provider),
                        task_type=TaskType(new_task_type),
                        description=new_description,
                        max_tokens=new_max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        capabilities=ModelCapabilities(
                            text_generation=True,
                            conversation=True
                        ),
                        resource_requirements=ResourceRequirements(
                            min_memory_gb=min_memory,
                            gpu_required=gpu_required
                        )
                    )
                    
                    # Register the model
                    integrator.adapter.enhanced_registry.register_model(
                        new_model_id.replace('/', '_'), 
                        new_config
                    )
                    
                    st.success(f"✅ Model {new_display_name} added successfully!")
                    log_user_action("model_added", model=new_model_id)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error adding model: {str(e)}")
            else:
                st.warning("Please fill in required fields")


def render_performance_monitoring(integrator: DurgasAIAutoClassesIntegrator):
    """Render performance monitoring interface."""
    st.header("Performance Monitoring")
    
    # System metrics
    system_info = integrator.adapter.enhanced_registry.get_system_info()
    
    # Memory usage chart
    st.subheader("Memory Usage")
    memory_data = {
        'Total': system_info['memory_total_gb'],
        'Used': system_info['memory_used_gb'],
        'Available': system_info['memory_available_gb']
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Memory", f"{memory_data['Total']:.1f} GB")
    with col2:
        st.metric("Used Memory", f"{memory_data['Used']:.1f} GB")
    with col3:
        st.metric("Available Memory", f"{memory_data['Available']:.1f} GB")
    
    # Model performance comparison
    st.subheader("Model Performance")
    
    loaded_models = integrator.adapter.enhanced_registry.list_loaded_models()
    if loaded_models:
        performance_data = []
        
        for model_name in loaded_models:
            info = integrator.adapter.get_model_info_enhanced(model_name)
            if info.get('loaded'):
                performance_data.append({
                    'Model': model_name,
                    'Parameters': info.get('parameters', 0),
                    'Loading Time (s)': info.get('loading_time', 0),
                    'Memory (MB)': info.get('memory_usage_mb', 0),
                    'Auto Class': info.get('auto_class', 'Unknown')
                })
        
        if performance_data:
            st.dataframe(performance_data)
        else:
            st.info("No performance data available")
    else:
        st.info("No models loaded for performance monitoring")
    
    # Migration status
    st.subheader("Migration Status")
    migration_plan = integrator.create_migration_plan()
    
    st.write(f"**Estimated Time:** {migration_plan['estimated_time']}")
    st.write(f"**Backup Required:** {'Yes' if migration_plan['backup_required'] else 'No'}")
    st.write(f"**Rollback Available:** {'Yes' if migration_plan['rollback_available'] else 'No'}")
    
    # Migration steps
    with st.expander("Migration Steps"):
        for step in migration_plan['steps']:
            st.write(f"**Step {step['step']}: {step['title']}**")
            st.write(f"Description: {step['description']}")
            st.write(f"Risk Level: {step['risk']}")
            if 'files' in step:
                st.write(f"Files: {', '.join(step['files'])}")
            st.write("---")


# Main execution
if __name__ == "__main__":
    render_auto_classes_page()
