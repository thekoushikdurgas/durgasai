"""
Processors Page for DurgasAI.

This module provides a comprehensive interface for processor management and testing:
- Processor selection and configuration
- Real-time multimodal processing testing
- Batch processing capabilities
- Performance monitoring and statistics
- Advanced multimodal features (text, image, audio coordination)
- Integration with enhanced processor manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.processor_manager import processor_manager, ProcessorInfo, MultimodalProcessingResult
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ProcessorsPage:
    """
    Comprehensive processor management and testing interface.
    
    This class provides:
    - Processor selection and loading
    - Real-time multimodal processing testing
    - Batch processing capabilities
    - Performance monitoring
    - Advanced configuration options
    - Integration with enhanced processor manager
    """
    
    def __init__(self):
        """Initialize the processors page."""
        debug("Initializing ProcessorsPage", "processors_page")
        self.config = Config()
        
        # Initialize session state for processors page
        if 'processors_page_state' not in st.session_state:
            st.session_state.processors_page_state = {
                'selected_model': 'google/paligemma-3b-pt-224',
                'input_modalities': [],
                'batch_mode': False,
                'show_advanced': False,
                'performance_stats': None,
                'processed_results': []
            }
        
        info("ProcessorsPage initialized successfully", "processors_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🔄 Processors Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📝 About Processors</h4>
        <p>Processors are unified preprocessing components that combine multiple modality-specific preprocessors 
        into a single interface for multimodal models. This page provides comprehensive tools for:</p>
        <ul>
        <li><strong>Processor Selection:</strong> Choose from various HuggingFace multimodal processors</li>
        <li><strong>Multimodal Processing:</strong> Process text, image, and audio inputs simultaneously</li>
        <li><strong>Batch Processing:</strong> Process multiple multimodal samples efficiently</li>
        <li><strong>Performance Monitoring:</strong> Track processor performance and caching</li>
        <li><strong>Advanced Features:</strong> Unified preprocessing coordination and optimization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_model_selection(self):
        """Render processor model selection interface."""
        st.markdown("### 🤖 Select Processor Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular processor models
            popular_models = [
                "google/paligemma-3b-pt-224",
                "llava-hf/llava-1.5-7b-hf",
                "openai/whisper-tiny",
                "facebook/blip-2-opt-2.7b",
                "microsoft/kosmos-2-patch14-224",
                "Salesforce/instructblip-vicuna-7b"
            ]
            
            selected_model = st.selectbox(
                "Choose a processor model:",
                options=popular_models,
                index=popular_models.index(st.session_state.processors_page_state['selected_model']) 
                if st.session_state.processors_page_state['selected_model'] in popular_models else 0,
                help="Select a HuggingFace processor model to work with"
            )
            
            # Custom model input
            custom_model = st.text_input(
                "Or enter custom model ID:",
                placeholder="e.g., google/paligemma-3b-pt-224",
                help="Enter any HuggingFace model ID for custom processor"
            )
            
            if custom_model:
                selected_model = custom_model
        
        with col2:
            # Model info display
            st.markdown("**Model Information:**")
            if selected_model:
                model_info = self.get_model_info(selected_model)
                if model_info:
                    st.write(f"**Type:** {model_info.get('processor_type', 'Unknown')}")
                    st.write(f"**Text:** {model_info.get('supports_text', False)}")
                    st.write(f"**Images:** {model_info.get('supports_images', False)}")
                    st.write(f"**Audio:** {model_info.get('supports_audio', False)}")
                    st.write(f"**Multimodal:** {model_info.get('supports_multimodal', False)}")
                else:
                    st.info("Click 'Load Processor' to get model information")
        
        # Update session state
        st.session_state.processors_page_state['selected_model'] = selected_model
        
        return selected_model
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get processor model information."""
        try:
            processor_info = processor_manager.get_processor_info(model_id)
            if processor_info:
                return processor_info
        except Exception as e:
            debug(f"Failed to get model info: {e}", "processors_page")
        
        return None
    
    def render_processor_controls(self, model_id: str):
        """Render processor loading and control buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Load Processor", type="primary"):
                with st.spinner(f"Loading processor for {model_id}..."):
                    start_time = time.time()
                    processor_info = processor_manager.load_processor(model_id)
                    load_time = time.time() - start_time
                    
                    if processor_info:
                        st.success(f"✅ Processor loaded in {load_time:.2f}s")
                        log_user_action("processor_loaded", model_id=model_id, load_time=load_time)
                    else:
                        st.error("❌ Failed to load processor")
                        log_user_action("processor_load_failed", model_id=model_id)
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = processor_manager.get_performance_stats()
                st.session_state.processors_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                processor_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("processor_cache_cleared")
        
        with col4:
            if st.button("📋 List Cached"):
                cached = processor_manager.list_cached_processors()
                st.session_state.processors_page_state['cached_processors'] = cached
    
    def render_input_modalities(self):
        """Render input modality selection interface."""
        st.markdown("### 🎯 Input Modalities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Text Input:**")
            text_input = st.text_area(
                "Enter text:",
                placeholder="Enter text to process...",
                height=100,
                help="Text input for the processor"
            )
        
        with col2:
            st.markdown("**Image Input:**")
            image_input_method = st.radio(
                "Image source:",
                ["Generate Sample Image", "Use Synthetic Data", "Upload Image (Coming Soon)"],
                key="image_input_method"
            )
            
            if image_input_method == "Generate Sample Image":
                image = self.generate_sample_image()
            elif image_input_method == "Use Synthetic Data":
                image = self.generate_synthetic_image()
            else:
                st.info("🖼️ Image upload functionality will be available in future updates")
                image = None
        
        with col3:
            st.markdown("**Audio Input:**")
            audio_input_method = st.radio(
                "Audio source:",
                ["Generate Sample Audio", "Use Synthetic Data", "Upload Audio (Coming Soon)"],
                key="audio_input_method"
            )
            
            if audio_input_method == "Generate Sample Audio":
                audio = self.generate_sample_audio()
            elif audio_input_method == "Use Synthetic Data":
                audio = self.generate_synthetic_audio()
            else:
                st.info("🎵 Audio upload functionality will be available in future updates")
                audio = None
        
        # Store inputs in session state
        inputs = {}
        if text_input.strip():
            inputs['text'] = text_input
        if image is not None:
            inputs['images'] = image
        if audio is not None:
            inputs['audio'] = audio
        
        st.session_state.processors_page_state['input_modalities'] = list(inputs.keys())
        st.session_state.processors_page_state['inputs'] = inputs
        
        return inputs
    
    def generate_sample_image(self):
        """Generate sample image for testing."""
        if PIL_AVAILABLE:
            # Create a simple colored image
            image = Image.new('RGB', (224, 224), (255, 0, 0))  # Red image
            return image
        return None
    
    def generate_synthetic_image(self):
        """Generate synthetic image data for testing."""
        if NUMPY_AVAILABLE and PIL_AVAILABLE:
            # Create synthetic image data
            data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(data)
            return image
        return None
    
    def generate_sample_audio(self):
        """Generate sample audio for testing."""
        if NUMPY_AVAILABLE:
            # Generate simple sine wave
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            frequency = 440  # A4 note
            audio = np.sin(frequency * 2 * np.pi * t).astype(np.float32)
            return audio
        return None
    
    def generate_synthetic_audio(self):
        """Generate synthetic audio data for testing."""
        if NUMPY_AVAILABLE:
            # Generate white noise
            sample_rate = 16000
            duration = 1.0
            audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
            return audio
        return None
    
    def render_advanced_options(self):
        """Render advanced processing options."""
        with st.expander("⚙️ Advanced Processing Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                return_tensors = st.selectbox("Return Tensors", ["None", "pt", "tf", "np"], index=1, help="Tensor format for output")
                padding = st.checkbox("Padding", value=True, help="Pad sequences to uniform length")
                truncation = st.checkbox("Truncation", value=False, help="Truncate sequences to maximum length")
                
            with col2:
                max_length = st.number_input("Max Length", min_value=100, max_value=10000, value=512, help="Maximum sequence length")
                return_attention_mask = st.checkbox("Return Attention Mask", value=True, help="Include attention mask in output")
                
            # Store options
            st.session_state.processors_page_state['advanced_options'] = {
                'return_tensors': return_tensors if return_tensors != "None" else None,
                'padding': padding,
                'truncation': truncation,
                'max_length': max_length,
                'return_attention_mask': return_attention_mask
            }
    
    def render_processing_interface(self, model_id: str, inputs: Dict[str, Any]):
        """Render multimodal processing interface."""
        st.markdown("### 🔄 Multimodal Processing")
        
        # Advanced options
        self.render_advanced_options()
        
        # Display input summary
        if inputs:
            st.markdown("**📋 Input Summary:**")
            for modality, value in inputs.items():
                if modality == 'text':
                    st.write(f"**Text:** {value[:100]}{'...' if len(value) > 100 else ''}")
                elif modality == 'images':
                    st.write(f"**Images:** {type(value).__name__} - {value.size if hasattr(value, 'size') else 'Unknown size'}")
                elif modality == 'audio':
                    st.write(f"**Audio:** {type(value).__name__} - {len(value) if hasattr(value, '__len__') else 'Unknown length'} samples")
        
        # Process button
        if st.button("🚀 Process Multimodal", type="primary"):
            self.process_multimodal(model_id, inputs)
    
    def process_multimodal(self, model_id: str, inputs: Dict[str, Any]):
        """Process multimodal inputs with the selected options."""
        if not inputs:
            st.error("No inputs provided for processing")
            return
        
        try:
            # Get options
            options = st.session_state.processors_page_state.get('advanced_options', {})
            
            # Process multimodal inputs
            with st.spinner("Processing multimodal inputs..."):
                start_time = time.time()
                result = processor_manager.process_multimodal(
                    model_id, **inputs, **options
                )
                processing_time = time.time() - start_time
            
            if result and result.success:
                self.display_results(result, processing_time)
                log_user_action("multimodal_processed", model_id=model_id, modalities=result.modalities_processed)
            else:
                st.error(f"Multimodal processing failed: {result.error if result else 'Unknown error'}")
                
        except Exception as e:
            st.error(f"Multimodal processing failed: {str(e)}")
            error(f"Multimodal processing failed", "processors_page", e)
    
    def display_results(self, result: MultimodalProcessingResult, processing_time: float):
        """Display multimodal processing results."""
        st.markdown("### 📊 Multimodal Processing Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Batch Size", result.batch_size)
            st.metric("Modalities", len(result.modalities_processed))
        
        with col2:
            if result.batch_size > 0:
                avg_time = processing_time / result.batch_size
                st.metric("Avg Time per Sample", f"{avg_time:.3f}s")
                throughput = result.batch_size / processing_time
                st.metric("Throughput", f"{throughput:.1f} samples/sec")
            
            if result.processor_info:
                st.metric("Processor Type", result.processor_info.processor_type)
        
        with col3:
            st.metric("Input IDs", "✅" if result.input_ids is not None else "❌")
            st.metric("Pixel Values", "✅" if result.pixel_values is not None else "❌")
            st.metric("Input Features", "✅" if result.input_features is not None else "❌")
            st.metric("Attention Mask", "✅" if result.attention_mask is not None else "❌")
        
        # Detailed results
        if result.success:
            st.markdown("**📝 Detailed Results:**")
            
            # Processor info
            if result.processor_info:
                with st.expander("🔧 Processor Information"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Model:** {result.processor_info.model_id}")
                        st.write(f"**Type:** {result.processor_info.processor_type}")
                        st.write(f"**Supported Modalities:** {', '.join(result.processor_info.supported_modalities)}")
                        st.write(f"**Tokenizer Class:** {result.processor_info.tokenizer_class}")
                    
                    with col2:
                        st.write(f"**Image Processor Class:** {result.processor_info.image_processor_class}")
                        st.write(f"**Feature Extractor Class:** {result.processor_info.feature_extractor_class}")
                        st.write(f"**Model Input Names:** {', '.join(result.processor_info.model_input_names)}")
                        st.write(f"**Supports Batch:** {result.processor_info.supports_batch}")
            
            # Processing details
            with st.expander("🔄 Processing Details"):
                st.write(f"**Modalities Processed:** {', '.join(result.modalities_processed)}")
                st.write(f"**Batch Size:** {result.batch_size}")
                st.write(f"**Processing Time:** {result.processing_time:.3f}s")
                
                if result.metadata:
                    st.write(f"**Result Keys:** {', '.join(result.metadata.get('result_keys', []))}")
                    st.write(f"**Return Tensors:** {result.metadata.get('return_tensors', 'None')}")
            
            # Output details
            with st.expander("📤 Output Details"):
                if result.input_ids is not None:
                    st.write("**Input IDs:**")
                    if hasattr(result.input_ids, 'shape'):
                        st.write(f"   Shape: {result.input_ids.shape}")
                    if hasattr(result.input_ids, 'dtype'):
                        st.write(f"   Data Type: {result.input_ids.dtype}")
                
                if result.pixel_values is not None:
                    st.write("**Pixel Values:**")
                    if hasattr(result.pixel_values, 'shape'):
                        st.write(f"   Shape: {result.pixel_values.shape}")
                    if hasattr(result.pixel_values, 'dtype'):
                        st.write(f"   Data Type: {result.pixel_values.dtype}")
                
                if result.input_features is not None:
                    st.write("**Input Features:**")
                    if hasattr(result.input_features, 'shape'):
                        st.write(f"   Shape: {result.input_features.shape}")
                    if hasattr(result.input_features, 'dtype'):
                        st.write(f"   Data Type: {result.input_features.dtype}")
                
                if result.attention_mask is not None:
                    st.write("**Attention Mask:**")
                    if hasattr(result.attention_mask, 'shape'):
                        st.write(f"   Shape: {result.attention_mask.shape}")
                    if hasattr(result.attention_mask, 'dtype'):
                        st.write(f"   Data Type: {result.attention_mask.dtype}")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.processors_page_state.get('performance_stats')
        
        if stats:
            st.markdown("### 📊 Performance Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1f}%")
                st.metric("Total Loads", stats['total_loads'])
                st.metric("Cache Hits", stats['cache_hits'])
            
            with col2:
                st.metric("Cache Misses", stats['cache_misses'])
                st.metric("Avg Load Time", f"{stats['average_load_time']:.3f}s")
                st.metric("Memory Cache Size", stats['memory_cache_size'])
            
            with col3:
                st.metric("Total Processings", stats['total_processings'])
                st.metric("Multimodal Processings", stats['multimodal_processings'])
                st.metric("Batch Processings", stats['batch_processings'])
                st.metric("Total Inputs", stats['total_inputs'])
    
    def render_cached_processors(self):
        """Render cached processors information."""
        cached = st.session_state.processors_page_state.get('cached_processors', [])
        
        if cached:
            st.markdown("### 💾 Cached Processors")
            
            for processor_info in cached:
                with st.expander(f"🔄 {processor_info['model_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {processor_info['processor_type']}")
                        st.write(f"**Supported Modalities:** {', '.join(processor_info['supported_modalities'])}")
                        st.write(f"**Supports Text:** {processor_info.get('supports_text', False)}")
                    
                    with col2:
                        st.write(f"**Load Time:** {processor_info['load_time']:.3f}s")
                        st.write(f"**Supports Images:** {processor_info.get('supports_images', False)}")
                        st.write(f"**Cache Key:** {processor_info['cache_key'][:16]}...")
    
    def render(self):
        """Render the complete processors page."""
        self.render_header()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Processor controls
        self.render_processor_controls(selected_model)
        
        # Input modalities
        inputs = self.render_input_modalities()
        
        # Processing interface
        if inputs:
            self.render_processing_interface(selected_model, inputs)
        
        # Performance stats
        if st.session_state.processors_page_state.get('performance_stats'):
            self.render_performance_stats()
        
        # Cached processors
        if st.session_state.processors_page_state.get('cached_processors'):
            self.render_cached_processors()


def render_processors_page():
    """Render function for the processors page."""
    try:
        page = ProcessorsPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering processors page: {str(e)}")
        error(f"Failed to render processors page", "processors_page", e)


if __name__ == "__main__":
    # For testing
    render_processors_page()
