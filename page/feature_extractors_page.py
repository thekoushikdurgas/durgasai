"""
Feature Extractors Page for DurgasAI.

This module provides a comprehensive interface for feature extractor management and testing:
- Feature extractor selection and configuration
- Real-time audio processing testing
- Batch processing capabilities
- Performance monitoring and statistics
- Advanced audio features (resampling, padding, truncation)
- Integration with enhanced feature extractor manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.feature_extractor_manager import feature_extractor_manager, FeatureExtractorInfo, AudioProcessingResult
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from transformers import AutoFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class FeatureExtractorsPage:
    """
    Comprehensive feature extractor management and testing interface.
    
    This class provides:
    - Feature extractor selection and loading
    - Real-time audio processing testing
    - Batch processing capabilities
    - Performance monitoring
    - Advanced configuration options
    - Integration with enhanced feature extractor manager
    """
    
    def __init__(self):
        """Initialize the feature extractors page."""
        debug("Initializing FeatureExtractorsPage", "feature_extractors_page")
        self.config = Config()
        
        # Initialize session state for feature extractors page
        if 'feature_extractors_page_state' not in st.session_state:
            st.session_state.feature_extractors_page_state = {
                'selected_model': 'facebook/wav2vec2-base',
                'uploaded_audio': [],
                'batch_mode': False,
                'show_advanced': False,
                'performance_stats': None,
                'processed_audio': []
            }
        
        info("FeatureExtractorsPage initialized successfully", "feature_extractors_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🎵 Feature Extractors Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📝 About Feature Extractors</h4>
        <p>Feature extractors preprocess audio data into the correct format for audio models. 
        This page provides comprehensive tools for:</p>
        <ul>
        <li><strong>Feature Extractor Selection:</strong> Choose from various HuggingFace audio feature extractors</li>
        <li><strong>Audio Processing:</strong> Process raw audio signals into model-ready tensors</li>
        <li><strong>Batch Processing:</strong> Process multiple audio samples efficiently</li>
        <li><strong>Performance Monitoring:</strong> Track feature extractor performance and caching</li>
        <li><strong>Advanced Features:</strong> Resampling, padding, truncation, and more</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_model_selection(self):
        """Render feature extractor model selection interface."""
        st.markdown("### 🤖 Select Feature Extractor Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular feature extractor models
            popular_models = [
                "facebook/wav2vec2-base",
                "facebook/wav2vec2-large",
                "openai/whisper-tiny",
                "openai/whisper-base",
                "facebook/hubert-base-ls960",
                "microsoft/wavlm-base"
            ]
            
            selected_model = st.selectbox(
                "Choose a feature extractor model:",
                options=popular_models,
                index=popular_models.index(st.session_state.feature_extractors_page_state['selected_model']) 
                if st.session_state.feature_extractors_page_state['selected_model'] in popular_models else 0,
                help="Select a HuggingFace feature extractor model to work with"
            )
            
            # Custom model input
            custom_model = st.text_input(
                "Or enter custom model ID:",
                placeholder="e.g., facebook/wav2vec2-base",
                help="Enter any HuggingFace model ID for custom feature extractor"
            )
            
            if custom_model:
                selected_model = custom_model
        
        with col2:
            # Model info display
            st.markdown("**Model Information:**")
            if selected_model:
                model_info = self.get_model_info(selected_model)
                if model_info:
                    st.write(f"**Type:** {model_info.get('extractor_type', 'Unknown')}")
                    st.write(f"**Sampling Rate:** {model_info.get('sampling_rate', 'Unknown')}Hz")
                    st.write(f"**Supports Padding:** {model_info.get('supports_padding', False)}")
                    st.write(f"**Supports Truncation:** {model_info.get('supports_truncation', False)}")
                else:
                    st.info("Click 'Load Extractor' to get model information")
        
        # Update session state
        st.session_state.feature_extractors_page_state['selected_model'] = selected_model
        
        return selected_model
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get feature extractor model information."""
        try:
            extractor_info = feature_extractor_manager.get_extractor_info(model_id)
            if extractor_info:
                return extractor_info
        except Exception as e:
            debug(f"Failed to get model info: {e}", "feature_extractors_page")
        
        return None
    
    def render_processor_controls(self, model_id: str):
        """Render feature extractor loading and control buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Load Extractor", type="primary"):
                with st.spinner(f"Loading feature extractor for {model_id}..."):
                    start_time = time.time()
                    extractor_info = feature_extractor_manager.load_extractor(model_id)
                    load_time = time.time() - start_time
                    
                    if extractor_info:
                        st.success(f"✅ Feature extractor loaded in {load_time:.2f}s")
                        log_user_action("feature_extractor_loaded", model_id=model_id, load_time=load_time)
                    else:
                        st.error("❌ Failed to load feature extractor")
                        log_user_action("feature_extractor_load_failed", model_id=model_id)
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = feature_extractor_manager.get_performance_stats()
                st.session_state.feature_extractors_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                feature_extractor_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("feature_extractor_cache_cleared")
        
        with col4:
            if st.button("📋 List Cached"):
                cached = feature_extractor_manager.list_cached_extractors()
                st.session_state.feature_extractors_page_state['cached_extractors'] = cached
    
    def render_audio_input(self):
        """Render audio input interface."""
        st.markdown("### 🎧 Audio Input")
        
        # Mode selection
        col1, col2 = st.columns([1, 3])
        with col1:
            batch_mode = st.checkbox(
                "Batch Mode", 
                value=st.session_state.feature_extractors_page_state['batch_mode'],
                help="Process multiple audio samples at once for better performance"
            )
            st.session_state.feature_extractors_page_state['batch_mode'] = batch_mode
        
        with col2:
            if batch_mode:
                st.info("💡 Batch mode processes multiple audio samples efficiently using optimized processing")
        
        # Audio input method
        input_method = st.radio(
            "Choose input method:",
            ["Generate Sample Audio", "Use Synthetic Data", "Upload Audio (Coming Soon)"],
            horizontal=True
        )
        
        if input_method == "Generate Sample Audio":
            self.render_sample_generation()
        elif input_method == "Use Synthetic Data":
            self.render_synthetic_data()
        elif input_method == "Upload Audio (Coming Soon)":
            st.info("🎧 Audio upload functionality will be available in future updates")
    
    def render_sample_generation(self):
        """Render sample audio generation interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            num_audio = st.number_input("Number of audio samples:", min_value=1, max_value=8, value=3)
            duration = st.number_input("Duration (seconds):", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        with col2:
            sample_rate = st.selectbox("Sample rate:", [8000, 16000, 22050, 44100, 48000], index=1)
            frequency = st.number_input("Frequency (Hz):", min_value=100, max_value=2000, value=440)
        
        if st.button("🎵 Generate Sample Audio"):
            if NUMPY_AVAILABLE:
                audio_samples = self.generate_sample_audio(num_audio, duration, sample_rate, frequency)
                st.session_state.feature_extractors_page_state['uploaded_audio'] = audio_samples
                
                # Display generated audio info
                st.markdown("**Generated Sample Audio:**")
                for i, audio in enumerate(audio_samples):
                    st.write(f"Audio {i+1}: Length {len(audio)}, Sample Rate: {sample_rate}Hz, Frequency: {frequency}Hz")
            else:
                st.error("NumPy not available for audio generation")
    
    def generate_sample_audio(self, num_audio: int, duration: float, sample_rate: int, frequency: int) -> List[np.ndarray]:
        """Generate sample audio data for testing."""
        audio_samples = []
        
        for i in range(num_audio):
            # Generate time array
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Generate different types of audio signals
            if i % 4 == 0:
                # Sine wave
                audio = np.sin(frequency * 2 * np.pi * t).astype(np.float32)
            elif i % 4 == 1:
                # Cosine wave
                audio = np.cos(frequency * 2 * np.pi * t).astype(np.float32)
            elif i % 4 == 2:
                # Square wave
                audio = np.sign(np.sin(frequency * 2 * np.pi * t)).astype(np.float32)
            else:
                # White noise
                audio = np.random.randn(len(t)).astype(np.float32) * 0.1
            
            audio_samples.append(audio)
        
        return audio_samples
    
    def render_synthetic_data(self):
        """Render synthetic data interface."""
        st.markdown("**Generate Synthetic Audio Data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input("Batch size:", min_value=1, max_value=16, value=4)
            sequence_length = st.number_input("Sequence length:", min_value=1000, max_value=100000, value=16000)
        
        with col2:
            sample_rate = st.selectbox("Sample rate:", [8000, 16000, 22050, 44100, 48000], index=1)
            noise_level = st.slider("Noise level:", 0.0, 1.0, 0.1, 0.1)
        
        if st.button("🎲 Generate Synthetic Data"):
            if NUMPY_AVAILABLE:
                synthetic_audio = np.random.randn(batch_size, sequence_length).astype(np.float32) * noise_level
                st.session_state.feature_extractors_page_state['uploaded_audio'] = [synthetic_audio]
                
                st.success(f"✅ Generated synthetic audio data: {synthetic_audio.shape}")
                st.write(f"📊 Batch size: {batch_size}")
                st.write(f"🎧 Sequence length: {sequence_length}")
                st.write(f"📈 Sample rate: {sample_rate}Hz")
            else:
                st.error("NumPy not available for synthetic data generation")
    
    def render_advanced_options(self):
        """Render advanced audio processing options."""
        with st.expander("⚙️ Advanced Processing Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                sampling_rate = st.number_input("Sampling Rate", min_value=8000, max_value=48000, value=16000, step=1000, help="Target sampling rate for processing")
                padding = st.checkbox("Padding", value=True, help="Pad sequences to uniform length")
                truncation = st.checkbox("Truncation", value=False, help="Truncate sequences to maximum length")
                return_tensors = st.selectbox("Return Tensors", ["None", "pt", "tf", "np"], index=0, help="Tensor format for output")
            
            with col2:
                max_length = st.number_input("Max Length", min_value=1000, max_value=500000, value=50000, help="Maximum sequence length for truncation")
                return_attention_mask = st.checkbox("Return Attention Mask", value=True, help="Include attention mask in output")
                normalize = st.checkbox("Normalize", value=False, help="Normalize audio values")
                
            # Store options
            st.session_state.feature_extractors_page_state['advanced_options'] = {
                'sampling_rate': sampling_rate,
                'padding': padding,
                'truncation': truncation,
                'max_length': max_length if truncation else None,
                'return_tensors': return_tensors if return_tensors != "None" else None,
                'return_attention_mask': return_attention_mask,
                'normalize': normalize
            }
    
    def render_processing_interface(self, model_id: str):
        """Render audio processing interface."""
        st.markdown("### 🔄 Audio Processing")
        
        # Advanced options
        self.render_advanced_options()
        
        # Process button
        if st.button("🚀 Process Audio", type="primary"):
            self.process_audio(model_id)
    
    def process_audio(self, model_id: str):
        """Process audio with the selected options."""
        audio_data = st.session_state.feature_extractors_page_state.get('uploaded_audio', [])
        
        if not audio_data:
            st.error("No audio data provided for processing")
            return
        
        try:
            # Get options
            options = st.session_state.feature_extractors_page_state.get('advanced_options', {})
            
            # Process audio
            with st.spinner("Processing audio..."):
                start_time = time.time()
                result = feature_extractor_manager.process_audio(
                    audio_data, model_id, **options
                )
                processing_time = time.time() - start_time
            
            if result and result.success:
                self.display_results(result, processing_time)
                log_user_action("audio_processed", model_id=model_id, num_audio=len(audio_data))
            else:
                st.error(f"Audio processing failed: {result.error if result else 'Unknown error'}")
                
        except Exception as e:
            st.error(f"Audio processing failed: {str(e)}")
            error(f"Audio processing failed", "feature_extractors_page", e)
    
    def display_results(self, result: AudioProcessingResult, processing_time: float):
        """Display audio processing results."""
        st.markdown("### 📊 Audio Processing Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Batch Size", result.batch_size)
            st.metric("Sequence Length", result.sequence_length)
        
        with col2:
            if result.batch_size > 0:
                avg_time = processing_time / result.batch_size
                st.metric("Avg Time per Sample", f"{avg_time:.3f}s")
                st.metric("Throughput", f"{result.batch_size/processing_time:.1f} samples/sec")
            
            if result.feature_extractor_info:
                st.metric("Sampling Rate", f"{result.feature_extractor_info.sampling_rate}Hz")
        
        with col3:
            if result.input_values is not None:
                if isinstance(result.input_values, list):
                    st.metric("Output Arrays", len(result.input_values))
                    if len(result.input_values) > 0:
                        st.metric("Array Shape", str(np.array(result.input_values[0]).shape))
                else:
                    st.metric("Output Shape", str(result.input_values.shape))
        
        # Detailed results
        if result.success:
            st.markdown("**📝 Detailed Results:**")
            
            # Feature extractor info
            if result.feature_extractor_info:
                with st.expander("🔧 Feature Extractor Information"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Model:** {result.feature_extractor_info.model_id}")
                        st.write(f"**Type:** {result.feature_extractor_info.extractor_type}")
                        st.write(f"**Sampling Rate:** {result.feature_extractor_info.sampling_rate}Hz")
                        st.write(f"**Max Length:** {result.feature_extractor_info.max_length}")
                    
                    with col2:
                        st.write(f"**Supports Padding:** {result.feature_extractor_info.supports_padding}")
                        st.write(f"**Supports Truncation:** {result.feature_extractor_info.supports_truncation}")
                        st.write(f"**Supports Resampling:** {result.feature_extractor_info.supports_resampling}")
                        st.write(f"**Supports Batch:** {result.feature_extractor_info.supports_batch}")
            
            # Input values info
            if result.input_values is not None:
                with st.expander("🔢 Input Values Information"):
                    if isinstance(result.input_values, list):
                        st.write(f"**Number of arrays:** {len(result.input_values)}")
                        if len(result.input_values) > 0:
                            first_array = np.array(result.input_values[0])
                            st.write(f"**First array shape:** {first_array.shape}")
                            st.write(f"**Data type:** {first_array.dtype}")
                            st.write(f"**Min value:** {first_array.min():.6f}")
                            st.write(f"**Max value:** {first_array.max():.6f}")
                            st.write(f"**Mean value:** {first_array.mean():.6f}")
                    else:
                        st.write(f"**Shape:** {result.input_values.shape}")
                        st.write(f"**Data type:** {result.input_values.dtype}")
                        if hasattr(result.input_values, 'min'):
                            st.write(f"**Min value:** {result.input_values.min():.6f}")
                            st.write(f"**Max value:** {result.input_values.max():.6f}")
                            st.write(f"**Mean value:** {result.input_values.mean():.6f}")
            
            # Attention mask info
            if result.attention_mask is not None:
                with st.expander("🎭 Attention Mask Information"):
                    if isinstance(result.attention_mask, list):
                        st.write(f"**Number of masks:** {len(result.attention_mask)}")
                        if len(result.attention_mask) > 0:
                            first_mask = np.array(result.attention_mask[0])
                            st.write(f"**First mask shape:** {first_mask.shape}")
                            st.write(f"**Unique values:** {np.unique(first_mask)}")
                    else:
                        st.write(f"**Shape:** {result.attention_mask.shape}")
                        st.write(f"**Unique values:** {np.unique(result.attention_mask)}")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.feature_extractors_page_state.get('performance_stats')
        
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
                st.metric("Batch Processings", stats['batch_processings'])
                st.metric("Total Audio Samples", stats['total_audio_samples'])
                st.metric("Resampling Operations", stats['resampling_operations'])
    
    def render_cached_extractors(self):
        """Render cached extractors information."""
        cached = st.session_state.feature_extractors_page_state.get('cached_extractors', [])
        
        if cached:
            st.markdown("### 💾 Cached Feature Extractors")
            
            for extractor_info in cached:
                with st.expander(f"🤖 {extractor_info['model_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {extractor_info['extractor_type']}")
                        st.write(f"**Sampling Rate:** {extractor_info['sampling_rate']}Hz")
                        st.write(f"**Supports Padding:** {extractor_info.get('supports_padding', False)}")
                    
                    with col2:
                        st.write(f"**Load Time:** {extractor_info['load_time']:.3f}s")
                        st.write(f"**Max Length:** {extractor_info.get('max_length', 'None')}")
                        st.write(f"**Cache Key:** {extractor_info['cache_key'][:16]}...")
    
    def render(self):
        """Render the complete feature extractors page."""
        self.render_header()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Feature extractor controls
        self.render_processor_controls(selected_model)
        
        # Audio input
        self.render_audio_input()
        
        # Processing interface
        if st.session_state.feature_extractors_page_state.get('uploaded_audio'):
            self.render_processing_interface(selected_model)
        
        # Performance stats
        if st.session_state.feature_extractors_page_state.get('performance_stats'):
            self.render_performance_stats()
        
        # Cached extractors
        if st.session_state.feature_extractors_page_state.get('cached_extractors'):
            self.render_cached_extractors()


def render_feature_extractors_page():
    """Render function for the feature extractors page."""
    try:
        page = FeatureExtractorsPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering feature extractors page: {str(e)}")
        error(f"Failed to render feature extractors page", "feature_extractors_page", e)


if __name__ == "__main__":
    # For testing
    render_feature_extractors_page()
