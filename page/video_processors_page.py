"""
Video Processors Page for DurgasAI.

This module provides a comprehensive interface for video processor management and testing:
- Video processor selection and configuration
- Real-time video processing testing
- Batch processing capabilities
- Performance monitoring and statistics
- Advanced video processing features (resize, normalize, temporal processing)
- Integration with enhanced video processor manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.video_processor_manager import video_processor_manager, VideoProcessorInfo, VideoProcessingResult
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoVideoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class VideoProcessorsPage:
    """
    Comprehensive video processor management and testing interface.
    
    This class provides:
    - Video processor selection and loading
    - Real-time video processing testing
    - Batch processing capabilities
    - Performance monitoring
    - Advanced configuration options
    - Integration with enhanced video processor manager
    """
    
    def __init__(self):
        """Initialize the video processors page."""
        debug("Initializing VideoProcessorsPage", "video_processors_page")
        self.config = Config()
        
        # Initialize session state for video processors page
        if 'video_processor_page_state' not in st.session_state:
            st.session_state.video_processor_page_state = {
                'selected_model': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
                'uploaded_videos': [],
                'batch_mode': False,
                'show_advanced': False,
                'performance_stats': None,
                'processed_videos': []
            }
        
        info("VideoProcessorsPage initialized successfully", "video_processors_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🎥 Video Processors Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📝 About Video Processors</h4>
        <p>Video processors convert videos into numerical representations (pixel values) that vision AI models can understand. 
        This page provides comprehensive tools for:</p>
        <ul>
        <li><strong>Processor Selection:</strong> Choose from various HuggingFace video processors</li>
        <li><strong>Real-time Processing:</strong> Process videos with your own settings</li>
        <li><strong>Batch Processing:</strong> Process multiple videos efficiently</li>
        <li><strong>Performance Monitoring:</strong> Track processor performance and caching</li>
        <li><strong>Advanced Features:</strong> Resize, normalize, temporal processing, and more</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_model_selection(self):
        """Render video processor model selection interface."""
        st.markdown("### 🤖 Select Video Processor Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular video processor models
            popular_models = [
                "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                "llava-hf/llava-1.5-7b-hf",
                "microsoft/git-base-videocap",
                "microsoft/xclip-base-patch32",
                "facebook/timesformer-base-finetuned-k400"
            ]
            
            selected_model = st.selectbox(
                "Choose a video processor model:",
                options=popular_models,
                index=popular_models.index(st.session_state.video_processor_page_state['selected_model']) 
                if st.session_state.video_processor_page_state['selected_model'] in popular_models else 0,
                help="Select a HuggingFace video processor model to work with"
            )
            
            # Custom model input
            custom_model = st.text_input(
                "Or enter custom model ID:",
                placeholder="e.g., microsoft/swin-base-patch4-window7-224",
                help="Enter any HuggingFace model ID for custom video processor"
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
                    st.write(f"**Fast:** {model_info.get('is_fast', False)}")
                    st.write(f"**GPU Support:** {model_info.get('supports_gpu', False)}")
                    st.write(f"**Compilation:** {model_info.get('supports_compilation', False)}")
                else:
                    st.info("Click 'Load Processor' to get model information")
        
        # Update session state
        st.session_state.video_processor_page_state['selected_model'] = selected_model
        
        return selected_model
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get video processor model information."""
        try:
            processor_info = video_processor_manager.get_processor_info(model_id)
            if processor_info:
                return processor_info
        except Exception as e:
            debug(f"Failed to get model info: {e}", "video_processors_page")
        
        return None
    
    def render_processor_controls(self, model_id: str):
        """Render video processor loading and control buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Load Processor", type="primary"):
                with st.spinner(f"Loading video processor for {model_id}..."):
                    start_time = time.time()
                    processor_info = video_processor_manager.load_processor(model_id)
                    load_time = time.time() - start_time
                    
                    if processor_info:
                        st.success(f"✅ Video processor loaded in {load_time:.2f}s")
                        log_user_action("video_processor_loaded", model_id=model_id, load_time=load_time)
                    else:
                        st.error("❌ Failed to load video processor")
                        log_user_action("video_processor_load_failed", model_id=model_id)
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = video_processor_manager.get_performance_stats()
                st.session_state.video_processor_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                video_processor_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("video_processor_cache_cleared")
        
        with col4:
            if st.button("📋 List Cached"):
                cached = video_processor_manager.list_cached_processors()
                st.session_state.video_processor_page_state['cached_processors'] = cached
    
    def render_video_input(self):
        """Render video input interface."""
        st.markdown("### 🎬 Video Input")
        
        # Mode selection
        col1, col2 = st.columns([1, 3])
        with col1:
            batch_mode = st.checkbox(
                "Batch Mode", 
                value=st.session_state.video_processor_page_state['batch_mode'],
                help="Process multiple videos at once for better performance"
            )
            st.session_state.video_processor_page_state['batch_mode'] = batch_mode
        
        with col2:
            if batch_mode:
                st.info("💡 Batch mode processes multiple videos efficiently using optimized processing")
        
        # Video input method
        input_method = st.radio(
            "Choose input method:",
            ["Generate Sample Videos", "Use Synthetic Data", "Upload Videos (Coming Soon)"],
            horizontal=True
        )
        
        if input_method == "Generate Sample Videos":
            self.render_sample_generation()
        elif input_method == "Use Synthetic Data":
            self.render_synthetic_data()
        elif input_method == "Upload Videos (Coming Soon)":
            st.info("🎬 Video upload functionality will be available in future updates")
    
    def render_sample_generation(self):
        """Render sample video generation interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            num_videos = st.number_input("Number of sample videos:", min_value=1, max_value=8, value=3)
            frames = st.number_input("Frames per video:", min_value=4, max_value=32, value=8)
            video_size = st.selectbox("Video size:", [(224, 224), (256, 256), (512, 512), (64, 64)], index=0)
        
        with col2:
            color_mode = st.selectbox("Color mode:", ["RGB", "Grayscale"])
            pattern = st.selectbox("Pattern:", ["Random", "Gradient", "Solid Color", "Moving Pattern"])
        
        if st.button("🎨 Generate Sample Videos"):
            if TORCH_AVAILABLE:
                videos = self.generate_sample_videos(num_videos, frames, video_size, color_mode, pattern)
                st.session_state.video_processor_page_state['uploaded_videos'] = videos
                
                # Display generated videos info
                st.markdown("**Generated Sample Videos:**")
                for i, video in enumerate(videos):
                    st.write(f"Video {i+1}: Shape {video.shape}, Frames: {video.shape[1] if len(video.shape) > 1 else 1}")
            else:
                st.error("PyTorch not available for video generation")
    
    def generate_sample_videos(self, num_videos: int, frames: int, size: tuple, color_mode: str, pattern: str) -> List[torch.Tensor]:
        """Generate sample videos for testing."""
        videos = []
        
        for i in range(num_videos):
            if pattern == "Random":
                if color_mode == 'RGB':
                    video = torch.randn(1, frames, size[0], size[1], 3)
                else:
                    video = torch.randn(1, frames, size[0], size[1], 1)
            
            elif pattern == "Gradient":
                if color_mode == 'RGB':
                    video = torch.zeros(1, frames, size[0], size[1], 3)
                    for f in range(frames):
                        for y in range(size[0]):
                            for x in range(size[1]):
                                r = (x / size[1]) * 255
                                g = (y / size[0]) * 255
                                b = (f / frames) * 255
                                video[0, f, y, x] = torch.tensor([r, g, b]) / 255.0
                else:
                    video = torch.zeros(1, frames, size[0], size[1], 1)
                    for f in range(frames):
                        for y in range(size[0]):
                            for x in range(size[1]):
                                gray = ((x + y + f) / (size[0] + size[1] + frames)) * 255
                                video[0, f, y, x, 0] = gray / 255.0
            
            elif pattern == "Solid Color":
                colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
                color = colors[i % len(colors)]
                if color_mode == 'RGB':
                    video = torch.full((1, frames, size[0], size[1], 3), color)
                else:
                    video = torch.full((1, frames, size[0], size[1], 1), color[0])
            
            elif pattern == "Moving Pattern":
                video = torch.zeros(1, frames, size[0], size[1], 3 if color_mode == 'RGB' else 1)
                for f in range(frames):
                    offset = f * 2
                    for y in range(size[0]):
                        for x in range(size[1]):
                            if ((x + offset) // 32 + (y + offset) // 32) % 2 == 0:
                                if color_mode == 'RGB':
                                    video[0, f, y, x] = torch.tensor([1.0, 1.0, 1.0])
                                else:
                                    video[0, f, y, x, 0] = 1.0
            
            videos.append(video)
        
        return videos
    
    def render_synthetic_data(self):
        """Render synthetic data interface."""
        st.markdown("**Generate Synthetic Video Data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input("Batch size:", min_value=1, max_value=16, value=4)
            frames = st.number_input("Frames:", min_value=4, max_value=64, value=8)
        
        with col2:
            height = st.number_input("Height:", min_value=64, max_value=1024, value=224)
            width = st.number_input("Width:", min_value=64, max_value=1024, value=224)
        
        if st.button("🎲 Generate Synthetic Data"):
            if TORCH_AVAILABLE:
                synthetic_video = torch.randn(batch_size, frames, height, width, 3)
                st.session_state.video_processor_page_state['uploaded_videos'] = [synthetic_video]
                
                st.success(f"✅ Generated synthetic video data: {synthetic_video.shape}")
                st.write(f"📊 Batch size: {batch_size}")
                st.write(f"🎬 Frames: {frames}")
                st.write(f"📐 Resolution: {height}x{width}")
            else:
                st.error("PyTorch not available for synthetic data generation")
    
    def render_advanced_options(self):
        """Render advanced video processing options."""
        with st.expander("⚙️ Advanced Processing Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                do_resize = st.checkbox("Resize", value=True, help="Resize video frames to model requirements")
                do_normalize = st.checkbox("Normalize", value=True, help="Normalize pixel values")
                do_rescale = st.checkbox("Rescale", value=True, help="Rescale pixel values")
                compile_processor = st.checkbox("Compile Processor", value=False, help="Compile processor for maximum performance")
            
            with col2:
                return_tensors = st.selectbox("Return Tensors", ["pt", "tf", "np", "None"], index=0, help="Tensor format for output")
                device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help="Processing device")
                max_frames = st.number_input("Max Frames", min_value=1, max_value=128, value=16, help="Maximum frames to process")
            
            # Store options
            st.session_state.video_processor_page_state['advanced_options'] = {
                'do_resize': do_resize,
                'do_normalize': do_normalize,
                'do_rescale': do_rescale,
                'return_tensors': return_tensors if return_tensors != "None" else None,
                'device': device if device != "auto" else None,
                'compile_processor': compile_processor,
                'max_frames': max_frames
            }
    
    def render_processing_interface(self, model_id: str):
        """Render video processing interface."""
        st.markdown("### 🔄 Video Processing")
        
        # Advanced options
        self.render_advanced_options()
        
        # Process button
        if st.button("🚀 Process Videos", type="primary"):
            self.process_videos(model_id)
    
    def process_videos(self, model_id: str):
        """Process videos with the selected options."""
        videos = st.session_state.video_processor_page_state.get('uploaded_videos', [])
        
        if not videos:
            st.error("No videos provided for processing")
            return
        
        try:
            # Get options
            options = st.session_state.video_processor_page_state.get('advanced_options', {})
            compile_processor = options.pop('compile_processor', False)
            
            # Process videos
            with st.spinner("Processing videos..."):
                start_time = time.time()
                result = video_processor_manager.process_videos(
                    videos, model_id, compile_processor=compile_processor, **options
                )
                processing_time = time.time() - start_time
            
            if result and result.success:
                self.display_results(result, processing_time)
                log_user_action("videos_processed", model_id=model_id, batch_size=len(videos))
            else:
                st.error(f"Video processing failed: {result.error if result else 'Unknown error'}")
                
        except Exception as e:
            st.error(f"Video processing failed: {str(e)}")
            error(f"Video processing failed", "video_processors_page", e)
    
    def display_results(self, result: VideoProcessingResult, processing_time: float):
        """Display video processing results."""
        st.markdown("### 📊 Processing Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Batch Size", result.batch_size)
            st.metric("Frame Count", result.frame_count)
        
        with col2:
            if result.batch_size > 0:
                avg_time = processing_time / result.batch_size
                st.metric("Avg Time per Video", f"{avg_time:.3f}s")
                st.metric("Throughput", f"{result.batch_size/processing_time:.1f} videos/sec")
            
            if result.frame_count > 0:
                frames_per_sec = result.frame_count / processing_time
                st.metric("Frame Rate", f"{frames_per_sec:.1f} frames/sec")
        
        with col3:
            if result.pixel_values is not None:
                st.metric("Output Shape", str(result.pixel_values.shape))
                if hasattr(result.pixel_values, 'min') and hasattr(result.pixel_values, 'max'):
                    st.metric("Pixel Range", f"{result.pixel_values.min():.3f} - {result.pixel_values.max():.3f}")
        
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
                        st.write(f"**Fast:** {result.processor_info.is_fast}")
                        st.write(f"**GPU Optimized:** {result.processor_info.gpu_optimized}")
                    
                    with col2:
                        st.write(f"**Size:** {result.processor_info.size}")
                        st.write(f"**Max Frames:** {result.processor_info.max_frames}")
                        st.write(f"**Frame Sampling Rate:** {result.processor_info.frame_sampling_rate}")
                        st.write(f"**Compilation Support:** {result.processor_info.supports_compilation}")
            
            # Pixel values info
            if result.pixel_values is not None:
                with st.expander("🔢 Pixel Values Information"):
                    st.write(f"**Shape:** {result.pixel_values.shape}")
                    st.write(f"**Data Type:** {result.pixel_values.dtype}")
                    
                    if hasattr(result.pixel_values, 'device'):
                        st.write(f"**Device:** {result.pixel_values.device}")
                    
                    # Show sample values
                    if result.batch_size == 1 and len(result.pixel_values.shape) >= 4:
                        sample_values = result.pixel_values[0, :2, :3, :3, :3]  # First 2 frames, first 3x3 pixels, RGB
                        st.write(f"**Sample Values (first 2 frames, 3x3 pixels, RGB):**")
                        st.write(sample_values)
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.video_processor_page_state.get('performance_stats')
        
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
                st.metric("GPU Processings", stats['gpu_processings'])
                st.metric("Compiled Processings", stats['compiled_processings'])
    
    def render_cached_processors(self):
        """Render cached processors information."""
        cached = st.session_state.video_processor_page_state.get('cached_processors', [])
        
        if cached:
            st.markdown("### 💾 Cached Video Processors")
            
            for processor_info in cached:
                with st.expander(f"🤖 {processor_info['model_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {processor_info['processor_type']}")
                        st.write(f"**Fast:** {processor_info['is_fast']}")
                        st.write(f"**GPU Support:** {processor_info['supports_gpu']}")
                    
                    with col2:
                        st.write(f"**Load Time:** {processor_info['load_time']:.3f}s")
                        st.write(f"**Compilation Support:** {processor_info['supports_compilation']}")
                        st.write(f"**Cache Key:** {processor_info['cache_key'][:16]}...")
    
    def render(self):
        """Render the complete video processors page."""
        self.render_header()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Processor controls
        self.render_processor_controls(selected_model)
        
        # Video input
        self.render_video_input()
        
        # Processing interface
        if st.session_state.video_processor_page_state.get('uploaded_videos'):
            self.render_processing_interface(selected_model)
        
        # Performance stats
        if st.session_state.video_processor_page_state.get('performance_stats'):
            self.render_performance_stats()
        
        # Cached processors
        if st.session_state.video_processor_page_state.get('cached_processors'):
            self.render_cached_processors()


def render_video_processors_page():
    """Render function for the video processors page."""
    try:
        page = VideoProcessorsPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering video processors page: {str(e)}")
        error(f"Failed to render video processors page", "video_processors_page", e)


if __name__ == "__main__":
    # For testing
    render_video_processors_page()
