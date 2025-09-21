"""
Image Processors Page for DurgasAI.

This module provides a comprehensive interface for image processor management and testing:
- Image processor selection and configuration
- Real-time image processing testing
- Batch processing capabilities
- Performance monitoring and statistics
- Advanced image processing features (resize, normalize, augmentation)
- Integration with enhanced image processor manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.image_processor_manager import image_processor_manager, ImageProcessorInfo, ImageProcessingResult
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from transformers import AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ImageProcessorsPage:
    """
    Comprehensive image processor management and testing interface.
    
    This class provides:
    - Image processor selection and loading
    - Real-time image processing testing
    - Batch processing capabilities
    - Performance monitoring
    - Advanced configuration options
    - Integration with enhanced image processor manager
    """
    
    def __init__(self):
        """Initialize the image processors page."""
        debug("Initializing ImageProcessorsPage", "image_processors_page")
        self.config = Config()
        
        # Initialize session state for image processors page
        if 'image_processor_page_state' not in st.session_state:
            st.session_state.image_processor_page_state = {
                'selected_model': 'google/vit-base-patch16-224',
                'uploaded_images': [],
                'batch_mode': False,
                'show_advanced': False,
                'performance_stats': None,
                'processed_images': []
            }
        
        info("ImageProcessorsPage initialized successfully", "image_processors_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🖼️ Image Processors Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📝 About Image Processors</h4>
        <p>Image processors convert images into numerical representations (pixel values) that vision AI models can understand. 
        This page provides comprehensive tools for:</p>
        <ul>
        <li><strong>Processor Selection:</strong> Choose from various HuggingFace image processors</li>
        <li><strong>Real-time Processing:</strong> Process images with your own settings</li>
        <li><strong>Batch Processing:</strong> Process multiple images efficiently</li>
        <li><strong>Performance Monitoring:</strong> Track processor performance and caching</li>
        <li><strong>Advanced Features:</strong> Resize, normalize, augment, and more</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_model_selection(self):
        """Render image processor model selection interface."""
        st.markdown("### 🤖 Select Image Processor Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular image processor models
            popular_models = [
                "google/vit-base-patch16-224",
                "facebook/detr-resnet-50",
                "microsoft/resnet-50",
                "google/vit-large-patch16-224",
                "facebook/convnext-base-224",
                "timm/resnet50.a1_in1k"
            ]
            
            selected_model = st.selectbox(
                "Choose an image processor model:",
                options=popular_models,
                index=popular_models.index(st.session_state.image_processor_page_state['selected_model']) 
                if st.session_state.image_processor_page_state['selected_model'] in popular_models else 0,
                help="Select a HuggingFace image processor model to work with"
            )
            
            # Custom model input
            custom_model = st.text_input(
                "Or enter custom model ID:",
                placeholder="e.g., microsoft/swin-base-patch4-window7-224",
                help="Enter any HuggingFace model ID for custom image processor"
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
                    st.write(f"**Size:** {model_info.get('size', 'Unknown')}")
                    st.write(f"**GPU Optimized:** {model_info.get('gpu_optimized', False)}")
                else:
                    st.info("Click 'Load Processor' to get model information")
        
        # Update session state
        st.session_state.image_processor_page_state['selected_model'] = selected_model
        
        return selected_model
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get image processor model information."""
        try:
            processor_info = image_processor_manager.get_processor_info(model_id)
            if processor_info:
                return processor_info
        except Exception as e:
            debug(f"Failed to get model info: {e}", "image_processors_page")
        
        return None
    
    def render_processor_controls(self, model_id: str):
        """Render image processor loading and control buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Load Processor", type="primary"):
                with st.spinner(f"Loading image processor for {model_id}..."):
                    start_time = time.time()
                    processor_info = image_processor_manager.load_processor(model_id)
                    load_time = time.time() - start_time
                    
                    if processor_info:
                        st.success(f"✅ Image processor loaded in {load_time:.2f}s")
                        log_user_action("image_processor_loaded", model_id=model_id, load_time=load_time)
                    else:
                        st.error("❌ Failed to load image processor")
                        log_user_action("image_processor_load_failed", model_id=model_id)
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = image_processor_manager.get_performance_stats()
                st.session_state.image_processor_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                image_processor_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("image_processor_cache_cleared")
        
        with col4:
            if st.button("📋 List Cached"):
                cached = image_processor_manager.list_cached_processors()
                st.session_state.image_processor_page_state['cached_processors'] = cached
    
    def render_image_input(self):
        """Render image input interface."""
        st.markdown("### 🖼️ Image Input")
        
        # Mode selection
        col1, col2 = st.columns([1, 3])
        with col1:
            batch_mode = st.checkbox(
                "Batch Mode", 
                value=st.session_state.image_processor_page_state['batch_mode'],
                help="Process multiple images at once for better performance"
            )
            st.session_state.image_processor_page_state['batch_mode'] = batch_mode
        
        with col2:
            if batch_mode:
                st.info("💡 Batch mode processes multiple images efficiently using optimized processing")
        
        # Image upload or generation
        input_method = st.radio(
            "Choose input method:",
            ["Upload Images", "Generate Sample Images", "Use URL"],
            horizontal=True
        )
        
        if input_method == "Upload Images":
            self.render_image_upload(batch_mode)
        elif input_method == "Generate Sample Images":
            self.render_sample_generation()
        elif input_method == "Use URL":
            self.render_url_input()
    
    def render_image_upload(self, batch_mode: bool):
        """Render image upload interface."""
        if batch_mode:
            uploaded_files = st.file_uploader(
                "Upload multiple images:",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
        else:
            uploaded_files = st.file_uploader(
                "Upload an image:",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image to process"
            )
        
        if uploaded_files:
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]
            
            # Convert uploaded files to PIL Images
            images = []
            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append(image)
                except Exception as e:
                    st.error(f"Failed to load image {uploaded_file.name}: {str(e)}")
            
            if images:
                st.session_state.image_processor_page_state['uploaded_images'] = images
                
                # Display uploaded images
                st.markdown("**Uploaded Images:**")
                cols = st.columns(min(len(images), 4))
                for i, image in enumerate(images):
                    with cols[i % 4]:
                        st.image(image, caption=f"Image {i+1}", use_column_width=True)
                        st.write(f"Size: {image.size}")
    
    def render_sample_generation(self):
        """Render sample image generation interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            num_images = st.number_input("Number of sample images:", min_value=1, max_value=8, value=3)
            image_size = st.selectbox("Image size:", [(224, 224), (256, 256), (512, 512), (64, 64)], index=0)
        
        with col2:
            color_mode = st.selectbox("Color mode:", ["RGB", "Grayscale"])
            pattern = st.selectbox("Pattern:", ["Solid Color", "Gradient", "Random", "Checkerboard"])
        
        if st.button("🎨 Generate Sample Images"):
            if PIL_AVAILABLE:
                images = self.generate_sample_images(num_images, image_size, color_mode, pattern)
                st.session_state.image_processor_page_state['uploaded_images'] = images
                
                # Display generated images
                st.markdown("**Generated Sample Images:**")
                cols = st.columns(min(len(images), 4))
                for i, image in enumerate(images):
                    with cols[i % 4]:
                        st.image(image, caption=f"Sample {i+1}", use_column_width=True)
            else:
                st.error("PIL not available for image generation")
    
    def generate_sample_images(self, num_images: int, size: tuple, color_mode: str, pattern: str) -> List[Image.Image]:
        """Generate sample images for testing."""
        images = []
        
        for i in range(num_images):
            if pattern == "Solid Color":
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                color = colors[i % len(colors)]
                image = Image.new('RGB' if color_mode == 'RGB' else 'L', size, color)
            
            elif pattern == "Gradient":
                if color_mode == 'RGB':
                    image = Image.new('RGB', size)
                    pixels = []
                    for y in range(size[1]):
                        for x in range(size[0]):
                            r = int(255 * x / size[0])
                            g = int(255 * y / size[1])
                            b = int(255 * (x + y) / (size[0] + size[1]))
                            pixels.append((r, g, b))
                    image.putdata(pixels)
                else:
                    image = Image.new('L', size)
                    pixels = []
                    for y in range(size[1]):
                        for x in range(size[0]):
                            gray = int(255 * (x + y) / (size[0] + size[1]))
                            pixels.append(gray)
                    image.putdata(pixels)
            
            elif pattern == "Random":
                if color_mode == 'RGB':
                    pixels = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) 
                             for _ in range(size[0] * size[1])]
                    image = Image.new('RGB', size)
                    image.putdata(pixels)
                else:
                    pixels = [np.random.randint(0, 256) for _ in range(size[0] * size[1])]
                    image = Image.new('L', size)
                    image.putdata(pixels)
            
            elif pattern == "Checkerboard":
                image = Image.new('RGB' if color_mode == 'RGB' else 'L', size)
                pixels = []
                for y in range(size[1]):
                    for x in range(size[0]):
                        if (x // 32 + y // 32) % 2 == 0:
                            pixel = (255, 255, 255) if color_mode == 'RGB' else 255
                        else:
                            pixel = (0, 0, 0) if color_mode == 'RGB' else 0
                        pixels.append(pixel)
                image.putdata(pixels)
            
            images.append(image)
        
        return images
    
    def render_url_input(self):
        """Render URL input interface."""
        st.markdown("**Enter image URL(s):**")
        url_input = st.text_area(
            "Image URLs (one per line):",
            placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg",
            height=100
        )
        
        if url_input and st.button("📥 Load from URLs"):
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            images = []
            
            for url in urls:
                try:
                    import requests
                    response = requests.get(url, timeout=10)
                    image = Image.open(BytesIO(response.content))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append(image)
                    st.success(f"✅ Loaded image from {url}")
                except Exception as e:
                    st.error(f"❌ Failed to load image from {url}: {str(e)}")
            
            if images:
                st.session_state.image_processor_page_state['uploaded_images'] = images
    
    def render_advanced_options(self):
        """Render advanced image processing options."""
        with st.expander("⚙️ Advanced Processing Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                do_resize = st.checkbox("Resize", value=True, help="Resize images to model requirements")
                do_normalize = st.checkbox("Normalize", value=True, help="Normalize pixel values")
                do_rescale = st.checkbox("Rescale", value=True, help="Rescale pixel values")
            
            with col2:
                return_tensors = st.selectbox("Return Tensors", ["pt", "tf", "np", "None"], index=0, help="Tensor format for output")
                device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help="Processing device")
            
            # Store options
            st.session_state.image_processor_page_state['advanced_options'] = {
                'do_resize': do_resize,
                'do_normalize': do_normalize,
                'do_rescale': do_rescale,
                'return_tensors': return_tensors if return_tensors != "None" else None,
                'device': device if device != "auto" else None
            }
    
    def render_processing_interface(self, model_id: str):
        """Render image processing interface."""
        st.markdown("### 🔄 Image Processing")
        
        # Advanced options
        self.render_advanced_options()
        
        # Process button
        if st.button("🚀 Process Images", type="primary"):
            self.process_images(model_id)
    
    def process_images(self, model_id: str):
        """Process images with the selected options."""
        images = st.session_state.image_processor_page_state.get('uploaded_images', [])
        
        if not images:
            st.error("No images provided for processing")
            return
        
        try:
            # Get options
            options = st.session_state.image_processor_page_state.get('advanced_options', {})
            
            # Process images
            with st.spinner("Processing images..."):
                start_time = time.time()
                result = image_processor_manager.process_images(images, model_id, **options)
                processing_time = time.time() - start_time
            
            if result and result.success:
                self.display_results(result, processing_time)
                log_user_action("images_processed", model_id=model_id, batch_size=len(images))
            else:
                st.error(f"Image processing failed: {result.error if result else 'Unknown error'}")
                
        except Exception as e:
            st.error(f"Image processing failed: {str(e)}")
            error(f"Image processing failed", "image_processors_page", e)
    
    def display_results(self, result: ImageProcessingResult, processing_time: float):
        """Display image processing results."""
        st.markdown("### 📊 Processing Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Batch Size", result.batch_size)
            st.metric("Success", "✅" if result.success else "❌")
        
        with col2:
            if result.batch_size > 0:
                avg_time = processing_time / result.batch_size
                st.metric("Avg Time per Image", f"{avg_time:.3f}s")
                st.metric("Throughput", f"{result.batch_size/processing_time:.1f} images/sec")
        
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
                        st.write(f"**Image Mean:** {result.processor_info.image_mean}")
                        st.write(f"**Image Std:** {result.processor_info.image_std}")
            
            # Pixel values info
            if result.pixel_values is not None:
                with st.expander("🔢 Pixel Values Information"):
                    st.write(f"**Shape:** {result.pixel_values.shape}")
                    st.write(f"**Data Type:** {result.pixel_values.dtype}")
                    
                    if hasattr(result.pixel_values, 'device'):
                        st.write(f"**Device:** {result.pixel_values.device}")
                    
                    # Show sample values
                    if result.batch_size == 1:
                        sample_values = result.pixel_values[0, :3, :3, :3]  # First 3x3 pixels of first 3 channels
                        st.write(f"**Sample Values (first 3x3 pixels, RGB):**")
                        st.write(sample_values)
            
            # Pixel mask info
            if result.pixel_mask is not None:
                with st.expander("🎭 Pixel Mask Information"):
                    st.write(f"**Shape:** {result.pixel_mask.shape}")
                    st.write(f"**Unique Values:** {result.pixel_mask.unique().tolist()}")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.image_processor_page_state.get('performance_stats')
        
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
    
    def render_cached_processors(self):
        """Render cached processors information."""
        cached = st.session_state.image_processor_page_state.get('cached_processors', [])
        
        if cached:
            st.markdown("### 💾 Cached Image Processors")
            
            for processor_info in cached:
                with st.expander(f"🤖 {processor_info['model_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {processor_info['processor_type']}")
                        st.write(f"**Fast:** {processor_info['is_fast']}")
                        st.write(f"**Size:** {processor_info['size']}")
                    
                    with col2:
                        st.write(f"**Load Time:** {processor_info['load_time']:.3f}s")
                        st.write(f"**GPU Optimized:** {processor_info['gpu_optimized']}")
                        st.write(f"**Cache Key:** {processor_info['cache_key'][:16]}...")
    
    def render(self):
        """Render the complete image processors page."""
        self.render_header()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Processor controls
        self.render_processor_controls(selected_model)
        
        # Image input
        self.render_image_input()
        
        # Processing interface
        if st.session_state.image_processor_page_state.get('uploaded_images'):
            self.render_processing_interface(selected_model)
        
        # Performance stats
        if st.session_state.image_processor_page_state.get('performance_stats'):
            self.render_performance_stats()
        
        # Cached processors
        if st.session_state.image_processor_page_state.get('cached_processors'):
            self.render_cached_processors()


def render_image_processors_page():
    """Render function for the image processors page."""
    try:
        page = ImageProcessorsPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering image processors page: {str(e)}")
        error(f"Failed to render image processors page", "image_processors_page", e)


if __name__ == "__main__":
    # For testing
    render_image_processors_page()
