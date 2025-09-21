"""
Backbones Page for DurgasAI.

This module provides a comprehensive interface for backbone management and testing:
- Backbone selection and configuration
- Real-time feature extraction testing
- Multi-layer feature extraction capabilities
- Performance monitoring and statistics
- Advanced backbone features (timm integration, GPU acceleration)
- Integration with enhanced backbone manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.backbone_manager import backbone_manager, BackboneInfo, FeatureExtractionResult
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
    from transformers import AutoBackbone
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class BackbonesPage:
    """
    Comprehensive backbone management and testing interface.
    
    This class provides:
    - Backbone selection and loading
    - Real-time feature extraction testing
    - Multi-layer feature extraction capabilities
    - Performance monitoring
    - Advanced configuration options
    - Integration with enhanced backbone manager
    """
    
    def __init__(self):
        """Initialize the backbones page."""
        debug("Initializing BackbonesPage", "backbones_page")
        self.config = Config()
        
        # Initialize session state for backbones page
        if 'backbones_page_state' not in st.session_state:
            st.session_state.backbones_page_state = {
                'selected_model': 'microsoft/swin-tiny-patch4-window7-224',
                'uploaded_images': [],
                'batch_mode': False,
                'show_advanced': False,
                'performance_stats': None,
                'extracted_features': []
            }
        
        info("BackbonesPage initialized successfully", "backbones_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🏗️ Backbones Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📝 About Backbones</h4>
        <p>Backbones are foundational components in computer vision that extract useful features from input images. 
        This page provides comprehensive tools for:</p>
        <ul>
        <li><strong>Backbone Selection:</strong> Choose from various HuggingFace and timm backbones</li>
        <li><strong>Feature Extraction:</strong> Extract multi-layer features from images</li>
        <li><strong>Multi-Scale Analysis:</strong> Extract features at different resolutions</li>
        <li><strong>Performance Monitoring:</strong> Track backbone performance and caching</li>
        <li><strong>Advanced Features:</strong> timm integration, GPU acceleration, and more</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_model_selection(self):
        """Render backbone model selection interface."""
        st.markdown("### 🤖 Select Backbone Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular backbone models
            popular_models = [
                "microsoft/swin-tiny-patch4-window7-224",
                "microsoft/swin-small-patch4-window7-224",
                "microsoft/swin-base-patch4-window7-224",
                "microsoft/resnet-50",
                "facebook/deit-tiny-patch16-224",
                "google/vit-base-patch16-224"
            ]
            
            selected_model = st.selectbox(
                "Choose a backbone model:",
                options=popular_models,
                index=popular_models.index(st.session_state.backbones_page_state['selected_model']) 
                if st.session_state.backbones_page_state['selected_model'] in popular_models else 0,
                help="Select a HuggingFace backbone model to work with"
            )
            
            # Custom model input
            custom_model = st.text_input(
                "Or enter custom model ID:",
                placeholder="e.g., microsoft/swin-base-patch4-window7-224",
                help="Enter any HuggingFace model ID for custom backbone"
            )
            
            if custom_model:
                selected_model = custom_model
        
        with col2:
            # Model info display
            st.markdown("**Model Information:**")
            if selected_model:
                model_info = self.get_model_info(selected_model)
                if model_info:
                    st.write(f"**Type:** {model_info.get('backbone_type', 'Unknown')}")
                    st.write(f"**timm Support:** {model_info.get('supports_timm', False)}")
                    st.write(f"**GPU Support:** {model_info.get('supports_gpu', False)}")
                    st.write(f"**Compilation:** {model_info.get('supports_compilation', False)}")
                else:
                    st.info("Click 'Load Backbone' to get model information")
        
        # Update session state
        st.session_state.backbones_page_state['selected_model'] = selected_model
        
        return selected_model
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get backbone model information."""
        try:
            backbone_info = backbone_manager.get_backbone_info(model_id)
            if backbone_info:
                return backbone_info
        except Exception as e:
            debug(f"Failed to get model info: {e}", "backbones_page")
        
        return None
    
    def render_processor_controls(self, model_id: str):
        """Render backbone loading and control buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Load Backbone", type="primary"):
                with st.spinner(f"Loading backbone for {model_id}..."):
                    start_time = time.time()
                    backbone_info = backbone_manager.load_backbone(model_id)
                    load_time = time.time() - start_time
                    
                    if backbone_info:
                        st.success(f"✅ Backbone loaded in {load_time:.2f}s")
                        log_user_action("backbone_loaded", model_id=model_id, load_time=load_time)
                    else:
                        st.error("❌ Failed to load backbone")
                        log_user_action("backbone_load_failed", model_id=model_id)
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = backbone_manager.get_performance_stats()
                st.session_state.backbones_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                backbone_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("backbone_cache_cleared")
        
        with col4:
            if st.button("📋 List Cached"):
                cached = backbone_manager.list_cached_backbones()
                st.session_state.backbones_page_state['cached_backbones'] = cached
    
    def render_layer_selection(self):
        """Render layer selection interface."""
        st.markdown("### 🎯 Layer Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Layer selection method
            selection_method = st.radio(
                "Selection method:",
                ["Use Indices", "Use Features", "Auto Select"],
                horizontal=True
            )
            
            if selection_method == "Use Indices":
                # Manual index selection
                num_layers = st.number_input("Number of layers:", min_value=1, max_value=8, value=4)
                selected_indices = []
                
                for i in range(num_layers):
                    index = st.number_input(f"Layer {i+1} index:", min_value=0, max_value=7, value=i)
                    selected_indices.append(index)
                
                st.session_state.backbones_page_state['out_indices'] = tuple(selected_indices)
                st.session_state.backbones_page_state['out_features'] = None
                
            elif selection_method == "Use Features":
                # Feature name selection
                feature_names = st.text_area(
                    "Feature names (one per line):",
                    value="stage1\nstage2\nstage3\nstage4",
                    help="Enter feature names for each stage"
                )
                features = [name.strip() for name in feature_names.split('\n') if name.strip()]
                st.session_state.backbones_page_state['out_features'] = features
                st.session_state.backbones_page_state['out_indices'] = None
                
            else:  # Auto Select
                st.info("Automatically select all available layers")
                st.session_state.backbones_page_state['out_indices'] = None
                st.session_state.backbones_page_state['out_features'] = None
        
        with col2:
            # Display current selection
            st.markdown("**Current Selection:**")
            if st.session_state.backbones_page_state.get('out_indices'):
                st.write(f"Indices: {st.session_state.backbones_page_state['out_indices']}")
            elif st.session_state.backbones_page_state.get('out_features'):
                st.write(f"Features: {st.session_state.backbones_page_state['out_features']}")
            else:
                st.write("Auto selection (all layers)")
    
    def render_image_input(self):
        """Render image input interface."""
        st.markdown("### 🖼️ Image Input")
        
        # Mode selection
        col1, col2 = st.columns([1, 3])
        with col1:
            batch_mode = st.checkbox(
                "Batch Mode", 
                value=st.session_state.backbones_page_state['batch_mode'],
                help="Process multiple images at once for better performance"
            )
            st.session_state.backbones_page_state['batch_mode'] = batch_mode
        
        with col2:
            if batch_mode:
                st.info("💡 Batch mode processes multiple images efficiently using optimized processing")
        
        # Image input method
        input_method = st.radio(
            "Choose input method:",
            ["Generate Sample Images", "Use Synthetic Data", "Upload Images (Coming Soon)"],
            horizontal=True
        )
        
        if input_method == "Generate Sample Images":
            self.render_sample_generation()
        elif input_method == "Use Synthetic Data":
            self.render_synthetic_data()
        elif input_method == "Upload Images (Coming Soon)":
            st.info("🖼️ Image upload functionality will be available in future updates")
    
    def render_sample_generation(self):
        """Render sample image generation interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            num_images = st.number_input("Number of sample images:", min_value=1, max_value=8, value=3)
            image_size = st.selectbox("Image size:", [(224, 224), (256, 256), (512, 512), (64, 64)], index=0)
        
        with col2:
            color_mode = st.selectbox("Color mode:", ["RGB", "Grayscale"])
            pattern = st.selectbox("Pattern:", ["Random", "Gradient", "Solid Color", "Checkerboard"])
        
        if st.button("🎨 Generate Sample Images"):
            if PIL_AVAILABLE:
                images = self.generate_sample_images(num_images, image_size, color_mode, pattern)
                st.session_state.backbones_page_state['uploaded_images'] = images
                
                # Display generated images info
                st.markdown("**Generated Sample Images:**")
                for i, image in enumerate(images):
                    st.write(f"Image {i+1}: Size {image.size}, Mode: {image.mode}")
            else:
                st.error("PIL not available for image generation")
    
    def generate_sample_images(self, num_images: int, size: tuple, color_mode: str, pattern: str) -> List[Image.Image]:
        """Generate sample images for testing."""
        images = []
        
        for i in range(num_images):
            if pattern == "Random":
                if color_mode == 'RGB':
                    # Create random RGB image
                    data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                    image = Image.fromarray(data)
                else:
                    # Create random grayscale image
                    data = np.random.randint(0, 256, (size[1], size[0]), dtype=np.uint8)
                    image = Image.fromarray(data, mode='L')
            
            elif pattern == "Gradient":
                if color_mode == 'RGB':
                    # Create gradient image
                    data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    for y in range(size[1]):
                        for x in range(size[0]):
                            r = int((x / size[0]) * 255)
                            g = int((y / size[1]) * 255)
                            b = int(((x + y) / (size[0] + size[1])) * 255)
                            data[y, x] = [r, g, b]
                    image = Image.fromarray(data)
                else:
                    # Create grayscale gradient
                    data = np.zeros((size[1], size[0]), dtype=np.uint8)
                    for y in range(size[1]):
                        for x in range(size[0]):
                            gray = int(((x + y) / (size[0] + size[1])) * 255)
                            data[y, x] = gray
                    image = Image.fromarray(data, mode='L')
            
            elif pattern == "Solid Color":
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                color = colors[i % len(colors)]
                if color_mode == 'RGB':
                    image = Image.new('RGB', size, color)
                else:
                    image = Image.new('L', size, color[0])
            
            elif pattern == "Checkerboard":
                if color_mode == 'RGB':
                    image = Image.new('RGB', size, (255, 255, 255))
                    # Create checkerboard pattern
                    for y in range(0, size[1], 32):
                        for x in range(0, size[0], 32):
                            if ((x // 32) + (y // 32)) % 2 == 0:
                                # Draw black squares
                                for dy in range(min(32, size[1] - y)):
                                    for dx in range(min(32, size[0] - x)):
                                        image.putpixel((x + dx, y + dy), (0, 0, 0))
                else:
                    image = Image.new('L', size, 255)
                    # Create checkerboard pattern
                    for y in range(0, size[1], 32):
                        for x in range(0, size[0], 32):
                            if ((x // 32) + (y // 32)) % 2 == 0:
                                # Draw black squares
                                for dy in range(min(32, size[1] - y)):
                                    for dx in range(min(32, size[0] - x)):
                                        image.putpixel((x + dx, y + dy), 0)
            
            images.append(image)
        
        return images
    
    def render_synthetic_data(self):
        """Render synthetic data interface."""
        st.markdown("**Generate Synthetic Image Data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input("Batch size:", min_value=1, max_value=16, value=4)
            height = st.number_input("Height:", min_value=64, max_value=1024, value=224)
        
        with col2:
            width = st.number_input("Width:", min_value=64, max_value=1024, value=224)
            channels = st.selectbox("Channels:", [1, 3], index=1)
        
        if st.button("🎲 Generate Synthetic Data"):
            if TORCH_AVAILABLE:
                synthetic_image = torch.randn(batch_size, channels, height, width)
                st.session_state.backbones_page_state['uploaded_images'] = [synthetic_image]
                
                st.success(f"✅ Generated synthetic image data: {synthetic_image.shape}")
                st.write(f"📊 Batch size: {batch_size}")
                st.write(f"📐 Resolution: {height}x{width}")
                st.write(f"🎨 Channels: {channels}")
            else:
                st.error("PyTorch not available for synthetic data generation")
    
    def render_advanced_options(self):
        """Render advanced backbone processing options."""
        with st.expander("⚙️ Advanced Processing Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                use_timm_backbone = st.checkbox("Use timm Backbone", value=False, help="Use timm library backbone")
                use_pretrained_backbone = st.checkbox("Use Pretrained Weights", value=True, help="Use pretrained or randomly initialized weights")
                compile_backbone = st.checkbox("Compile Backbone", value=False, help="Compile backbone for maximum performance")
                device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help="Processing device")
            
            with col2:
                return_tensors = st.selectbox("Return Tensors", ["pt", "tf", "np", "None"], index=0, help="Tensor format for output")
                normalize = st.checkbox("Normalize", value=True, help="Normalize pixel values")
                resize = st.checkbox("Resize", value=True, help="Resize images to model requirements")
                
            # Store options
            st.session_state.backbones_page_state['advanced_options'] = {
                'use_timm_backbone': use_timm_backbone,
                'use_pretrained_backbone': use_pretrained_backbone,
                'return_tensors': return_tensors if return_tensors != "None" else None,
                'device': device if device != "auto" else None,
                'compile_backbone': compile_backbone,
                'normalize': normalize,
                'resize': resize
            }
    
    def render_processing_interface(self, model_id: str):
        """Render feature extraction interface."""
        st.markdown("### 🔄 Feature Extraction")
        
        # Advanced options
        self.render_advanced_options()
        
        # Process button
        if st.button("🚀 Extract Features", type="primary"):
            self.extract_features(model_id)
    
    def extract_features(self, model_id: str):
        """Extract features from images with the selected options."""
        images = st.session_state.backbones_page_state.get('uploaded_images', [])
        
        if not images:
            st.error("No images provided for feature extraction")
            return
        
        try:
            # Get options
            options = st.session_state.backbones_page_state.get('advanced_options', {})
            compile_backbone = options.pop('compile_backbone', False)
            
            # Add layer selection options
            if st.session_state.backbones_page_state.get('out_indices'):
                options['out_indices'] = st.session_state.backbones_page_state['out_indices']
            elif st.session_state.backbones_page_state.get('out_features'):
                options['out_features'] = st.session_state.backbones_page_state['out_features']
            
            # Extract features
            with st.spinner("Extracting features..."):
                start_time = time.time()
                result = backbone_manager.extract_features(
                    images, model_id, compile_backbone=compile_backbone, **options
                )
                extraction_time = time.time() - start_time
            
            if result and result.success:
                self.display_results(result, extraction_time)
                log_user_action("features_extracted", model_id=model_id, num_images=len(images))
            else:
                st.error(f"Feature extraction failed: {result.error if result else 'Unknown error'}")
                
        except Exception as e:
            st.error(f"Feature extraction failed: {str(e)}")
            error(f"Feature extraction failed", "backbones_page", e)
    
    def display_results(self, result: FeatureExtractionResult, extraction_time: float):
        """Display feature extraction results."""
        st.markdown("### 📊 Feature Extraction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Extraction Time", f"{extraction_time:.3f}s")
            st.metric("Input Shape", str(result.input_shape))
            st.metric("Feature Maps", len(result.feature_maps))
        
        with col2:
            if result.feature_maps:
                total_features = sum(fm.numel() for fm in result.feature_maps)
                st.metric("Total Features", f"{total_features:,}")
                st.metric("Avg Features per Map", f"{total_features // len(result.feature_maps):,}")
            
            if result.backbone_info:
                st.metric("Layers Used", len(result.backbone_info.out_indices))
        
        with col3:
            if result.feature_maps:
                shapes = [fm.shape for fm in result.feature_maps]
                st.metric("Feature Shapes", f"{len(shapes)} maps")
                st.metric("First Map Shape", str(shapes[0]) if shapes else "N/A")
                st.metric("Last Map Shape", str(shapes[-1]) if shapes else "N/A")
        
        # Detailed results
        if result.success:
            st.markdown("**📝 Detailed Results:**")
            
            # Backbone info
            if result.backbone_info:
                with st.expander("🔧 Backbone Information"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Model:** {result.backbone_info.model_id}")
                        st.write(f"**Type:** {result.backbone_info.backbone_type}")
                        st.write(f"**timm:** {result.backbone_info.is_timm}")
                        st.write(f"**GPU Optimized:** {result.backbone_info.gpu_optimized}")
                    
                    with col2:
                        st.write(f"**Out Indices:** {result.backbone_info.out_indices}")
                        st.write(f"**Num Channels:** {result.backbone_info.num_channels}")
                        st.write(f"**Image Size:** {result.backbone_info.image_size}")
                        st.write(f"**Embed Dim:** {result.backbone_info.embed_dim}")
            
            # Feature maps info
            if result.feature_maps:
                with st.expander("🗺️ Feature Maps Information"):
                    for i, feature_map in enumerate(result.feature_maps):
                        st.write(f"**Feature Map {i+1}:**")
                        st.write(f"   Shape: {feature_map.shape}")
                        st.write(f"   Data Type: {feature_map.dtype}")
                        st.write(f"   Device: {feature_map.device}")
                        st.write(f"   Min: {feature_map.min():.3f}")
                        st.write(f"   Max: {feature_map.max():.3f}")
                        st.write(f"   Mean: {feature_map.mean():.3f}")
                        st.write("")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.backbones_page_state.get('performance_stats')
        
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
                st.metric("Total Extractions", stats['total_extractions'])
                st.metric("Batch Extractions", stats['batch_extractions'])
                st.metric("GPU Extractions", stats['gpu_extractions'])
                st.metric("timm Backbones", stats['timm_backbones'])
    
    def render_cached_backbones(self):
        """Render cached backbones information."""
        cached = st.session_state.backbones_page_state.get('cached_backbones', [])
        
        if cached:
            st.markdown("### 💾 Cached Backbones")
            
            for backbone_info in cached:
                with st.expander(f"🤖 {backbone_info['model_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {backbone_info['backbone_type']}")
                        st.write(f"**timm:** {backbone_info.get('is_timm', False)}")
                        st.write(f"**GPU Support:** {backbone_info.get('supports_gpu', False)}")
                    
                    with col2:
                        st.write(f"**Load Time:** {backbone_info['load_time']:.3f}s")
                        st.write(f"**Out Indices:** {backbone_info['out_indices']}")
                        st.write(f"**Cache Key:** {backbone_info['cache_key'][:16]}...")
    
    def render(self):
        """Render the complete backbones page."""
        self.render_header()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Backbone controls
        self.render_processor_controls(selected_model)
        
        # Layer selection
        self.render_layer_selection()
        
        # Image input
        self.render_image_input()
        
        # Processing interface
        if st.session_state.backbones_page_state.get('uploaded_images'):
            self.render_processing_interface(selected_model)
        
        # Performance stats
        if st.session_state.backbones_page_state.get('performance_stats'):
            self.render_performance_stats()
        
        # Cached backbones
        if st.session_state.backbones_page_state.get('cached_backbones'):
            self.render_cached_backbones()


def render_backbones_page():
    """Render function for the backbones page."""
    try:
        page = BackbonesPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering backbones page: {str(e)}")
        error(f"Failed to render backbones page", "backbones_page", e)


if __name__ == "__main__":
    # For testing
    render_backbones_page()
