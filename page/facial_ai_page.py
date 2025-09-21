"""
Facial AI Page - Interactive facial feature analysis and AI image generation.

This page provides a user interface for the facial AI pipeline, allowing users to:
- Upload facial images for analysis
- Extract detailed facial features
- Generate AI portraits using various styles and methods
- Download and save generated images
- View facial analysis results and control maps

The page integrates with the existing DurgasAI architecture and provides
a comprehensive interface for facial AI operations.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import json
from PIL import Image
import numpy as np

# Import existing utilities
from utils.config import Config
from utils.logger import debug, info, warning, error, log_user_action, log_session_event

# Import facial AI pipeline components
try:
    from utils.facial_ai_pipeline import FacialAIPipeline, PipelineConfig, create_facial_ai_pipeline
    from utils.facial_feature_extractor import create_facial_extractor
    from utils.generative_model_manager import create_generative_manager
    FACIAL_AI_AVAILABLE = True
except ImportError as e:
    FACIAL_AI_AVAILABLE = False
    warning(f"Facial AI components not available: {e}", "facial_ai_page")


class FacialAIPage:
    """
    Main page class for facial AI operations.
    
    This class provides the complete user interface for facial feature analysis
    and AI image generation within the DurgasAI application.
    """
    
    def __init__(self):
        """Initialize the facial AI page."""
        self.pipeline = None
        self.facial_extractor = None
        self.generative_manager = None
        
        # Initialize components
        self._initialize_components()
        
        info("FacialAIPage initialized", "facial_ai_page")
    
    def _initialize_components(self):
        """Initialize facial AI components."""
        try:
            if FACIAL_AI_AVAILABLE:
                # Initialize pipeline
                config = PipelineConfig()
                self.pipeline = create_facial_ai_pipeline(config)
                
                # Initialize individual components for advanced features
                self.facial_extractor = create_facial_extractor()
                self.generative_manager = create_generative_manager()
                
                if self.pipeline:
                    info("Facial AI pipeline initialized successfully", "facial_ai_page")
                else:
                    warning("Failed to initialize facial AI pipeline", "facial_ai_page")
            else:
                warning("Facial AI components not available", "facial_ai_page")
                
        except Exception as e:
            error(f"Error initializing facial AI components: {str(e)}", "facial_ai_page", e)
    
    def render(self):
        """Render the facial AI page."""
        st.markdown('<h1 class="main-header">🎭 Facial AI Studio</h1>', unsafe_allow_html=True)
        
        if not FACIAL_AI_AVAILABLE:
            self._render_unavailable_message()
            return
        
        if not self.pipeline:
            self._render_setup_required()
            return
        
        # Create tabs for different features
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎨 Generate Portrait", 
            "🔍 Analyze Features", 
            "⚙️ Pipeline Settings", 
            "📊 Status & Metrics"
        ])
        
        with tab1:
            self._render_portrait_generation()
        
        with tab2:
            self._render_feature_analysis()
        
        with tab3:
            self._render_pipeline_settings()
        
        with tab4:
            self._render_status_metrics()
    
    def _render_unavailable_message(self):
        """Render message when facial AI is not available."""
        st.error("🚫 Facial AI Features Not Available")
        
        st.markdown("""
        The facial AI features require additional dependencies that are not currently installed.
        
        **Required Dependencies:**
        - `mediapipe` - For facial landmark detection
        - `diffusers` - For generative AI models
        - `torch` - For deep learning operations
        - `opencv-python` - For image processing
        
        **Installation:**
        ```bash
        pip install mediapipe diffusers torch opencv-python
        ```
        
        After installing the dependencies, restart the application to enable facial AI features.
        """)
        
        # Show installation status
        with st.expander("📋 Dependency Status"):
            self._check_dependencies()
    
    def _render_setup_required(self):
        """Render setup required message."""
        st.warning("⚠️ Facial AI Pipeline Setup Required")
        
        st.markdown("""
        The facial AI pipeline failed to initialize. This could be due to:
        
        - Missing or invalid HuggingFace API token
        - Insufficient system resources
        - Network connectivity issues
        - Model download failures
        
        **Troubleshooting Steps:**
        1. Check your HuggingFace API token in Settings
        2. Ensure stable internet connection
        3. Check system resources (RAM/GPU)
        4. Review the logs for detailed error information
        """)
        
        if st.button("🔄 Retry Initialization"):
            st.rerun()
    
    def _render_portrait_generation(self):
        """Render the portrait generation interface."""
        st.markdown("### 🎨 Generate AI Portrait from Facial Image")
        
        # Image upload section
        uploaded_file = st.file_uploader(
            "Upload a facial image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear photo of a face for AI portrait generation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generation settings
            col1, col2 = st.columns(2)
            
            with col1:
                style_preference = st.selectbox(
                    "Style Preference",
                    ["professional", "artistic", "casual", "fantasy", "vintage"],
                    index=0,
                    help="Choose the artistic style for the generated portrait"
                )
                
                generation_method = st.selectbox(
                    "Generation Method",
                    ["controlnet", "stable_diffusion", "hunyuan_image"],
                    index=0,
                    help="ControlNet: better facial structure control, Stable Diffusion: balanced quality, HunyuanImage: 2K high-resolution"
                )
            
            with col2:
                num_inference_steps = st.slider(
                    "Inference Steps",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="More steps = higher quality but slower generation"
                )
                
                guidance_scale = st.slider(
                    "Guidance Scale",
                    min_value=1.0,
                    max_value=20.0,
                    value=7.5,
                    step=0.5,
                    help="Higher values = better prompt adherence"
                )
            
            # HunyuanImage-specific parameters
            if generation_method == "hunyuan_image":
                st.markdown("**HunyuanImage-2.1 Settings**")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    aspect_ratio = st.selectbox(
                        "Aspect Ratio",
                        ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                        index=0,
                        help="Choose the aspect ratio for the generated image"
                    )
                    
                    use_refiner = st.checkbox(
                        "Use Refiner Model",
                        value=True,
                        help="Enable refiner model for higher quality (slower generation)"
                    )
                
                with col4:
                    use_prompt_enhancement = st.checkbox(
                        "Prompt Enhancement",
                        value=False,
                        help="Automatically enhance prompts for better results"
                    )
                    
                    # Override inference steps for HunyuanImage
                    num_inference_steps = st.slider(
                        "Inference Steps (HunyuanImage)",
                        min_value=8,
                        max_value=50,
                        value=50,
                        help="For HunyuanImage: 8 steps (distilled) or 50 steps (full model)"
                    )
            
            # Generate button
            if st.button("🚀 Generate AI Portrait", type="primary"):
                # Prepare parameters for HunyuanImage
                hunyuan_params = {}
                if generation_method == "hunyuan_image":
                    hunyuan_params = {
                        "aspect_ratio": aspect_ratio,
                        "use_refiner": use_refiner,
                        "use_prompt_enhancement": use_prompt_enhancement
                    }
                
                self._generate_portrait(
                    uploaded_file, 
                    style_preference, 
                    generation_method,
                    num_inference_steps,
                    guidance_scale,
                    **hunyuan_params
                )
        
        # Display generation results
        if "generated_portrait" in st.session_state:
            self._display_generation_results()
    
    def _render_feature_analysis(self):
        """Render the facial feature analysis interface."""
        st.markdown("### 🔍 Facial Feature Analysis")
        
        # Image upload for analysis
        uploaded_file = st.file_uploader(
            "Upload image for facial analysis",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a facial image to extract detailed features",
            key="analysis_upload"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image for Analysis", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Facial Features"):
                self._analyze_facial_features(uploaded_file)
        
        # Display analysis results
        if "facial_analysis" in st.session_state:
            self._display_analysis_results()
    
    def _render_pipeline_settings(self):
        """Render pipeline configuration settings."""
        st.markdown("### ⚙️ Pipeline Configuration")
        
        if not self.pipeline:
            st.error("Pipeline not available")
            return
        
        # Get current config
        current_config = self.pipeline.config
        
        # Configuration form
        with st.form("pipeline_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Generation Settings**")
                
                generation_method = st.selectbox(
                    "Default Generation Method",
                    ["controlnet", "stable_diffusion", "hunyuan_image"],
                    index=0 if current_config.generation_method == "controlnet" else (1 if current_config.generation_method == "stable_diffusion" else 2)
                )
                
                style_preference = st.selectbox(
                    "Default Style Preference",
                    ["professional", "artistic", "casual", "fantasy", "vintage"],
                    index=["professional", "artistic", "casual", "fantasy", "vintage"].index(current_config.style_preference)
                )
                
                num_inference_steps = st.number_input(
                    "Default Inference Steps",
                    min_value=10,
                    max_value=50,
                    value=current_config.num_inference_steps
                )
            
            with col2:
                st.markdown("**Image Processing**")
                
                enhancement_type = st.selectbox(
                    "Enhancement Type",
                    ["basic", "advanced", "none"],
                    index=["basic", "advanced", "none"].index(current_config.enhancement_type)
                )
                
                output_quality = st.slider(
                    "Output Quality",
                    min_value=50,
                    max_value=100,
                    value=current_config.output_quality
                )
                
                save_features = st.checkbox(
                    "Save Facial Features",
                    value=current_config.save_features
                )
            
            # Submit button
            if st.form_submit_button("💾 Update Configuration"):
                # Create new config
                new_config = PipelineConfig(
                    generation_method=generation_method,
                    style_preference=style_preference,
                    num_inference_steps=num_inference_steps,
                    enhancement_type=enhancement_type,
                    output_quality=output_quality,
                    save_features=save_features,
                    device=current_config.device
                )
                
                # Update pipeline config
                self.pipeline.update_config(new_config)
                
                st.success("✅ Configuration updated successfully!")
                log_user_action("pipeline_config_updated", config=new_config.to_dict())
    
    def _render_status_metrics(self):
        """Render pipeline status and performance metrics."""
        st.markdown("### 📊 Pipeline Status & Performance")
        
        if not self.pipeline:
            st.error("Pipeline not available")
            return
        
        # Get pipeline status
        status = self.pipeline.get_pipeline_status()
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Facial Extraction",
                "✅ Available" if status["facial_extraction_available"] else "❌ Unavailable"
            )
        
        with col2:
            st.metric(
                "Generative Models",
                "✅ Available" if status["generative_models_available"] else "❌ Unavailable"
            )
        
        with col3:
            st.metric(
                "Success Rate",
                f"{status['success_rate']:.1f}%"
            )
        
        # Performance metrics
        st.markdown("#### 📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pipelines", status["total_pipelines"])
        
        with col2:
            st.metric("Successful", status["successful_pipelines"])
        
        with col3:
            st.metric("Failed", status["failed_pipelines"])
        
        with col4:
            st.metric("Avg Processing Time", f"{status['average_processing_time']:.2f}s")
        
        # Current configuration
        with st.expander("🔧 Current Configuration"):
            st.json(status["current_config"])
        
        # Component status
        if self.facial_extractor:
            with st.expander("🔍 Facial Extractor Metrics"):
                extractor_metrics = self.facial_extractor.get_performance_metrics()
                st.json(extractor_metrics)
        
        if self.generative_manager:
            with st.expander("🎨 Generative Manager Metrics"):
                generator_metrics = self.generative_manager.get_performance_metrics()
                st.json(generator_metrics)
    
    def _generate_portrait(self, uploaded_file, style_preference, generation_method, 
                          num_inference_steps, guidance_scale, **kwargs):
        """Generate AI portrait from uploaded image."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Show progress
            with st.spinner("🎨 Generating AI portrait..."):
                # Create custom config
                config = PipelineConfig(
                    generation_method=generation_method,
                    style_preference=style_preference,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                # Add HunyuanImage-specific parameters
                if generation_method == "hunyuan_image":
                    config.hunyuan_params = kwargs
                
                # Generate portrait
                result = self.pipeline.generate_portrait(tmp_path, config)
                
                if result.success:
                    # Store result in session state
                    st.session_state.generated_portrait = result
                    
                    # Log successful generation
                    log_user_action("portrait_generated", 
                                  style=style_preference,
                                  method=generation_method,
                                  processing_time=result.processing_time)
                    
                    st.success("✅ Portrait generated successfully!")
                else:
                    st.error(f"❌ Generation failed: {result.error}")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            error(f"Portrait generation failed: {str(e)}", "facial_ai_page", e)
            st.error(f"❌ Generation failed: {str(e)}")
    
    def _analyze_facial_features(self, uploaded_file):
        """Analyze facial features from uploaded image."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Show progress
            with st.spinner("🔍 Analyzing facial features..."):
                # Extract features
                if self.facial_extractor:
                    features = self.facial_extractor.extract_features(tmp_path)
                    
                    if features.get("success", False):
                        # Store result in session state
                        st.session_state.facial_analysis = features
                        
                        # Log successful analysis
                        log_user_action("facial_features_analyzed",
                                      landmark_count=features.get("metadata", {}).get("landmark_count", 0),
                                      processing_time=features.get("metadata", {}).get("processing_time", 0))
                        
                        st.success("✅ Facial features analyzed successfully!")
                    else:
                        st.error(f"❌ Analysis failed: {features.get('error', 'Unknown error')}")
                else:
                    st.error("❌ Facial extractor not available")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            error(f"Facial analysis failed: {str(e)}", "facial_ai_page", e)
            st.error(f"❌ Analysis failed: {str(e)}")
    
    def _display_generation_results(self):
        """Display generation results."""
        result = st.session_state.generated_portrait
        
        st.markdown("### 🎨 Generated Portrait")
        
        # Display generated image
        if result.generated_image:
            st.image(result.generated_image, caption="Generated AI Portrait", use_column_width=True)
            
            # Download button
            if st.button("💾 Download Portrait"):
                self._download_image(result.generated_image, "generated_portrait.jpg")
        
        # Display metadata
        with st.expander("📊 Generation Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Generation Method:**", result.generation_metadata.get("method", "N/A"))
                st.write("**Processing Time:**", f"{result.processing_time:.2f}s")
                st.write("**Device:**", result.generation_metadata.get("device", "N/A"))
            
            with col2:
                st.write("**Inference Steps:**", result.generation_metadata.get("num_inference_steps", "N/A"))
                st.write("**Guidance Scale:**", result.generation_metadata.get("guidance_scale", "N/A"))
                st.write("**Enhancement:**", result.generation_metadata.get("enhancement_type", "N/A"))
            
            # Display prompt if available
            if "prompt" in result.generation_metadata:
                st.write("**Generated Prompt:**")
                st.code(result.generation_metadata["prompt"])
        
        # Display facial features used
        if result.facial_features:
            with st.expander("🔍 Facial Features Used"):
                self._display_facial_features_summary(result.facial_features)
    
    def _display_analysis_results(self):
        """Display facial analysis results."""
        analysis = st.session_state.facial_analysis
        
        st.markdown("### 🔍 Facial Feature Analysis Results")
        
        # Display facial attributes
        if "attributes" in analysis:
            attributes = analysis["attributes"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Face Shape")
                face_shape = attributes.get("face_shape", {})
                st.write("**Shape:**", face_shape.get("face_shape", "N/A"))
                st.write("**Ratio:**", f"{face_shape.get('face_ratio', 0):.2f}")
                st.write("**Width:**", f"{face_shape.get('face_width', 0):.1f}px")
            
            with col2:
                st.markdown("#### 👁️ Eye Features")
                eye_features = attributes.get("eye_features", {})
                st.write("**Shape:**", eye_features.get("eye_shape", "N/A"))
                st.write("**Aspect Ratio:**", f"{eye_features.get('eye_aspect_ratio', 0):.2f}")
                st.write("**Avg Width:**", f"{eye_features.get('avg_eye_width', 0):.1f}px")
            
            with col3:
                st.markdown("#### 👃 Nose & Mouth")
                nose_features = attributes.get("nose_features", {})
                mouth_features = attributes.get("mouth_features", {})
                st.write("**Nose Type:**", nose_features.get("nose_type", "N/A"))
                st.write("**Nose Width:**", f"{nose_features.get('nose_width', 0):.1f}px")
                st.write("**Lip Fullness:**", mouth_features.get("lip_fullness", "N/A"))
        
        # Display control maps
        if "control_maps" in analysis:
            with st.expander("🗺️ Control Maps"):
                control_maps = analysis["control_maps"]
                
                for map_type, control_map in control_maps.items():
                    if isinstance(control_map, np.ndarray):
                        st.write(f"**{map_type.title()} Map:**")
                        st.image(control_map, caption=f"{map_type} control map", use_column_width=True)
        
        # Display metadata
        with st.expander("📊 Analysis Metadata"):
            metadata = analysis.get("metadata", {})
            st.write("**Landmark Count:**", metadata.get("landmark_count", "N/A"))
            st.write("**Processing Time:**", metadata.get("processing_time", "N/A"))
            st.write("**Extraction Method:**", metadata.get("extraction_method", "N/A"))
            st.write("**Face Count:**", metadata.get("face_count", "N/A"))
    
    def _display_facial_features_summary(self, facial_features):
        """Display a summary of facial features."""
        attributes = facial_features.get("attributes", {})
        
        # Face shape
        face_shape = attributes.get("face_shape", {})
        st.write(f"**Face Shape:** {face_shape.get('face_shape', 'N/A')}")
        
        # Eye features
        eye_features = attributes.get("eye_features", {})
        st.write(f"**Eye Shape:** {eye_features.get('eye_shape', 'N/A')}")
        
        # Nose features
        nose_features = attributes.get("nose_features", {})
        st.write(f"**Nose Type:** {nose_features.get('nose_type', 'N/A')}")
        
        # Mouth features
        mouth_features = attributes.get("mouth_features", {})
        st.write(f"**Lip Fullness:** {mouth_features.get('lip_fullness', 'N/A')}")
    
    def _download_image(self, image, filename):
        """Download image to user's device."""
        try:
            # Convert PIL Image to bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            
            # Create download button
            st.download_button(
                label="💾 Download Image",
                data=img_buffer.getvalue(),
                file_name=filename,
                mime="image/jpeg"
            )
            
        except Exception as e:
            error(f"Download failed: {str(e)}", "facial_ai_page", e)
            st.error("❌ Download failed")
    
    def _check_dependencies(self):
        """Check and display dependency status."""
        dependencies = {
            "mediapipe": False,
            "diffusers": False,
            "torch": False,
            "cv2": False
        }
        
        try:
            import mediapipe
            dependencies["mediapipe"] = True
        except ImportError:
            pass
        
        try:
            import diffusers
            dependencies["diffusers"] = True
        except ImportError:
            pass
        
        try:
            import torch
            dependencies["torch"] = True
        except ImportError:
            pass
        
        try:
            import cv2
            dependencies["cv2"] = True
        except ImportError:
            pass
        
        # Display status
        for dep, available in dependencies.items():
            status = "✅" if available else "❌"
            st.write(f"{status} {dep}")


# Page initialization function
def render_facial_ai_page():
    """Render the facial AI page."""
    try:
        page = FacialAIPage()
        page.render()
        
        # Log page view
        log_session_event("facial_ai_page_viewed")
        
    except Exception as e:
        error(f"Error rendering facial AI page: {str(e)}", "facial_ai_page", e)
        st.error("❌ Error loading facial AI page")
