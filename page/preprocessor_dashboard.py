"""
Preprocessor Dashboard for DurgasAI.

This module provides a unified dashboard for managing both tokenizers and image processors:
- Combined performance monitoring
- Unified cache management
- Cross-modal processing capabilities
- Integration testing and validation
- Comprehensive analytics and insights
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

from utils.tokenizer_manager import tokenizer_manager
from utils.image_processor_manager import image_processor_manager
from utils.video_processor_manager import video_processor_manager
from utils.backbone_manager import backbone_manager
from utils.feature_extractor_manager import feature_extractor_manager
from utils.processor_manager import processor_manager
from utils.tokenizer_summary_manager import tokenizer_summary_manager
from utils.padding_truncation_manager import padding_truncation_manager
from utils.pipeline_manager import pipeline_manager
from utils.ml_apps_manager import ml_apps_manager
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class PreprocessorDashboard:
    """
    Unified dashboard for tokenizer and image processor management.
    
    This class provides:
    - Combined performance monitoring
    - Unified cache management
    - Cross-modal processing capabilities
    - Integration testing and validation
    - Comprehensive analytics and insights
    """
    
    def __init__(self):
        """Initialize the preprocessor dashboard."""
        debug("Initializing PreprocessorDashboard", "preprocessor_dashboard")
        self.config = Config()
        
        # Initialize session state
        if 'preprocessor_dashboard_state' not in st.session_state:
            st.session_state.preprocessor_dashboard_state = {
                'last_refresh': time.time(),
                'combined_stats': None,
                'integration_tests': []
            }
        
        info("PreprocessorDashboard initialized successfully", "preprocessor_dashboard")
    
    def render_header(self):
        """Render the dashboard header."""
        st.markdown('<h1 class="main-header">🔧 Preprocessor Dashboard</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Unified Preprocessing Management</h4>
        <p>This dashboard provides comprehensive management and monitoring for tokenizers, image processors, video processors, backbones, feature extractors, processors, tokenizer summaries, padding/truncation, pipelines, and ML apps:</p>
        <ul>
        <li><strong>Performance Analytics:</strong> Combined metrics and insights across all processors</li>
        <li><strong>Cache Management:</strong> Unified cache operations and monitoring</li>
        <li><strong>Integration Testing:</strong> Cross-modal processing validation</li>
        <li><strong>System Health:</strong> Overall preprocessing system status</li>
        <li><strong>Multimodal Support:</strong> Complete text, image, video, backbone, audio, and unified processing capabilities</li>
        <li><strong>Feature Extraction:</strong> Advanced backbone feature extraction and analysis</li>
        <li><strong>Audio Processing:</strong> Comprehensive audio feature extraction and preprocessing</li>
        <li><strong>Unified Processing:</strong> Multimodal processor coordination and optimization</li>
        <li><strong>Tokenizer Analysis:</strong> Advanced tokenizer algorithm analysis and comparison</li>
        <li><strong>Padding & Truncation:</strong> Advanced sequence length management and optimization</li>
        <li><strong>Pipeline Management:</strong> Complete ML pipeline orchestration and optimization</li>
        <li><strong>ML Apps Management:</strong> Complete ML app creation, deployment, and management</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_overview_metrics(self):
        """Render overview metrics for all ten systems."""
        st.markdown("### 📊 System Overview")
        
        # Get stats from all managers
        tokenizer_stats = tokenizer_manager.get_performance_stats()
        image_processor_stats = image_processor_manager.get_performance_stats()
        video_processor_stats = video_processor_manager.get_performance_stats()
        backbone_stats = backbone_manager.get_performance_stats()
        feature_extractor_stats = feature_extractor_manager.get_performance_stats()
        processor_stats = processor_manager.get_performance_stats()
        tokenizer_summary_stats = tokenizer_summary_manager.get_performance_stats()
        padding_truncation_stats = padding_truncation_manager.get_performance_stats()
        pipeline_stats = pipeline_manager.get_performance_stats()
        ml_apps_stats = ml_apps_manager.get_performance_stats()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col6, col7, col8, col9, col10 = st.columns(5)
        
        with col1:
            st.metric(
                "Tokenizers Cached",
                tokenizer_stats['memory_cache_size'],
                help="Number of tokenizers in memory cache"
            )
        
        with col2:
            st.metric(
                "Image Processors Cached",
                image_processor_stats['memory_cache_size'],
                help="Number of image processors in memory cache"
            )
        
        with col3:
            st.metric(
                "Video Processors Cached",
                video_processor_stats['memory_cache_size'],
                help="Number of video processors in memory cache"
            )
        
        with col4:
            st.metric(
                "Backbones Cached",
                backbone_stats['memory_cache_size'],
                help="Number of backbones in memory cache"
            )
        
        with col5:
            st.metric(
                "Feature Extractors Cached",
                feature_extractor_stats['memory_cache_size'],
                help="Number of feature extractors in memory cache"
            )
        
        with col6:
            st.metric(
                "Processors Cached",
                processor_stats['memory_cache_size'],
                help="Number of processors in memory cache"
            )
        
        with col7:
            st.metric(
                "Tokenizer Summaries Cached",
                tokenizer_summary_stats['memory_cache_size'],
                help="Number of tokenizer summaries in memory cache"
            )
        
        with col8:
            st.metric(
                "Padding/Truncation Configs Cached",
                padding_truncation_stats['memory_cache_size'],
                help="Number of padding/truncation configurations in memory cache"
            )
        
        with col9:
            st.metric(
                "Pipelines Cached",
                pipeline_stats['memory_cache_size'],
                help="Number of pipelines in memory cache"
            )
        
        with col10:
            st.metric(
                "ML Apps Cached",
                ml_apps_stats['memory_cache_size'],
                help="Number of ML apps in memory cache"
            )
        
        # Add a new row for additional metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_operations = (tokenizer_stats['total_loads'] + 
                              image_processor_stats['total_loads'] + 
                              video_processor_stats['total_loads'] +
                              backbone_stats['total_loads'] +
                              feature_extractor_stats['total_loads'] +
                              processor_stats['total_loads'] +
                              tokenizer_summary_stats['total_loads'] +
                              padding_truncation_stats['total_operations'] +
                              pipeline_stats['total_executions'] +
                              ml_apps_stats['total_app_creations'])
            st.metric(
                "Total Operations",
                total_operations,
                help="Combined total operations for all systems"
            )
        
        # Add a new row for additional metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_cache_hit = ((tokenizer_stats['cache_hit_rate'] + 
                            image_processor_stats['cache_hit_rate'] + 
                            video_processor_stats['cache_hit_rate'] +
                            backbone_stats['cache_hit_rate'] +
                            feature_extractor_stats['cache_hit_rate'] +
                            processor_stats['cache_hit_rate'] +
                            tokenizer_summary_stats['cache_hit_rate']) / 7)
            st.metric(
                "Avg Cache Hit Rate",
                f"{avg_cache_hit:.1f}%",
                help="Average cache hit rate across all systems"
            )
        
        with col2:
            total_memory_usage = (tokenizer_stats.get('memory_usage_mb', 0) +
                                image_processor_stats.get('memory_usage_mb', 0) +
                                video_processor_stats.get('memory_usage_mb', 0) +
                                backbone_stats.get('memory_usage_mb', 0) +
                                feature_extractor_stats.get('memory_usage_mb', 0) +
                                processor_stats.get('memory_usage_mb', 0) +
                                tokenizer_summary_stats.get('memory_usage_mb', 0) +
                                padding_truncation_stats.get('memory_usage_mb', 0) +
                                pipeline_stats.get('memory_usage', 0) +
                                ml_apps_stats.get('memory_usage', 0))
            st.metric(
                "Total Memory Usage",
                f"{total_memory_usage:.1f} MB",
                help="Combined memory usage across all systems"
            )
        
        with col3:
            total_errors = (tokenizer_stats['error_count'] +
                          image_processor_stats['error_count'] +
                          video_processor_stats['error_count'] +
                          backbone_stats['error_count'] +
                          feature_extractor_stats['error_count'] +
                          processor_stats['error_count'] +
                          tokenizer_summary_stats['error_count'] +
                          padding_truncation_stats['error_count'] +
                          pipeline_stats['error_count'] +
                          ml_apps_stats['error_count'])
            st.metric(
                "Total Errors",
                total_errors,
                help="Combined error count across all systems"
            )
        
        with col4:
            avg_processing_time = ((tokenizer_stats.get('average_load_time', 0) +
                                  image_processor_stats.get('average_load_time', 0) +
                                  video_processor_stats.get('average_load_time', 0) +
                                  backbone_stats.get('average_load_time', 0) +
                                  feature_extractor_stats.get('average_load_time', 0) +
                                  processor_stats.get('average_load_time', 0) +
                                  tokenizer_summary_stats.get('average_load_time', 0) +
                                  padding_truncation_stats.get('average_processing_time', 0) +
                                  pipeline_stats.get('average_execution_time', 0) +
                                  ml_apps_stats.get('average_creation_time', 0)) / 10)
            st.metric(
                "Avg Processing Time",
                f"{avg_processing_time:.3f}s",
                help="Average processing time across all systems"
            )
        
        with col5:
            total_optimizations = (tokenizer_stats.get('memory_optimizations', 0) +
                                 image_processor_stats.get('memory_optimizations', 0) +
                                 video_processor_stats.get('memory_optimizations', 0) +
                                 backbone_stats.get('memory_optimizations', 0) +
                                 feature_extractor_stats.get('memory_optimizations', 0) +
                                 processor_stats.get('memory_optimizations', 0) +
                                 tokenizer_summary_stats.get('memory_optimizations', 0) +
                                 padding_truncation_stats.get('memory_optimizations', 0) +
                                 pipeline_stats.get('memory_optimizations', 0) +
                                 ml_apps_stats.get('memory_optimizations', 0))
            st.metric(
                "Total Optimizations",
                total_optimizations,
                help="Combined optimization count across all systems"
            )
    
    def render_performance_comparison(self):
        """Render performance comparison charts."""
        st.markdown("### 📈 Performance Comparison")
        
        # Get stats
        tokenizer_stats = tokenizer_manager.get_performance_stats()
        image_processor_stats = image_processor_manager.get_performance_stats()
        video_processor_stats = video_processor_manager.get_performance_stats()
        backbone_stats = backbone_manager.get_performance_stats()
        feature_extractor_stats = feature_extractor_manager.get_performance_stats()
        processor_stats = processor_manager.get_performance_stats()
        tokenizer_summary_stats = tokenizer_summary_manager.get_performance_stats()
        padding_truncation_stats = padding_truncation_manager.get_performance_stats()
        pipeline_stats = pipeline_manager.get_performance_stats()
        ml_apps_stats = ml_apps_manager.get_performance_stats()
        
        # Create comparison data
        comparison_data = {
            'Metric': [
                'Cache Hit Rate (%)',
                'Total Loads/Operations',
                'Cache Hits',
                'Cache Misses',
                'Avg Load Time (s)',
                'Memory Cache Size',
                'Disk Cache Size',
                'Total Processings',
                'GPU Processings'
            ],
            'Tokenizers': [
                tokenizer_stats['cache_hit_rate'],
                tokenizer_stats['total_loads'],
                tokenizer_stats['cache_hits'],
                tokenizer_stats['cache_misses'],
                tokenizer_stats['average_load_time'],
                tokenizer_stats['memory_cache_size'],
                tokenizer_stats['disk_cache_size'],
                tokenizer_stats.get('total_tokenizations', 0),
                tokenizer_stats.get('gpu_tokenizations', 0)
            ],
            'Image Processors': [
                image_processor_stats['cache_hit_rate'],
                image_processor_stats['total_loads'],
                image_processor_stats['cache_hits'],
                image_processor_stats['cache_misses'],
                image_processor_stats['average_load_time'],
                image_processor_stats['memory_cache_size'],
                image_processor_stats['disk_cache_size'],
                image_processor_stats['total_processings'],
                image_processor_stats['gpu_processings']
            ],
            'Video Processors': [
                video_processor_stats['cache_hit_rate'],
                video_processor_stats['total_loads'],
                video_processor_stats['cache_hits'],
                video_processor_stats['cache_misses'],
                video_processor_stats['average_load_time'],
                video_processor_stats['memory_cache_size'],
                video_processor_stats['disk_cache_size'],
                video_processor_stats['total_processings'],
                video_processor_stats['gpu_processings']
            ],
            'Backbones': [
                backbone_stats['cache_hit_rate'],
                backbone_stats['total_loads'],
                backbone_stats['cache_hits'],
                backbone_stats['cache_misses'],
                backbone_stats['average_load_time'],
                backbone_stats['memory_cache_size'],
                backbone_stats['disk_cache_size'],
                backbone_stats['total_extractions'],
                backbone_stats['gpu_extractions']
            ],
            'Feature Extractors': [
                feature_extractor_stats['cache_hit_rate'],
                feature_extractor_stats['total_loads'],
                feature_extractor_stats['cache_hits'],
                feature_extractor_stats['cache_misses'],
                feature_extractor_stats['average_load_time'],
                feature_extractor_stats['memory_cache_size'],
                feature_extractor_stats['disk_cache_size'],
                feature_extractor_stats['total_processings'],
                feature_extractor_stats.get('gpu_processings', 0)
            ],
            'Processors': [
                processor_stats['cache_hit_rate'],
                processor_stats['total_loads'],
                processor_stats['cache_hits'],
                processor_stats['cache_misses'],
                processor_stats['average_load_time'],
                processor_stats['memory_cache_size'],
                processor_stats['disk_cache_size'],
                processor_stats['total_processings'],
                processor_stats.get('gpu_processings', 0)
            ],
            'Tokenizer Summaries': [
                tokenizer_summary_stats['cache_hit_rate'],
                tokenizer_summary_stats['total_loads'],
                tokenizer_summary_stats['cache_hits'],
                tokenizer_summary_stats['cache_misses'],
                tokenizer_summary_stats['average_load_time'],
                tokenizer_summary_stats['memory_cache_size'],
                tokenizer_summary_stats['disk_cache_size'],
                tokenizer_summary_stats['total_analyses'],
                tokenizer_summary_stats.get('gpu_analyses', 0)
            ],
            'Padding/Truncation': [
                0.0,  # No cache hit rate for padding/truncation
                padding_truncation_stats['total_operations'],
                0,    # No cache hits
                0,    # No cache misses
                padding_truncation_stats['average_processing_time'],
                padding_truncation_stats['memory_cache_size'],
                padding_truncation_stats['disk_cache_size'],
                padding_truncation_stats['total_tokens_processed'],
                0     # No GPU processing
            ],
            'Pipelines': [
                pipeline_stats['cache_hit_rate'],
                pipeline_stats['total_executions'],
                pipeline_stats.get('cache_hits', 0),
                pipeline_stats.get('cache_misses', 0),
                pipeline_stats['average_execution_time'],
                pipeline_stats['memory_cache_size'],
                pipeline_stats['disk_cache_size'],
                pipeline_stats['total_inputs_processed'],
                0     # No GPU processing tracking
            ],
            'ML Apps': [
                ml_apps_stats['cache_hit_rate'],
                ml_apps_stats['total_app_creations'],
                ml_apps_stats.get('cache_hits', 0),
                ml_apps_stats.get('cache_misses', 0),
                ml_apps_stats['average_creation_time'],
                ml_apps_stats['memory_cache_size'],
                ml_apps_stats['disk_cache_size'],
                ml_apps_stats['total_deployments'],
                0     # No GPU processing tracking
            ]
        }
        
        # Display comparison table
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cache Performance:**")
            cache_data = pd.DataFrame({
                'System': ['Tokenizers', 'Image Processors'],
                'Cache Hit Rate (%)': [tokenizer_stats['cache_hit_rate'], image_processor_stats['cache_hit_rate']],
                'Cache Hits': [tokenizer_stats['cache_hits'], image_processor_stats['cache_hits']]
            })
            st.bar_chart(cache_data.set_index('System')[['Cache Hit Rate (%)', 'Cache Hits']])
        
        with col2:
            st.markdown("**Load Performance:**")
            load_data = pd.DataFrame({
                'System': ['Tokenizers', 'Image Processors'],
                'Total Loads': [tokenizer_stats['total_loads'], image_processor_stats['total_loads']],
                'Avg Load Time (s)': [tokenizer_stats['average_load_time'], image_processor_stats['average_load_time']]
            })
            st.bar_chart(load_data.set_index('System')[['Total Loads', 'Avg Load Time (s)']])
    
    def render_cache_management(self):
        """Render unified cache management interface."""
        st.markdown("### 💾 Cache Management")
        
        # Create two rows for better layout
        st.markdown("#### Individual Cache Management")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        with col1:
            st.markdown("**Tokenizer Cache:**")
            tokenizer_cached = tokenizer_manager.list_cached_tokenizers()
            st.write(f"Cached tokenizers: {len(tokenizer_cached)}")
            
            if st.button("🗑️ Clear Tokenizer Cache"):
                tokenizer_manager.clear_cache()
                st.success("Tokenizer cache cleared")
                log_user_action("tokenizer_cache_cleared_from_dashboard")
        
        with col2:
            st.markdown("**Image Processor Cache:**")
            image_processor_cached = image_processor_manager.list_cached_processors()
            st.write(f"Cached processors: {len(image_processor_cached)}")
            
            if st.button("🗑️ Clear Image Processor Cache"):
                image_processor_manager.clear_cache()
                st.success("Image processor cache cleared")
                log_user_action("image_processor_cache_cleared_from_dashboard")
        
        with col3:
            st.markdown("**Video Processor Cache:**")
            video_processor_cached = video_processor_manager.list_cached_processors()
            st.write(f"Cached processors: {len(video_processor_cached)}")
            
            if st.button("🗑️ Clear Video Processor Cache"):
                video_processor_manager.clear_cache()
                st.success("Video processor cache cleared")
                log_user_action("video_processor_cache_cleared_from_dashboard")
        
        with col4:
            st.markdown("**Backbone Cache:**")
            backbone_cached = backbone_manager.list_cached_backbones()
            st.write(f"Cached backbones: {len(backbone_cached)}")
            
            if st.button("🗑️ Clear Backbone Cache"):
                backbone_manager.clear_cache()
                st.success("Backbone cache cleared")
                log_user_action("backbone_cache_cleared_from_dashboard")
        
        with col5:
            st.markdown("**Feature Extractor Cache:**")
            feature_extractor_cached = feature_extractor_manager.list_cached_extractors()
            st.write(f"Cached extractors: {len(feature_extractor_cached)}")
            
            if st.button("🗑️ Clear Feature Extractor Cache"):
                feature_extractor_manager.clear_cache()
                st.success("Feature extractor cache cleared")
                log_user_action("feature_extractor_cache_cleared_from_dashboard")
        
        with col6:
            st.markdown("**Processor Cache:**")
            processor_cached = processor_manager.list_cached_processors()
            st.write(f"Cached processors: {len(processor_cached)}")
            
            if st.button("🗑️ Clear Processor Cache"):
                processor_manager.clear_cache()
                st.success("Processor cache cleared")
                log_user_action("processor_cache_cleared_from_dashboard")
        
        # Add tokenizer summary and padding/truncation cache in a new row
        st.markdown("#### Additional Cache Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Tokenizer Summary Cache:**")
            tokenizer_summary_cached = tokenizer_summary_manager.list_cached_summaries()
            st.write(f"Cached summaries: {len(tokenizer_summary_cached)}")
            
            if st.button("🗑️ Clear Tokenizer Summary Cache"):
                tokenizer_summary_manager.clear_cache()
                st.success("Tokenizer summary cache cleared")
                log_user_action("tokenizer_summary_cache_cleared_from_dashboard")
        
        with col2:
            st.markdown("**Padding/Truncation Cache:**")
            padding_truncation_cached = padding_truncation_manager.config_cache
            st.write(f"Cached configs: {len(padding_truncation_cached)}")
            
            if st.button("🗑️ Clear Padding/Truncation Cache"):
                padding_truncation_manager.clear_cache()
                st.success("Padding/truncation cache cleared")
                log_user_action("padding_truncation_cache_cleared_from_dashboard")
        
        with col3:
            st.markdown("**Pipeline Cache:**")
            pipeline_cached = pipeline_manager.list_cached_pipelines()
            st.write(f"Cached pipelines: {len(pipeline_cached)}")
            
            if st.button("🗑️ Clear Pipeline Cache"):
                pipeline_manager.clear_cache()
                st.success("Pipeline cache cleared")
                log_user_action("pipeline_cache_cleared_from_dashboard")
        
        # Add ML Apps cache in a new row
        st.markdown("#### Additional Cache Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**ML Apps Cache:**")
            ml_apps_cached = ml_apps_manager.list_cached_ml_apps()
            st.write(f"Cached ML apps: {len(ml_apps_cached)}")
            
            if st.button("🗑️ Clear ML Apps Cache"):
                ml_apps_manager.clear_cache()
                st.success("ML apps cache cleared")
                log_user_action("ml_apps_cache_cleared_from_dashboard")
        
        # Combined operations
        st.markdown("#### Combined Operations")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Caches", type="primary"):
                tokenizer_manager.clear_cache()
                image_processor_manager.clear_cache()
                video_processor_manager.clear_cache()
                backbone_manager.clear_cache()
                feature_extractor_manager.clear_cache()
                processor_manager.clear_cache()
                tokenizer_summary_manager.clear_cache()
                padding_truncation_manager.clear_cache()
                pipeline_manager.clear_cache()
                ml_apps_manager.clear_cache()
                st.success("All caches cleared")
                log_user_action("all_caches_cleared_from_dashboard")
        
        with col2:
            if st.button("🔄 Refresh Stats"):
                st.rerun()
    
    def render_integration_testing(self):
        """Render integration testing interface."""
        st.markdown("### 🧪 Integration Testing")
        
        st.markdown("""
        Test the integration between tokenizers and image processors with multimodal models.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Multimodal Model Test:**")
            
            test_model = st.selectbox(
                "Select multimodal model:",
                ["idefics2_8b", "glm_4_5v", "llava-hf/llava-1.5-7b-hf"],
                help="Select a multimodal model for integration testing"
            )
            
            if st.button("🚀 Run Integration Test"):
                self.run_integration_test(test_model)
        
        with col2:
            st.markdown("**Custom Integration Test:**")
            
            text_input = st.text_input(
                "Test text:",
                value="Describe this image in detail.",
                help="Text to use for tokenization testing"
            )
            
            if st.button("🔤 Test Tokenization"):
                self.test_tokenization(text_input)
    
    def run_integration_test(self, model_id: str):
        """Run integration test for multimodal model."""
        with st.spinner(f"Running integration test for {model_id}..."):
            try:
                start_time = time.time()
                
                # Test tokenizer loading
                tokenizer_info = tokenizer_manager.load_tokenizer(model_id)
                tokenizer_load_time = time.time() - start_time
                
                # Test image processor loading (if available)
                image_processor_info = None
                image_processor_load_time = 0
                
                try:
                    start_time = time.time()
                    image_processor_info = image_processor_manager.load_processor(model_id)
                    image_processor_load_time = time.time() - start_time
                except Exception as e:
                    debug(f"Image processor not available for {model_id}: {e}", "preprocessor_dashboard")
                
                # Test tokenization
                test_text = "Hello, this is a test message for integration testing."
                tokenization_result = tokenizer_manager.tokenize(test_text, model_id)
                
                total_time = time.time() - start_time
                
                # Display results
                st.success("✅ Integration test completed successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Tokenizer Results:**")
                    st.write(f"✅ Loaded: {tokenizer_info.model_id}")
                    st.write(f"⏱️ Load Time: {tokenizer_load_time:.3f}s")
                    st.write(f"🔤 Tokenization: Success")
                    st.write(f"📊 Token Count: {len(tokenization_result.input_ids)}")
                
                with col2:
                    st.markdown("**Image Processor Results:**")
                    if image_processor_info:
                        st.write(f"✅ Loaded: {image_processor_info.model_id}")
                        st.write(f"⏱️ Load Time: {image_processor_load_time:.3f}s")
                        st.write(f"🖼️ Processing: Available")
                    else:
                        st.write("❌ Not Available")
                        st.write("ℹ️ This model may not support image processing")
                
                st.markdown("**Overall Results:**")
                st.write(f"⏱️ Total Time: {total_time:.3f}s")
                st.write(f"🎯 Integration: {'✅ Success' if tokenization_result.success else '❌ Failed'}")
                
                # Store test results
                test_result = {
                    'timestamp': time.time(),
                    'model_id': model_id,
                    'tokenizer_loaded': bool(tokenizer_info),
                    'image_processor_loaded': bool(image_processor_info),
                    'tokenization_success': tokenization_result.success,
                    'total_time': total_time
                }
                
                if 'integration_tests' not in st.session_state.preprocessor_dashboard_state:
                    st.session_state.preprocessor_dashboard_state['integration_tests'] = []
                
                st.session_state.preprocessor_dashboard_state['integration_tests'].append(test_result)
                
                log_user_action("integration_test_completed", model_id=model_id, success=True)
                
            except Exception as e:
                st.error(f"❌ Integration test failed: {str(e)}")
                error(f"Integration test failed", "preprocessor_dashboard", e)
                log_user_action("integration_test_failed", model_id=model_id, error=str(e))
    
    def test_tokenization(self, text: str):
        """Test tokenization with sample text."""
        if not text:
            st.error("Please enter text to test")
            return
        
        with st.spinner("Testing tokenization..."):
            try:
                # Test with a common model
                model_id = "google/gemma-2-2b"
                
                start_time = time.time()
                result = tokenizer_manager.tokenize(text, model_id)
                processing_time = time.time() - start_time
                
                if result and result.success:
                    st.success("✅ Tokenization test successful!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Input:**")
                        st.write(text)
                        st.markdown("**Tokens:**")
                        tokens = result.tokenizer.tokenize(text)
                        st.write(tokens)
                    
                    with col2:
                        st.markdown("**Statistics:**")
                        st.write(f"Token Count: {len(tokens)}")
                        st.write(f"Processing Time: {processing_time:.3f}s")
                        st.write(f"Sequence Length: {len(result.input_ids)}")
                        
                        if result.attention_mask is not None:
                            valid_tokens = sum(result.attention_mask.tolist() if hasattr(result.attention_mask, 'tolist') else result.attention_mask)
                            st.write(f"Valid Tokens: {valid_tokens}")
                
                else:
                    st.error("❌ Tokenization test failed")
                    
            except Exception as e:
                st.error(f"❌ Tokenization test failed: {str(e)}")
                error(f"Tokenization test failed", "preprocessor_dashboard", e)
    
    def render_system_health(self):
        """Render system health status."""
        st.markdown("### 🏥 System Health")
        
        # Check system status
        tokenizer_healthy = True
        image_processor_healthy = True
        
        try:
            tokenizer_stats = tokenizer_manager.get_performance_stats()
        except Exception:
            tokenizer_healthy = False
        
        try:
            image_processor_stats = image_processor_manager.get_performance_stats()
        except Exception:
            image_processor_healthy = False
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Tokenizer System:**")
            if tokenizer_healthy:
                st.success("✅ Healthy")
                st.write(f"Cache Hit Rate: {tokenizer_stats['cache_hit_rate']:.1f}%")
                st.write(f"Error Count: {tokenizer_stats['error_count']}")
            else:
                st.error("❌ Unhealthy")
        
        with col2:
            st.markdown("**Image Processor System:**")
            if image_processor_healthy:
                st.success("✅ Healthy")
                st.write(f"Cache Hit Rate: {image_processor_stats['cache_hit_rate']:.1f}%")
                st.write(f"Error Count: {image_processor_stats['error_count']}")
            else:
                st.error("❌ Unhealthy")
        
        with col3:
            st.markdown("**Overall System:**")
            if tokenizer_healthy and image_processor_healthy:
                st.success("✅ All Systems Healthy")
                
                # Calculate overall health score
                avg_error_rate = (tokenizer_stats['error_count'] + image_processor_stats['error_count']) / 2
                health_score = max(0, 100 - (avg_error_rate * 10))
                st.metric("Health Score", f"{health_score:.0f}%")
            else:
                st.error("❌ System Issues Detected")
                st.metric("Health Score", "0%")
    
    def render_test_history(self):
        """Render integration test history."""
        tests = st.session_state.preprocessor_dashboard_state.get('integration_tests', [])
        
        if tests:
            st.markdown("### 📝 Integration Test History")
            
            # Convert to DataFrame for display
            test_data = []
            for test in tests[-10:]:  # Show last 10 tests
                test_data.append({
                    'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(test['timestamp'])),
                    'Model': test['model_id'],
                    'Tokenizer': '✅' if test['tokenizer_loaded'] else '❌',
                    'Image Processor': '✅' if test['image_processor_loaded'] else '❌',
                    'Tokenization': '✅' if test['tokenization_success'] else '❌',
                    'Time (s)': f"{test['total_time']:.3f}"
                })
            
            if test_data:
                df = pd.DataFrame(test_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No integration tests have been run yet.")
    
    def render(self):
        """Render the complete preprocessor dashboard."""
        self.render_header()
        
        # Overview metrics
        self.render_overview_metrics()
        
        # Performance comparison
        self.render_performance_comparison()
        
        # Cache management
        self.render_cache_management()
        
        # System health
        self.render_system_health()
        
        # Integration testing
        self.render_integration_testing()
        
        # Test history
        self.render_test_history()


def render_preprocessor_dashboard():
    """Render function for the preprocessor dashboard."""
    try:
        dashboard = PreprocessorDashboard()
        dashboard.render()
    except Exception as e:
        st.error(f"Error rendering preprocessor dashboard: {str(e)}")
        error(f"Failed to render preprocessor dashboard", "preprocessor_dashboard", e)


if __name__ == "__main__":
    # For testing
    render_preprocessor_dashboard()
