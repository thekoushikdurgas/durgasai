"""
Padding and Truncation Page for DurgasAI.

This module provides a comprehensive interface for padding and truncation management:
- Strategy selection and configuration
- Real-time padding and truncation testing
- Performance monitoring and optimization
- Memory usage analysis
- Batch processing optimization
- Integration with enhanced padding and truncation manager
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

from utils.padding_truncation_manager import (
    padding_truncation_manager, 
    PaddingTruncationConfig, 
    PaddingTruncationResult,
    StrategyAnalysisResult
)
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import required libraries
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PaddingTruncationPage:
    """
    Comprehensive padding and truncation management interface.
    
    This class provides:
    - Strategy selection and configuration
    - Real-time padding and truncation testing
    - Performance monitoring and optimization
    - Memory usage analysis
    - Batch processing optimization
    - Integration with enhanced padding and truncation manager
    """
    
    def __init__(self):
        """Initialize the padding and truncation page."""
        debug("Initializing PaddingTruncationPage", "padding_truncation_page")
        self.config = Config()
        
        # Initialize session state for padding and truncation page
        if 'padding_truncation_page_state' not in st.session_state:
            st.session_state.padding_truncation_page_state = {
                'selected_tokenizer': 'google-bert/bert-base-uncased',
                'selected_strategy': 'training',
                'test_texts': [
                    "Hello world!",
                    "This is a longer sentence with more words.",
                    "This is an even longer sentence that contains many more words and will demonstrate padding and truncation behavior effectively."
                ],
                'show_advanced': False,
                'performance_stats': None,
                'test_results': {},
                'comparison_results': None
            }
        
        info("PaddingTruncationPage initialized successfully", "padding_truncation_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">📏 Padding and Truncation Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📏 About Padding and Truncation</h4>
        <p>Padding and Truncation are essential strategies for handling variable-length sequences in batched inputs for transformer models. 
        This page provides advanced tools for:</p>
        <ul>
        <li><strong>Strategy Configuration:</strong> Choose from predefined or custom padding and truncation strategies</li>
        <li><strong>Real-time Testing:</strong> Test different strategies with your own text data</li>
        <li><strong>Performance Analysis:</strong> Monitor processing times, memory usage, and efficiency</li>
        <li><strong>Batch Optimization:</strong> Optimize batch processing for maximum efficiency</li>
        <li><strong>Memory Management:</strong> Analyze and optimize memory usage patterns</li>
        <li><strong>Strategy Comparison:</strong> Compare different strategies side-by-side</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_tokenizer_selection(self):
        """Render tokenizer selection interface."""
        st.markdown("### 🤖 Select Tokenizer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular tokenizer models
            tokenizer_models = [
                "google-bert/bert-base-uncased",
                "gpt2",
                "xlnet-base-cased",
                "roberta-base",
                "distilbert-base-uncased",
                "albert-base-v1"
            ]
            
            selected_tokenizer = st.selectbox(
                "Choose a tokenizer:",
                options=tokenizer_models,
                index=tokenizer_models.index(st.session_state.padding_truncation_page_state['selected_tokenizer']) 
                if st.session_state.padding_truncation_page_state['selected_tokenizer'] in tokenizer_models else 0,
                help="Select a HuggingFace tokenizer for testing padding and truncation"
            )
            
            # Custom tokenizer input
            custom_tokenizer = st.text_input(
                "Or enter custom tokenizer ID:",
                placeholder="e.g., google-bert/bert-base-uncased",
                help="Enter any HuggingFace tokenizer ID"
            )
            
            if custom_tokenizer:
                selected_tokenizer = custom_tokenizer
        
        with col2:
            # Tokenizer info display
            st.markdown("**Tokenizer Information:**")
            if selected_tokenizer:
                try:
                    if TRANSFORMERS_AVAILABLE:
                        tokenizer = AutoTokenizer.from_pretrained(selected_tokenizer)
                        st.write(f"**Type:** {type(tokenizer).__name__}")
                        st.write(f"**Vocab Size:** {tokenizer.vocab_size:,}")
                        st.write(f"**Model Max Length:** {tokenizer.model_max_length}")
                        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
                            st.write(f"**Pad Token:** {tokenizer.pad_token}")
                        else:
                            st.write("**Pad Token:** None")
                    else:
                        st.info("Transformers not available")
                except Exception as e:
                    st.error(f"Error loading tokenizer: {e}")
        
        # Update session state
        st.session_state.padding_truncation_page_state['selected_tokenizer'] = selected_tokenizer
        
        return selected_tokenizer
    
    def render_strategy_selection(self):
        """Render strategy selection interface."""
        st.markdown("### ⚙️ Select Strategy")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Predefined strategies
            predefined_strategies = padding_truncation_manager.list_predefined_strategies()
            
            selected_strategy = st.selectbox(
                "Choose a predefined strategy:",
                options=predefined_strategies,
                index=predefined_strategies.index(st.session_state.padding_truncation_page_state['selected_strategy']) 
                if st.session_state.padding_truncation_page_state['selected_strategy'] in predefined_strategies else 0,
                help="Select a predefined padding and truncation strategy"
            )
            
            # Custom strategy toggle
            use_custom = st.checkbox("Use custom strategy", help="Create a custom padding and truncation configuration")
            
            if use_custom:
                selected_strategy = "custom"
        
        with col2:
            # Strategy info display
            st.markdown("**Strategy Information:**")
            if selected_strategy != "custom":
                strategy_config = padding_truncation_manager.get_predefined_strategy(selected_strategy)
                if strategy_config:
                    st.write(f"**Padding:** {strategy_config.padding}")
                    st.write(f"**Truncation:** {strategy_config.truncation}")
                    st.write(f"**Max Length:** {strategy_config.max_length}")
                    st.write(f"**Return Tensors:** {strategy_config.return_tensors}")
            else:
                st.info("Custom strategy selected")
        
        # Update session state
        st.session_state.padding_truncation_page_state['selected_strategy'] = selected_strategy
        
        return selected_strategy
    
    def render_text_input(self):
        """Render text input interface."""
        st.markdown("### 📝 Test Text Input")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input options
            text_input_method = st.radio(
                "Choose text input method:",
                ["Use Sample Texts", "Enter Custom Texts", "Upload Text File (Coming Soon)"],
                horizontal=True
            )
            
            if text_input_method == "Use Sample Texts":
                sample_sets = {
                    "Short Sentences": [
                        "Hello world!",
                        "How are you?",
                        "Good morning!"
                    ],
                    "Mixed Lengths": [
                        "Hello world!",
                        "This is a longer sentence with more words.",
                        "This is an even longer sentence that contains many more words and will demonstrate padding and truncation behavior effectively."
                    ],
                    "Very Long": [
                        "This is an extremely long sentence that contains many, many words and will definitely need to be truncated to fit within reasonable limits for most transformer models.",
                        "Another very long sentence that goes on and on with many different words and phrases to test the truncation capabilities of our padding and truncation system.",
                        "Yet another extremely lengthy sentence that continues for a very long time with numerous words, phrases, and clauses to thoroughly test the padding and truncation functionality."
                    ]
                }
                
                selected_sample = st.selectbox(
                    "Choose sample text set:",
                    options=list(sample_sets.keys()),
                    index=list(sample_sets.keys()).index("Mixed Lengths")
                )
                test_texts = sample_sets[selected_sample]
                
            elif text_input_method == "Enter Custom Texts":
                custom_texts = st.text_area(
                    "Enter your texts (one per line):",
                    value="\n".join(st.session_state.padding_truncation_page_state.get('test_texts', [])),
                    height=150,
                    help="Enter multiple texts, one per line"
                )
                test_texts = [text.strip() for text in custom_texts.split('\n') if text.strip()]
            else:
                st.info("📄 Text file upload functionality will be available in future updates")
                test_texts = st.session_state.padding_truncation_page_state.get('test_texts', [])
        
        with col2:
            # Text statistics
            if test_texts:
                st.markdown("**Text Statistics:**")
                st.write(f"**Number of texts:** {len(test_texts)}")
                
                # Calculate word counts
                word_counts = [len(text.split()) for text in test_texts]
                st.write(f"**Min words:** {min(word_counts)}")
                st.write(f"**Max words:** {max(word_counts)}")
                st.write(f"**Avg words:** {sum(word_counts) / len(word_counts):.1f}")
                
                # Show preview
                st.markdown("**Preview:**")
                for i, text in enumerate(test_texts[:3]):
                    preview = text[:50] + "..." if len(text) > 50 else text
                    st.write(f"{i+1}. {preview}")
                if len(test_texts) > 3:
                    st.write(f"... and {len(test_texts) - 3} more")
        
        # Update session state
        st.session_state.padding_truncation_page_state['test_texts'] = test_texts
        
        return test_texts
    
    def render_custom_strategy_config(self):
        """Render custom strategy configuration interface."""
        st.markdown("### 🔧 Custom Strategy Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Padding Configuration:**")
            padding_option = st.selectbox(
                "Padding Strategy:",
                ["True (longest)", "max_length", "False (no padding)"],
                index=0
            )
            
            if padding_option == "True (longest)":
                padding = True
            elif padding_option == "max_length":
                padding = "max_length"
            else:
                padding = False
            
            max_length = st.number_input(
                "Max Length:",
                min_value=1,
                max_value=4096,
                value=512,
                help="Maximum sequence length"
            )
            
            pad_to_multiple_of = st.number_input(
                "Pad to Multiple Of:",
                min_value=1,
                max_value=128,
                value=8,
                help="Pad sequences to multiples of this number"
            )
        
        with col2:
            st.markdown("**Truncation Configuration:**")
            truncation_option = st.selectbox(
                "Truncation Strategy:",
                ["True (longest_first)", "only_first", "only_second", "False (no truncation)"],
                index=0
            )
            
            if truncation_option == "True (longest_first)":
                truncation = True
            elif truncation_option == "only_first":
                truncation = "only_first"
            elif truncation_option == "only_second":
                truncation = "only_second"
            else:
                truncation = False
            
            return_tensors = st.selectbox(
                "Return Tensors:",
                ["pt", "tf", "np", "None"],
                index=0
            )
            
            return_tensors = return_tensors if return_tensors != "None" else None
            
            return_attention_mask = st.checkbox("Return Attention Mask", value=True)
        
        # Create custom config
        custom_config = padding_truncation_manager.create_custom_config(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask
        )
        
        return custom_config
    
    def render_test_controls(self, selected_tokenizer: str, selected_strategy: str, test_texts: List[str]):
        """Render test control buttons."""
        st.markdown("### 🧪 Test Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Test Strategy", type="primary"):
                if selected_tokenizer and selected_strategy and test_texts:
                    self.run_strategy_test(selected_tokenizer, selected_strategy, test_texts)
                else:
                    st.error("Please select tokenizer, strategy, and enter texts for testing")
        
        with col2:
            if st.button("🔄 Compare Strategies"):
                if selected_tokenizer and test_texts:
                    self.run_strategy_comparison(selected_tokenizer, test_texts)
                else:
                    st.error("Please select tokenizer and enter texts for comparison")
        
        with col3:
            if st.button("📊 Show Stats"):
                stats = padding_truncation_manager.get_performance_stats()
                st.session_state.padding_truncation_page_state['performance_stats'] = stats
        
        with col4:
            if st.button("🗑️ Clear Cache"):
                padding_truncation_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("padding_truncation_cache_cleared")
    
    def run_strategy_test(self, selected_tokenizer: str, selected_strategy: str, test_texts: List[str]):
        """Run strategy test."""
        with st.spinner("Testing padding and truncation strategy..."):
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(selected_tokenizer)
                
                # Get strategy configuration
                if selected_strategy == "custom":
                    config = self.render_custom_strategy_config()
                else:
                    config = padding_truncation_manager.get_predefined_strategy(selected_strategy)
                
                if config:
                    # Apply padding and truncation
                    result = padding_truncation_manager.apply_padding_truncation(
                        tokenizer, test_texts, config
                    )
                    
                    if result and result.success:
                        st.session_state.padding_truncation_page_state['test_results'] = {
                            'strategy': selected_strategy,
                            'result': result
                        }
                        st.success(f"✅ Strategy test completed successfully")
                        log_user_action("padding_truncation_test_completed", 
                                      strategy=selected_strategy, 
                                      num_texts=len(test_texts))
                    else:
                        st.error(f"❌ Strategy test failed: {result.error if result else 'Unknown error'}")
                else:
                    st.error("❌ Failed to get strategy configuration")
                    
            except Exception as e:
                st.error(f"❌ Strategy test failed: {str(e)}")
                error(f"Strategy test failed", "padding_truncation_page", e)
    
    def run_strategy_comparison(self, selected_tokenizer: str, test_texts: List[str]):
        """Run strategy comparison."""
        with st.spinner("Comparing padding and truncation strategies..."):
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(selected_tokenizer)
                
                # Get all predefined strategies
                strategy_names = padding_truncation_manager.list_predefined_strategies()
                
                # Compare strategies
                results = padding_truncation_manager.compare_strategies(
                    tokenizer, test_texts, strategy_names
                )
                
                st.session_state.padding_truncation_page_state['comparison_results'] = results
                st.success(f"✅ Strategy comparison completed for {len(results)} strategies")
                
                log_user_action("padding_truncation_comparison_completed", 
                              num_strategies=len(results), 
                              num_texts=len(test_texts))
                
            except Exception as e:
                st.error(f"❌ Strategy comparison failed: {str(e)}")
                error(f"Strategy comparison failed", "padding_truncation_page", e)
    
    def render_test_results(self):
        """Render test results."""
        test_results = st.session_state.padding_truncation_page_state.get('test_results', {})
        
        if not test_results:
            return
        
        result = test_results.get('result')
        strategy = test_results.get('strategy')
        
        if not result:
            return
        
        st.markdown("### 📊 Test Results")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Processing Time", f"{result.processing_time:.3f}s")
            st.metric("Batch Size", result.batch_size)
        
        with col2:
            st.metric("Sequence Length", result.sequence_length)
            st.metric("Padding Applied", "✅" if result.padding_applied else "❌")
        
        with col3:
            st.metric("Truncation Applied", "✅" if result.truncation_applied else "❌")
            if result.metadata:
                st.metric("Memory Usage", f"{result.metadata.get('memory_usage_mb', 0):.1f} MB")
        
        with col4:
            if result.metadata:
                st.metric("Efficiency Score", f"{result.metadata.get('efficiency_score', 0):.2f}")
                st.metric("Padding Tokens", result.metadata.get('padding_tokens_added', 0))
        
        # Detailed results
        if result.success:
            with st.expander("🔍 Detailed Results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Configuration:**")
                    st.write(f"**Padding:** {result.config.padding}")
                    st.write(f"**Truncation:** {result.config.truncation}")
                    st.write(f"**Max Length:** {result.config.max_length}")
                    st.write(f"**Return Tensors:** {result.config.return_tensors}")
                    st.write(f"**Return Attention Mask:** {result.config.return_attention_mask}")
                
                with col2:
                    st.markdown("**Results:**")
                    st.write(f"**Input IDs Shape:** {result.input_ids.shape if hasattr(result.input_ids, 'shape') else 'N/A'}")
                    st.write(f"**Attention Mask Shape:** {result.attention_mask.shape if hasattr(result.attention_mask, 'shape') else 'N/A'}")
                    st.write(f"**Processing Time:** {result.processing_time:.3f}s")
                    st.write(f"**Success:** {result.success}")
                
                # Show input IDs and attention mask
                if result.input_ids is not None:
                    st.markdown("**Input IDs:**")
                    st.write(result.input_ids)
                
                if result.attention_mask is not None:
                    st.markdown("**Attention Mask:**")
                    st.write(result.attention_mask)
    
    def render_comparison_results(self):
        """Render strategy comparison results."""
        comparison_results = st.session_state.padding_truncation_page_state.get('comparison_results')
        
        if not comparison_results:
            return
        
        st.markdown("### 📊 Strategy Comparison Results")
        
        # Create comparison table
        comparison_data = []
        for strategy_name, result in comparison_results.items():
            if result.success:
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Padding': str(result.config.padding),
                    'Truncation': str(result.config.truncation),
                    'Max Length': result.config.max_length,
                    'Processing Time (s)': f"{result.processing_time:.3f}",
                    'Memory Usage (MB)': f"{result.memory_usage:.1f}",
                    'Sequence Length': result.sequence_length,
                    'Efficiency Score': f"{result.efficiency_score:.2f}",
                    'Padding Tokens': result.padding_tokens_added,
                    'Success': "✅" if result.success else "❌"
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Find best strategy
            if comparison_data:
                best_strategy = max(comparison_data, key=lambda x: x['Efficiency Score'])
                st.success(f"🏆 Best Strategy: {best_strategy['Strategy']} (Efficiency: {best_strategy['Efficiency Score']})")
            
            # Detailed comparison
            with st.expander("🔍 Detailed Comparison"):
                for strategy_name, result in comparison_results.items():
                    if result.success:
                        with st.expander(f"📋 {strategy_name}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Configuration:**")
                                st.write(f"Padding: {result.config.padding}")
                                st.write(f"Truncation: {result.config.truncation}")
                                st.write(f"Max Length: {result.config.max_length}")
                                st.write(f"Return Tensors: {result.config.return_tensors}")
                            
                            with col2:
                                st.write(f"**Performance:**")
                                st.write(f"Processing Time: {result.processing_time:.3f}s")
                                st.write(f"Memory Usage: {result.memory_usage:.1f} MB")
                                st.write(f"Efficiency Score: {result.efficiency_score:.2f}")
                                st.write(f"Padding Tokens: {result.padding_tokens_added}")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.padding_truncation_page_state.get('performance_stats')
        
        if stats:
            st.markdown("### 📊 Performance Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Operations", stats['total_operations'])
                st.metric("Cache Hits", stats['cache_hits'])
                st.metric("Cache Misses", stats['cache_misses'])
            
            with col2:
                st.metric("Avg Processing Time", f"{stats['average_processing_time']:.3f}s")
                st.metric("Total Tokens Processed", f"{stats['total_tokens_processed']:,}")
                st.metric("Total Padding Tokens", f"{stats['total_padding_tokens']:,}")
            
            with col3:
                st.metric("Strategy Comparisons", stats['strategy_comparisons'])
                st.metric("Memory Optimizations", stats['memory_optimizations'])
                st.metric("Batch Optimizations", stats['batch_optimizations'])
                st.metric("Error Count", stats['error_count'])
    
    def render_advanced_options(self):
        """Render advanced options."""
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Memory Management:**")
                show_memory_analysis = st.checkbox("Show Memory Analysis", value=True)
                enable_optimization = st.checkbox("Enable Auto-Optimization", value=True)
                
            with col2:
                st.markdown("**Performance Monitoring:**")
                show_performance_details = st.checkbox("Show Performance Details", value=True)
                enable_caching = st.checkbox("Enable Caching", value=True)
            
            # Store options
            st.session_state.padding_truncation_page_state['advanced_options'] = {
                'show_memory_analysis': show_memory_analysis,
                'enable_optimization': enable_optimization,
                'show_performance_details': show_performance_details,
                'enable_caching': enable_caching
            }
    
    def render(self):
        """Render the complete padding and truncation page."""
        self.render_header()
        
        # Tokenizer selection
        selected_tokenizer = self.render_tokenizer_selection()
        
        # Strategy selection
        selected_strategy = self.render_strategy_selection()
        
        # Text input
        test_texts = self.render_text_input()
        
        # Test controls
        if selected_tokenizer and selected_strategy and test_texts:
            self.render_test_controls(selected_tokenizer, selected_strategy, test_texts)
        
        # Advanced options
        self.render_advanced_options()
        
        # Test results
        self.render_test_results()
        
        # Comparison results
        self.render_comparison_results()
        
        # Performance stats
        self.render_performance_stats()


def render_padding_truncation_page():
    """Render function for the padding and truncation page."""
    try:
        page = PaddingTruncationPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering padding and truncation page: {str(e)}")
        error(f"Failed to render padding and truncation page", "padding_truncation_page", e)


if __name__ == "__main__":
    # For testing
    render_padding_truncation_page()
