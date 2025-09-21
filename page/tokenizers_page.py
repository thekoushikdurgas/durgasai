"""
Tokenizers Page for DurgasAI.

This module provides a comprehensive interface for tokenizer management and testing:
- Tokenizer selection and configuration
- Real-time tokenization testing
- Batch processing capabilities
- Performance monitoring and statistics
- Advanced tokenizer features (padding, truncation, special tokens)
- Integration with enhanced tokenizer manager
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.tokenizer_manager import tokenizer_manager, TokenizerInfo, TokenizationResult
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config

# Try to import transformers for validation
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TokenizersPage:
    """
    Comprehensive tokenizer management and testing interface.
    
    This class provides:
    - Tokenizer selection and loading
    - Real-time tokenization testing
    - Batch processing capabilities
    - Performance monitoring
    - Advanced configuration options
    - Integration with enhanced tokenizer manager
    """
    
    def __init__(self):
        """Initialize the tokenizers page."""
        debug("Initializing TokenizersPage", "tokenizers_page")
        self.config = Config()
        
        # Initialize session state for tokenizers page
        if 'tokenizer_page_state' not in st.session_state:
            st.session_state.tokenizer_page_state = {
                'selected_model': 'google/gemma-2-2b',
                'test_texts': [
                    "Hello, how are you today?",
                    "The quick brown fox jumps over the lazy dog.",
                    "Transformers are amazing AI models! 🤗"
                ],
                'batch_mode': False,
                'show_advanced': False,
                'performance_stats': None
            }
        
        info("TokenizersPage initialized successfully", "tokenizers_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">🔤 Tokenizers Management</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📝 About Tokenizers</h4>
        <p>Tokenizers convert text into numerical representations (tokens) that AI models can understand. 
        This page provides comprehensive tools for:</p>
        <ul>
        <li><strong>Tokenizer Selection:</strong> Choose from various HuggingFace tokenizers</li>
        <li><strong>Real-time Testing:</strong> Test tokenization with your own text</li>
        <li><strong>Batch Processing:</strong> Process multiple texts efficiently</li>
        <li><strong>Performance Monitoring:</strong> Track tokenizer performance and caching</li>
        <li><strong>Advanced Features:</strong> Padding, truncation, special tokens, and more</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_model_selection(self):
        """Render tokenizer model selection interface."""
        st.markdown("### 🤖 Select Tokenizer Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular tokenizer models
            popular_models = [
                "google/gemma-2-2b",
                "microsoft/DialoGPT-medium", 
                "gpt2",
                "bert-base-uncased",
                "t5-small",
                "facebook/blenderbot-400M-distill",
                "EleutherAI/gpt-neo-125M"
            ]
            
            selected_model = st.selectbox(
                "Choose a tokenizer model:",
                options=popular_models,
                index=popular_models.index(st.session_state.tokenizer_page_state['selected_model']) 
                if st.session_state.tokenizer_page_state['selected_model'] in popular_models else 0,
                help="Select a HuggingFace tokenizer model to work with"
            )
            
            # Custom model input
            custom_model = st.text_input(
                "Or enter custom model ID:",
                placeholder="e.g., microsoft/DialoGPT-large",
                help="Enter any HuggingFace model ID for custom tokenizer"
            )
            
            if custom_model:
                selected_model = custom_model
        
        with col2:
            # Model info display
            st.markdown("**Model Information:**")
            if selected_model:
                model_info = self.get_model_info(selected_model)
                if model_info:
                    st.write(f"**Type:** {model_info.get('tokenizer_type', 'Unknown')}")
                    st.write(f"**Fast:** {model_info.get('is_fast', False)}")
                    st.write(f"**Vocab Size:** {model_info.get('vocab_size', 'Unknown')}")
                else:
                    st.info("Click 'Load Tokenizer' to get model information")
        
        # Update session state
        st.session_state.tokenizer_page_state['selected_model'] = selected_model
        
        return selected_model
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get tokenizer model information."""
        try:
            tokenizer_info = tokenizer_manager.get_tokenizer_info(model_id)
            if tokenizer_info:
                return tokenizer_info
        except Exception as e:
            debug(f"Failed to get model info: {e}", "tokenizers_page")
        
        return None
    
    def render_tokenizer_controls(self, model_id: str):
        """Render tokenizer loading and control buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Load Tokenizer", type="primary"):
                with st.spinner(f"Loading tokenizer for {model_id}..."):
                    start_time = time.time()
                    tokenizer_info = tokenizer_manager.load_tokenizer(model_id)
                    load_time = time.time() - start_time
                    
                    if tokenizer_info:
                        st.success(f"✅ Tokenizer loaded in {load_time:.2f}s")
                        log_user_action("tokenizer_loaded", model_id=model_id, load_time=load_time)
                    else:
                        st.error("❌ Failed to load tokenizer")
                        log_user_action("tokenizer_load_failed", model_id=model_id)
        
        with col2:
            if st.button("📊 Show Stats"):
                stats = tokenizer_manager.get_performance_stats()
                st.session_state.tokenizer_page_state['performance_stats'] = stats
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                tokenizer_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("tokenizer_cache_cleared")
        
        with col4:
            if st.button("📋 List Cached"):
                cached = tokenizer_manager.list_cached_tokenizers()
                st.session_state.tokenizer_page_state['cached_tokenizers'] = cached
    
    def render_testing_interface(self, model_id: str):
        """Render tokenization testing interface."""
        st.markdown("### 🧪 Tokenization Testing")
        
        # Mode selection
        col1, col2 = st.columns([1, 3])
        with col1:
            batch_mode = st.checkbox(
                "Batch Mode", 
                value=st.session_state.tokenizer_page_state['batch_mode'],
                help="Process multiple texts at once for better performance"
            )
            st.session_state.tokenizer_page_state['batch_mode'] = batch_mode
        
        with col2:
            if batch_mode:
                st.info("💡 Batch mode processes multiple texts efficiently using optimized tokenization")
        
        # Text input
        if batch_mode:
            self.render_batch_input()
        else:
            self.render_single_input()
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            self.render_advanced_options()
        
        # Process button
        if st.button("🚀 Tokenize", type="primary"):
            self.process_tokenization(model_id, batch_mode)
    
    def render_single_input(self):
        """Render single text input interface."""
        default_text = st.session_state.tokenizer_page_state['test_texts'][0]
        
        text_input = st.text_area(
            "Enter text to tokenize:",
            value=default_text,
            height=100,
            help="Enter any text to see how it gets tokenized"
        )
        
        st.session_state.tokenizer_page_state['current_text'] = text_input
    
    def render_batch_input(self):
        """Render batch text input interface."""
        st.markdown("**Enter multiple texts (one per line):**")
        
        # Get current test texts
        current_texts = st.session_state.tokenizer_page_state['test_texts']
        
        # Text area for batch input
        batch_text = st.text_area(
            "Batch texts:",
            value="\n".join(current_texts),
            height=150,
            help="Enter multiple texts, one per line"
        )
        
        # Parse batch texts
        batch_texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
        st.session_state.tokenizer_page_state['batch_texts'] = batch_texts
        
        st.write(f"📝 {len(batch_texts)} texts ready for batch processing")
    
    def render_advanced_options(self):
        """Render advanced tokenization options."""
        col1, col2 = st.columns(2)
        
        with col1:
            padding = st.checkbox("Padding", value=True, help="Pad sequences to same length")
            truncation = st.checkbox("Truncation", value=True, help="Truncate long sequences")
            add_special_tokens = st.checkbox("Add Special Tokens", value=True, help="Add BOS, EOS tokens")
        
        with col2:
            max_length = st.number_input("Max Length", min_value=1, max_value=2048, value=512, help="Maximum sequence length")
            return_tensors = st.selectbox("Return Tensors", ["pt", "tf", "np", "None"], index=0, help="Tensor format for output")
            return_attention_mask = st.checkbox("Return Attention Mask", value=True, help="Include attention mask")
        
        # Store options
        st.session_state.tokenizer_page_state['advanced_options'] = {
            'padding': padding,
            'truncation': truncation,
            'add_special_tokens': add_special_tokens,
            'max_length': max_length,
            'return_tensors': return_tensors if return_tensors != "None" else None,
            'return_attention_mask': return_attention_mask
        }
    
    def process_tokenization(self, model_id: str, batch_mode: bool):
        """Process tokenization with the selected options."""
        try:
            # Get options
            options = st.session_state.tokenizer_page_state.get('advanced_options', {})
            
            if batch_mode:
                texts = st.session_state.tokenizer_page_state.get('batch_texts', [])
                if not texts:
                    st.error("No texts provided for batch processing")
                    return
                
                # Process batch
                with st.spinner("Processing batch tokenization..."):
                    start_time = time.time()
                    result = tokenizer_manager.safe_batch_tokenize(texts, model_id, **options)
                    processing_time = time.time() - start_time
                
                self.display_batch_results(result, processing_time)
                
            else:
                text = st.session_state.tokenizer_page_state.get('current_text', '')
                if not text:
                    st.error("No text provided for tokenization")
                    return
                
                # Process single text
                with st.spinner("Processing tokenization..."):
                    start_time = time.time()
                    result = tokenizer_manager.tokenize(text, model_id, **options)
                    processing_time = time.time() - start_time
                
                self.display_single_results(result, processing_time)
            
            log_user_action("tokenization_processed", model_id=model_id, batch_mode=batch_mode)
            
        except Exception as e:
            st.error(f"Tokenization failed: {str(e)}")
            error(f"Tokenization processing failed", "tokenizers_page", e)
    
    def display_single_results(self, result: TokenizationResult, processing_time: float):
        """Display single text tokenization results."""
        st.markdown("### 📊 Tokenization Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input text
            st.markdown("**📝 Input Text:**")
            st.text_area("", value=result.input_text, height=100, disabled=True)
            
            # Tokens
            st.markdown("**🔤 Tokens:**")
            tokens = result.tokenizer.tokenize(result.input_text)
            st.write(f"Tokens: `{tokens}`")
            
            # Token IDs
            st.markdown("**🔢 Token IDs:**")
            st.write(f"Input IDs: `{result.input_ids.tolist() if hasattr(result.input_ids, 'tolist') else result.input_ids}`")
            
            if result.attention_mask is not None:
                st.markdown("**🎭 Attention Mask:**")
                st.write(f"Attention Mask: `{result.attention_mask.tolist() if hasattr(result.attention_mask, 'tolist') else result.attention_mask}`")
        
        with col2:
            # Statistics
            st.markdown("**📈 Statistics:**")
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Token Count", len(tokens))
            st.metric("Sequence Length", len(result.input_ids))
            
            if result.attention_mask is not None:
                attention_sum = sum(result.attention_mask.tolist() if hasattr(result.attention_mask, 'tolist') else result.attention_mask)
                st.metric("Valid Tokens", int(attention_sum))
            
            # Decode test
            if st.button("🔄 Decode Tokens"):
                decoded = result.tokenizer.decode(result.input_ids, skip_special_tokens=True)
                st.text_area("Decoded Text:", value=decoded, height=80, disabled=True)
    
    def display_batch_results(self, result: TokenizationResult, processing_time: float):
        """Display batch tokenization results."""
        st.markdown("### 📊 Batch Tokenization Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Batch Size", result.batch_size)
            st.metric("Success", "✅" if result.success else "❌")
        
        with col2:
            if result.batch_size > 0:
                avg_time = processing_time / result.batch_size
                st.metric("Avg Time per Text", f"{avg_time:.3f}s")
                st.metric("Throughput", f"{result.batch_size/processing_time:.1f} texts/sec")
        
        # Detailed results
        if result.success and hasattr(result, 'batch_results'):
            st.markdown("**📝 Detailed Results:**")
            
            for i, batch_result in enumerate(result.batch_results):
                with st.expander(f"Text {i+1}: {result.input_texts[i][:50]}..."):
                    tokens = result.tokenizer.tokenize(result.input_texts[i])
                    st.write(f"**Tokens:** {tokens}")
                    st.write(f"**Token Count:** {len(tokens)}")
                    
                    if st.button(f"Decode Text {i+1}", key=f"decode_{i}"):
                        decoded = result.tokenizer.decode(batch_result['input_ids'], skip_special_tokens=True)
                        st.text_area("Decoded:", value=decoded, height=60, disabled=True)
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.tokenizer_page_state.get('performance_stats')
        
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
                st.metric("Disk Cache Size", stats['disk_cache_size'])
                st.metric("Total Tokenizations", stats['total_tokenizations'])
                st.metric("Batch Tokenizations", stats['batch_tokenizations'])
    
    def render_cached_tokenizers(self):
        """Render cached tokenizers information."""
        cached = st.session_state.tokenizer_page_state.get('cached_tokenizers', [])
        
        if cached:
            st.markdown("### 💾 Cached Tokenizers")
            
            for tokenizer_info in cached:
                with st.expander(f"🤖 {tokenizer_info['model_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {tokenizer_info['tokenizer_type']}")
                        st.write(f"**Fast:** {tokenizer_info['is_fast']}")
                        st.write(f"**Vocab Size:** {tokenizer_info['vocab_size']}")
                    
                    with col2:
                        st.write(f"**Load Time:** {tokenizer_info['load_time']:.3f}s")
                        st.write(f"**Last Used:** {tokenizer_info['last_used']}")
                        st.write(f"**Cache Key:** {tokenizer_info['cache_key'][:16]}...")
    
    def render(self):
        """Render the complete tokenizers page."""
        self.render_header()
        
        # Model selection
        selected_model = self.render_model_selection()
        
        # Tokenizer controls
        self.render_tokenizer_controls(selected_model)
        
        # Testing interface
        self.render_testing_interface(selected_model)
        
        # Performance stats
        if st.session_state.tokenizer_page_state.get('performance_stats'):
            self.render_performance_stats()
        
        # Cached tokenizers
        if st.session_state.tokenizer_page_state.get('cached_tokenizers'):
            self.render_cached_tokenizers()


def render_tokenizers_page():
    """Render function for the tokenizers page."""
    try:
        page = TokenizersPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering tokenizers page: {str(e)}")
        error(f"Failed to render tokenizers page", "tokenizers_page", e)


if __name__ == "__main__":
    # For testing
    render_tokenizers_page()
