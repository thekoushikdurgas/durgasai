"""
Tokenizer Summary Page for DurgasAI.

This module provides a comprehensive interface for tokenizer analysis and comparison:
- Tokenizer algorithm analysis and comparison
- Real-time tokenization testing and analysis
- Vocabulary analysis and insights
- Performance monitoring and statistics
- Advanced tokenization features (subword analysis, language support)
- Integration with enhanced tokenizer summary manager
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

from utils.tokenizer_summary_manager import tokenizer_summary_manager, TokenizerSummaryInfo, TokenizationAnalysisResult
from utils.logger import debug, info, warning, error, log_user_action
from utils.config import Config


class TokenizerSummaryPage:
    """
    Comprehensive tokenizer analysis and comparison interface.
    
    This class provides:
    - Tokenizer algorithm analysis and comparison
    - Real-time tokenization testing and analysis
    - Vocabulary analysis and insights
    - Performance monitoring
    - Advanced configuration options
    - Integration with enhanced tokenizer summary manager
    """
    
    def __init__(self):
        """Initialize the tokenizer summary page."""
        debug("Initializing TokenizerSummaryPage", "tokenizer_summary_page")
        self.config = Config()
        
        # Initialize session state for tokenizer summary page
        if 'tokenizer_summary_page_state' not in st.session_state:
            st.session_state.tokenizer_summary_page_state = {
                'selected_models': ['google-bert/bert-base-uncased'],
                'test_text': "Don't you love 🤗 Transformers? We sure do.",
                'show_advanced': False,
                'performance_stats': None,
                'analysis_results': {},
                'comparison_results': None
            }
        
        info("TokenizerSummaryPage initialized successfully", "tokenizer_summary_page")
    
    def render_header(self):
        """Render the page header with title and description."""
        st.markdown('<h1 class="main-header">📝 Tokenizer Summary Analysis</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📝 About Tokenizer Summary</h4>
        <p>Tokenizer Summary provides comprehensive analysis of tokenization algorithms used in Hugging Face Transformers. 
        This page provides advanced tools for:</p>
        <ul>
        <li><strong>Algorithm Analysis:</strong> Compare BPE, WordPiece, SentencePiece, and Unigram algorithms</li>
        <li><strong>Tokenization Analysis:</strong> Analyze subword ratios, vocabulary coverage, and unknown tokens</li>
        <li><strong>Vocabulary Insights:</strong> Understand vocabulary size, merge rules, and language support</li>
        <li><strong>Performance Monitoring:</strong> Track tokenizer performance and caching statistics</li>
        <li><strong>Comparative Analysis:</strong> Compare multiple tokenizers side-by-side</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_model_selection(self):
        """Render tokenizer model selection interface."""
        st.markdown("### 🤖 Select Tokenizer Models")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Popular tokenizer models organized by algorithm
            model_groups = {
                "WordPiece (BERT-style)": [
                    "google-bert/bert-base-uncased",
                    "google-bert/bert-base-cased",
                    "distilbert-base-uncased",
                    "electra-base-discriminator"
                ],
                "BPE (GPT-style)": [
                    "gpt2",
                    "gpt2-medium",
                    "roberta-base",
                    "facebook/opt-125m"
                ],
                "SentencePiece": [
                    "xlnet-base-cased",
                    "albert-base-v1",
                    "t5-small",
                    "google/flan-t5-small"
                ],
                "Multilingual": [
                    "xlm-roberta-base",
                    "facebook/mbart-large-cc25",
                    "facebook/m2m100_418M"
                ]
            }
            
            selected_models = []
            
            # Allow selection from each group
            for group_name, models in model_groups.items():
                st.markdown(f"**{group_name}:**")
                group_selections = st.multiselect(
                    f"Select {group_name.lower()} models:",
                    options=models,
                    default=[models[0]] if models[0] in st.session_state.tokenizer_summary_page_state['selected_models'] else [],
                    key=f"select_{group_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                )
                selected_models.extend(group_selections)
            
            # Custom model input
            st.markdown("**Custom Models:**")
            custom_models = st.text_area(
                "Enter custom model IDs (one per line):",
                placeholder="google-bert/bert-base-uncased\ngpt2\nxlnet-base-cased",
                help="Enter any HuggingFace model ID for custom tokenizer analysis"
            )
            
            if custom_models.strip():
                custom_model_list = [model.strip() for model in custom_models.split('\n') if model.strip()]
                selected_models.extend(custom_model_list)
            
            # Remove duplicates and update session state
            selected_models = list(set(selected_models))
            st.session_state.tokenizer_summary_page_state['selected_models'] = selected_models
        
        with col2:
            # Model info display
            st.markdown("**Selected Models:**")
            if selected_models:
                for i, model in enumerate(selected_models, 1):
                    st.write(f"{i}. {model}")
                st.write(f"**Total:** {len(selected_models)} models selected")
            else:
                st.info("No models selected")
        
        return selected_models
    
    def render_text_input(self):
        """Render text input interface."""
        st.markdown("### 📝 Test Text Input")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input options
            text_input_method = st.radio(
                "Choose text input method:",
                ["Use Sample Text", "Enter Custom Text", "Upload Text File (Coming Soon)"],
                horizontal=True
            )
            
            if text_input_method == "Use Sample Text":
                sample_texts = {
                    "Simple English": "Don't you love 🤗 Transformers? We sure do.",
                    "Technical Text": "The transformer architecture uses self-attention mechanisms for sequence modeling.",
                    "Multilingual": "Hello world! Bonjour le monde! ¡Hola mundo!",
                    "Special Characters": "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
                    "Long Text": "This is a longer text sample that will help us analyze how different tokenizers handle more complex sentences with various punctuation marks and word combinations."
                }
                
                selected_sample = st.selectbox(
                    "Choose sample text:",
                    options=list(sample_texts.keys()),
                    index=list(sample_texts.keys()).index("Simple English")
                )
                test_text = sample_texts[selected_sample]
                
            elif text_input_method == "Enter Custom Text":
                test_text = st.text_area(
                    "Enter your text:",
                    value=st.session_state.tokenizer_summary_page_state.get('test_text', ''),
                    height=100,
                    help="Enter any text to analyze tokenization"
                )
            else:
                st.info("📄 Text file upload functionality will be available in future updates")
                test_text = st.session_state.tokenizer_summary_page_state.get('test_text', '')
        
        with col2:
            # Text statistics
            if test_text:
                st.markdown("**Text Statistics:**")
                st.write(f"**Length:** {len(test_text)} characters")
                st.write(f"**Words:** {len(test_text.split())} words")
                st.write(f"**Lines:** {len(test_text.splitlines())} lines")
                
                # Show preview
                st.markdown("**Preview:**")
                preview = test_text[:100] + "..." if len(test_text) > 100 else test_text
                st.text(preview)
        
        # Update session state
        st.session_state.tokenizer_summary_page_state['test_text'] = test_text
        
        return test_text
    
    def render_analysis_controls(self, selected_models: List[str], test_text: str):
        """Render analysis control buttons."""
        st.markdown("### 🔄 Analysis Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Analyze Tokenizers", type="primary"):
                if selected_models and test_text:
                    self.run_tokenizer_analysis(selected_models, test_text)
                else:
                    st.error("Please select models and enter text for analysis")
        
        with col2:
            if st.button("📊 Compare Algorithms"):
                if len(selected_models) >= 2 and test_text:
                    self.run_algorithm_comparison(selected_models, test_text)
                else:
                    st.error("Please select at least 2 models for comparison")
        
        with col3:
            if st.button("📈 Show Stats"):
                stats = tokenizer_summary_manager.get_performance_stats()
                st.session_state.tokenizer_summary_page_state['performance_stats'] = stats
        
        with col4:
            if st.button("🗑️ Clear Cache"):
                tokenizer_summary_manager.clear_cache()
                st.success("Cache cleared successfully")
                log_user_action("tokenizer_summary_cache_cleared")
    
    def run_tokenizer_analysis(self, selected_models: List[str], test_text: str):
        """Run tokenizer analysis for selected models."""
        with st.spinner("Analyzing tokenizers..."):
            results = {}
            
            progress_bar = st.progress(0)
            total_models = len(selected_models)
            
            for i, model_id in enumerate(selected_models):
                try:
                    # Update progress
                    progress_bar.progress((i + 1) / total_models)
                    
                    # Analyze tokenizer
                    result = tokenizer_summary_manager.analyze_tokenization(test_text, model_id)
                    if result:
                        results[model_id] = result
                        
                except Exception as e:
                    st.error(f"Failed to analyze {model_id}: {str(e)}")
                    continue
            
            # Store results
            st.session_state.tokenizer_summary_page_state['analysis_results'] = results
            st.success(f"✅ Analysis completed for {len(results)} tokenizers")
            
            log_user_action("tokenizer_analysis_completed", 
                          num_models=len(results), 
                          text_length=len(test_text))
    
    def run_algorithm_comparison(self, selected_models: List[str], test_text: str):
        """Run algorithm comparison for selected models."""
        with st.spinner("Comparing tokenizer algorithms..."):
            try:
                results = tokenizer_summary_manager.compare_tokenizers(test_text, selected_models)
                st.session_state.tokenizer_summary_page_state['comparison_results'] = results
                st.success(f"✅ Comparison completed for {len(results)} tokenizers")
                
                log_user_action("tokenizer_comparison_completed", 
                              num_models=len(results), 
                              text_length=len(test_text))
                
            except Exception as e:
                st.error(f"Algorithm comparison failed: {str(e)}")
    
    def render_analysis_results(self):
        """Render tokenization analysis results."""
        results = st.session_state.tokenizer_summary_page_state.get('analysis_results', {})
        
        if not results:
            return
        
        st.markdown("### 📊 Tokenization Analysis Results")
        
        # Create comparison table
        comparison_data = []
        for model_id, result in results.items():
            if result.success:
                comparison_data.append({
                    'Model': model_id,
                    'Algorithm': result.tokenizer_info.algorithm,
                    'Tokens': result.token_count,
                    'Subword Ratio': f"{result.subword_ratio:.2f}",
                    'Unknown Tokens': len(result.unknown_tokens),
                    'Vocabulary Coverage': f"{result.vocabulary_coverage:.2%}",
                    'Special Tokens': result.special_token_count,
                    'Analysis Time': f"{result.analysis_time:.3f}s"
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Detailed results for each tokenizer
            for model_id, result in results.items():
                if result.success:
                    self.render_detailed_result(model_id, result)
    
    def render_detailed_result(self, model_id: str, result: TokenizationAnalysisResult):
        """Render detailed analysis result for a specific tokenizer."""
        with st.expander(f"🔍 Detailed Analysis: {model_id}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Tokenizer Information:**")
                st.write(f"**Type:** {result.tokenizer_info.tokenizer_type}")
                st.write(f"**Algorithm:** {result.tokenizer_info.algorithm}")
                st.write(f"**Vocabulary Size:** {result.tokenizer_info.vocabulary_size:,}")
                st.write(f"**Base Vocabulary:** {result.tokenizer_info.base_vocabulary_size:,}")
                st.write(f"**Merge Rules:** {result.tokenizer_info.merge_rules_count:,}")
                st.write(f"**Language Support:** {', '.join(result.tokenizer_info.language_support)}")
                st.write(f"**Pre-tokenizer:** {result.tokenizer_info.pre_tokenizer}")
            
            with col2:
                st.markdown("**Analysis Results:**")
                st.write(f"**Total Tokens:** {result.token_count}")
                st.write(f"**Subword Ratio:** {result.subword_ratio:.2f}")
                st.write(f"**Unknown Tokens:** {len(result.unknown_tokens)}")
                st.write(f"**Vocabulary Coverage:** {result.vocabulary_coverage:.2%}")
                st.write(f"**Special Tokens:** {result.special_token_count}")
                st.write(f"**Analysis Time:** {result.analysis_time:.3f}s")
            
            # Show tokens
            st.markdown("**Generated Tokens:**")
            st.write(result.tokens)
            
            # Show unknown tokens if any
            if result.unknown_tokens:
                st.markdown("**Unknown Tokens:**")
                st.write(result.unknown_tokens)
            
            # Show special tokens
            if result.tokenizer_info.special_tokens:
                st.markdown("**Special Tokens:**")
                for token_type, token_value in result.tokenizer_info.special_tokens.items():
                    st.write(f"**{token_type}:** {token_value}")
    
    def render_comparison_results(self):
        """Render algorithm comparison results."""
        results = st.session_state.tokenizer_summary_page_state.get('comparison_results')
        
        if not results:
            return
        
        st.markdown("### 🔄 Algorithm Comparison Results")
        
        # Create comparison table
        comparison_data = []
        for model_id, result in results.items():
            if result.success:
                comparison_data.append({
                    'Model': model_id,
                    'Algorithm': result.tokenizer_info.algorithm,
                    'Tokenizer Type': result.tokenizer_info.tokenizer_type,
                    'Vocabulary Size': f"{result.tokenizer_info.vocabulary_size:,}",
                    'Tokens Generated': result.token_count,
                    'Subword Ratio': f"{result.subword_ratio:.2f}",
                    'Unknown Tokens': len(result.unknown_tokens),
                    'Coverage': f"{result.vocabulary_coverage:.2%}",
                    'Analysis Time': f"{result.analysis_time:.3f}s"
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Algorithm insights
            self.render_algorithm_insights(comparison_data)
    
    def render_algorithm_insights(self, comparison_data: List[Dict[str, Any]]):
        """Render algorithm comparison insights."""
        st.markdown("### 💡 Algorithm Insights")
        
        # Group by algorithm
        algorithm_groups = {}
        for row in comparison_data:
            algorithm = row['Algorithm']
            if algorithm not in algorithm_groups:
                algorithm_groups[algorithm] = []
            algorithm_groups[algorithm].append(row)
        
        # Show insights for each algorithm
        for algorithm, models in algorithm_groups.items():
            with st.expander(f"🔍 {algorithm} Analysis"):
                st.write(f"**Models using {algorithm}:** {len(models)}")
                
                # Calculate averages
                avg_tokens = sum(float(row['Tokens Generated']) for row in models) / len(models)
                avg_ratio = sum(float(row['Subword Ratio']) for row in models) / len(models)
                avg_coverage = sum(float(row['Coverage'].rstrip('%')) for row in models) / len(models)
                
                st.write(f"**Average Tokens:** {avg_tokens:.1f}")
                st.write(f"**Average Subword Ratio:** {avg_ratio:.2f}")
                st.write(f"**Average Coverage:** {avg_coverage:.1f}%")
                
                # Show models
                st.write("**Models:**")
                for model in models:
                    st.write(f"- {model['Model']}: {model['Tokens Generated']} tokens, ratio {model['Subword Ratio']}")
    
    def render_performance_stats(self):
        """Render performance statistics."""
        stats = st.session_state.tokenizer_summary_page_state.get('performance_stats')
        
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
                st.metric("Total Analyses", stats['total_analyses'])
                st.metric("Algorithm Comparisons", stats['algorithm_comparisons'])
                st.metric("Total Tokens Analyzed", stats['total_tokens_analyzed'])
                st.metric("Unknown Token Rate", f"{stats['unknown_token_rate']:.2%}")
    
    def render_advanced_options(self):
        """Render advanced analysis options."""
        with st.expander("⚙️ Advanced Analysis Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Analysis Settings:**")
                include_special_tokens = st.checkbox("Include Special Tokens", value=True)
                return_tensors = st.selectbox("Return Tensors", ["None", "pt", "tf", "np"], index=0)
                padding = st.checkbox("Padding", value=False)
                truncation = st.checkbox("Truncation", value=False)
                
            with col2:
                st.markdown("**Display Options:**")
                show_token_ids = st.checkbox("Show Token IDs", value=False)
                show_attention_mask = st.checkbox("Show Attention Mask", value=False)
                show_statistics = st.checkbox("Show Detailed Statistics", value=True)
                
            # Store options
            st.session_state.tokenizer_summary_page_state['advanced_options'] = {
                'include_special_tokens': include_special_tokens,
                'return_tensors': return_tensors if return_tensors != "None" else None,
                'padding': padding,
                'truncation': truncation,
                'show_token_ids': show_token_ids,
                'show_attention_mask': show_attention_mask,
                'show_statistics': show_statistics
            }
    
    def render(self):
        """Render the complete tokenizer summary page."""
        self.render_header()
        
        # Model selection
        selected_models = self.render_model_selection()
        
        # Text input
        test_text = self.render_text_input()
        
        # Analysis controls
        if selected_models and test_text:
            self.render_analysis_controls(selected_models, test_text)
        
        # Advanced options
        self.render_advanced_options()
        
        # Analysis results
        self.render_analysis_results()
        
        # Comparison results
        self.render_comparison_results()
        
        # Performance stats
        self.render_performance_stats()


def render_tokenizer_summary_page():
    """Render function for the tokenizer summary page."""
    try:
        page = TokenizerSummaryPage()
        page.render()
    except Exception as e:
        st.error(f"Error rendering tokenizer summary page: {str(e)}")
        error(f"Failed to render tokenizer summary page", "tokenizer_summary_page", e)


if __name__ == "__main__":
    # For testing
    render_tokenizer_summary_page()
