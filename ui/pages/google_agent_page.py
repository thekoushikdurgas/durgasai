"""
Google Agent Page Module
Integrates llama-cpp-agent with web search capabilities
"""

import streamlit as st
import json
import time
import logging
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import os

# Import necessary components
from utils.logging_utils import log_info, log_error, log_warning
from utils.config_manager import ConfigManager
from utils.google_agent_integration import GoogleAgentIntegration, GoogleAgentModelInterface
from utils.llama_cpp_agent_integration import LlamaCppAgentIntegration
from utils.tool_manager import DurgasAIToolManager

class GoogleAgentPage:
    """Google Agent page implementation with web search capabilities"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.setup_logging()
        self.google_agent = GoogleAgentIntegration(config_manager)
        self.model_interface = GoogleAgentModelInterface(config_manager)
        self.llama_cpp_agent = LlamaCppAgentIntegration(config_manager)
        self.tool_manager = DurgasAIToolManager()
        self.initialize_session_state()
    
    def setup_logging(self):
        """Setup logging for the Google Agent"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_session_state(self):
        """Initialize session state for Google Agent"""
        if "google_agent_history" not in st.session_state:
            st.session_state.google_agent_history = []
        
        if "google_agent_model" not in st.session_state:
            st.session_state.google_agent_model = "gpt-4o"
        
        if "google_agent_temperature" not in st.session_state:
            st.session_state.google_agent_temperature = 0.45
        
        if "google_agent_max_tokens" not in st.session_state:
            st.session_state.google_agent_max_tokens = 2048
        
        if "google_agent_search_mode" not in st.session_state:
            st.session_state.google_agent_search_mode = "advanced"  # advanced, basic, llama-cpp
        
        # Default LLaMA model constant
        self.DEFAULT_LLAMA_MODEL = "Mistral-7B-Instruct-v0.3-Q6_K.gguf"
        if "google_agent_llama_model" not in st.session_state:
            st.session_state.google_agent_llama_model = self.DEFAULT_LLAMA_MODEL
    
    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using trafilatura for content extraction.
        Args:
            url (str): URL to get website content from.
        Returns:
            str: Extracted content including title, main text, and tables.
        """
        try:
            from trafilatura import fetch_url, extract
            
            downloaded = fetch_url(url)
            result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)
            
            if result:
                result = json.loads(result)
                return f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n'
            else:
                return ""
        except Exception as e:
            log_error("Error extracting website content", "google_agent", e)
            return f"An error occurred: {str(e)}"
    
    def get_server_time(self):
        """Get current server time"""
        utc_time = datetime.now(timezone.utc)
        return utc_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def perform_web_search(self, query: str, num_results: int = 10) -> List[dict]:
        """
        Perform web search using the integrated Google Agent
        """
        try:
            return self.google_agent.search_web(query, num_results)
        except Exception as e:
            log_error("Error performing web search", "google_agent", e)
            return []
    
    def process_search_query(self, query: str, model: str, temperature: float, max_tokens: int) -> str:
        """
        Process a search query and generate a response using different modes
        """
        try:
            log_info(f"Processing search query: {query}", "google_agent")
            
            search_mode = st.session_state.get("google_agent_search_mode", "advanced")
            
            if search_mode == "llama-cpp":
                return self.process_with_llama_cpp_agent(query, temperature, max_tokens)
            elif search_mode == "tool":
                return self.process_with_google_agent_tool(query, model, temperature, max_tokens)
            else:
                return self.process_with_basic_integration(query, model, temperature, max_tokens)
            
        except Exception as e:
            log_error("Error processing search query", "google_agent", e)
            return f"An error occurred while processing your query: {str(e)}"
    
    def process_with_llama_cpp_agent(self, query: str, temperature: float, max_tokens: int) -> str:
        """Process query using llama-cpp-agent integration"""
        try:
            llama_model = st.session_state.get("google_agent_llama_model", self.DEFAULT_LLAMA_MODEL)
            
            result = self.llama_cpp_agent.perform_web_search_with_analysis(
                query=query,
                model_name=llama_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if result.get("status") == "success":
                response = result.get("response", "No response generated")
                citations = result.get("citations", [])
                
                if citations:
                    response += "\n\n**Sources:**\n"
                    for i, citation in enumerate(citations[:5], 1):
                        response += f"{i}. {citation}\n"
                
                return response
            else:
                return result.get("response", "Failed to process query with llama-cpp-agent")
                
        except Exception as e:
            log_error("Error processing with llama-cpp-agent", "google_agent", e)
            return f"An error occurred with llama-cpp-agent: {str(e)}"
    
    def process_with_google_agent_tool(self, query: str, model: str, temperature: float, max_tokens: int) -> str:
        """Process query using the Google Agent tool"""
        try:
            # Use the Google Agent tool
            tool_result = self.tool_manager.execute_tool(
                "google_agent_search",
                {
                    "query": query,
                    "num_results": 10,
                    "model": model,
                    "provider": self.google_agent.provider,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "search_provider": "auto"
                }
            )
            
            if tool_result and tool_result.get("status") == "success":
                result_data = json.loads(tool_result.get("result", "{}"))
                return result_data.get("comprehensive_response", "No response generated")
            else:
                error_msg = tool_result.get("error", "Unknown error") if tool_result else "Tool execution failed"
                return f"Tool execution error: {error_msg}"
                
        except Exception as e:
            log_error("Error processing with Google Agent tool", "google_agent", e)
            return f"An error occurred with Google Agent tool: {str(e)}"
    
    def process_with_basic_integration(self, query: str, model: str, temperature: float, max_tokens: int) -> str:
        """Process query using basic integration (fallback)"""
        try:
            # Perform web search using the integration
            search_results = self.perform_web_search(query, num_results=10)
            
            # Format search results for context
            search_context = self.google_agent.format_search_results_for_context(search_results)
            
            # Generate response using model interface
            response = self.model_interface.generate_response(
                query=query,
                search_context=search_context,
                model=model,
                provider=self.google_agent.provider,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Add citations
            citations = self.google_agent.extract_citations_from_results(search_results)
            if citations:
                response += "\n\n**Sources:**\n"
                for i, citation in enumerate(citations[:5], 1):  # Limit to top 5 citations
                    response += f"{i}. {citation}\n"
            
            return response
            
        except Exception as e:
            log_error("Error processing with basic integration", "google_agent", e)
            return f"An error occurred with basic integration: {str(e)}"
    
    def render_configuration_panel(self):
        """Render the configuration panel for Google Agent settings"""
        st.sidebar.markdown("### 🔧 Google Agent Configuration")
        
        # Search mode selection
        search_modes = {
            "advanced": "Advanced Integration (Recommended)",
            "tool": "Google Agent Tool",
            "llama-cpp": "LLaMA-CPP-Agent",
            "basic": "Basic Integration"
        }
        
        st.session_state.google_agent_search_mode = st.sidebar.selectbox(
            "Search Mode",
            options=list(search_modes.keys()),
            format_func=lambda x: search_modes[x],
            index=list(search_modes.keys()).index(st.session_state.google_agent_search_mode) if st.session_state.google_agent_search_mode in search_modes else 0,
            help="Choose the search and analysis method"
        )
        
        # Model selection (for non-llama-cpp modes)
        if st.session_state.google_agent_search_mode != "llama-cpp":
            llm_provider = st.sidebar.selectbox(
                "LLM Provider",
                ["openai", "openrouter","cohere","nvidia", "anthropic", "google", "deepseek", "groq", "mistral", "perplexity", "huggingface", "tavily", "web_search", "exa", "serpapi", "github"],
                index=0
            )
            # Get models dynamically from config manager
            models = self.config_manager.get_provider_models(llm_provider)
            
            selected_model = st.sidebar.selectbox("Model", [model.split("/")[1] if "/" in model else model for model in models])
            st.session_state.google_agent_model = selected_model
            
            # Setup configuration with both provider and model
            self.google_agent.setup_configuration(llm_provider, selected_model)
        else:
            # LLaMA model selection
            available_llama_models = self.llama_cpp_agent.get_available_models()
            if not available_llama_models:
                st.sidebar.warning("⚠️ No LLaMA models found. Click 'Download Models' below.")
                available_llama_models = [self.DEFAULT_LLAMA_MODEL]  # Default fallback
            
            st.session_state.google_agent_llama_model = st.sidebar.selectbox(
                "LLaMA Model",
                available_llama_models,
                index=available_llama_models.index(st.session_state.google_agent_llama_model) if st.session_state.google_agent_llama_model in available_llama_models else 0
            )
            
            # Model management buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("📥 Download Models", help="Download default LLaMA models"):
                    with st.spinner("Downloading models..."):
                        success = self.llama_cpp_agent.download_default_models()
                        if success:
                            st.success("✅ Models downloaded!")
                            st.rerun()
                        else:
                            st.error("❌ Download failed")
            
            with col2:
                if st.button("🔄 Refresh", help="Refresh available models"):
                    st.rerun()
        
        # Temperature slider
        st.session_state.google_agent_temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.google_agent_temperature,
            step=0.1,
            help="Controls randomness in responses. Lower values are more focused."
        )
        
        # Max tokens slider
        st.session_state.google_agent_max_tokens = st.sidebar.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=st.session_state.google_agent_max_tokens,
            step=256,
            help="Maximum length of the response"
        )
        
        # Additional settings
        with st.sidebar.expander("Advanced Settings"):
            search_depth = st.slider("Search Depth", 1, 10, 5, help="Number of search results to analyze")
            include_citations = st.checkbox("Include Citations", True, help="Include source citations in responses")
            summarize_results = st.checkbox("Summarize Results", True, help="Provide summary of findings")
            
            # Store advanced settings in session state
            st.session_state.search_depth = search_depth
            st.session_state.include_citations = include_citations
            st.session_state.summarize_results = summarize_results
        
        # Configuration status
        with st.sidebar.expander("Configuration Status"):
            # Basic configuration
            validation_results = self.google_agent.validate_configuration()
            
            for feature, status in validation_results.items():
                status_icon = "✅" if status else "❌"
                feature_name = feature.replace("_", " ").title()
                st.write(f"{status_icon} {feature_name}")
            
            # LLaMA-CPP-Agent status
            st.write("---")
            st.write("**LLaMA-CPP-Agent Status:**")
            llama_validation = self.llama_cpp_agent.validate_configuration()
            
            llama_status_icon = "✅" if llama_validation.get("llama_cpp_available") else "❌"
            st.write(f"{llama_status_icon} LLaMA-CPP Available")
            
            models_count = len(llama_validation.get("available_models", []))
            models_icon = "✅" if models_count > 0 else "❌"
            st.write(f"{models_icon} Models Available ({models_count})")
            
            # Overall status
            st.write("---")
            search_apis_available = any([
                validation_results.get("serpapi_configured"), 
                validation_results.get("tavily_configured")
            ])
            
            if search_apis_available and llama_validation.get("llama_cpp_available"):
                st.success("🎉 Full functionality available!")
            elif search_apis_available:
                st.info("ℹ️ Web search available (no local models)")
            elif llama_validation.get("llama_cpp_available"):
                st.info("ℹ️ Local models available (limited web search)")
            else:
                st.warning("⚠️ Limited functionality - configure APIs or download models")
    
    def render_chat_interface(self):
        """Render the main chat interface for Google Agent"""
        st.markdown("## 🌐 Google Agent - Web Search Assistant")
        st.markdown("Ask me anything and I'll search the web to provide comprehensive, up-to-date answers with citations.")
        
        # Display example queries
        st.markdown("### 💡 Example Queries:")
        example_cols = st.columns(2)
        
        with example_cols[0]:
            if st.button("Latest news about AI developments", key="example1"):
                st.session_state.google_agent_input = "Latest news about AI developments"
            if st.button("Best practices for Python web development", key="example2"):
                st.session_state.google_agent_input = "Best practices for Python web development"
        
        with example_cols[1]:
            if st.button("Current weather in major cities", key="example3"):
                st.session_state.google_agent_input = "Current weather in major cities"
            if st.button("Recent scientific breakthroughs 2024", key="example4"):
                st.session_state.google_agent_input = "Recent scientific breakthroughs 2024"
        
        # Chat history display
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.google_agent_history):
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # Input area
        query_input = st.chat_input(
            "Ask me anything... I'll search the web for the latest information!",
            key="google_agent_chat_input"
        )
        
        # Handle manual input from examples
        if "google_agent_input" in st.session_state and st.session_state.google_agent_input:
            query_input = st.session_state.google_agent_input
            st.session_state.google_agent_input = ""
        
        if query_input:
            # Add user message to history
            st.session_state.google_agent_history.append({
                "role": "user",
                "content": query_input,
                "timestamp": self.get_server_time()
            })
            
            # Display user message
            st.chat_message("user").write(query_input)
            
            # Process query and generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching the web and analyzing results..."):
                    response = self.process_search_query(
                        query_input,
                        st.session_state.google_agent_model,
                        st.session_state.google_agent_temperature,
                        st.session_state.google_agent_max_tokens
                    )
                
                st.write(response)
                
                # Add assistant response to history
                st.session_state.google_agent_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": self.get_server_time()
                })
    
    def render_history_management(self):
        """Render history management controls"""
        if st.session_state.google_agent_history:
            col1, col2, _ = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🗑️ Clear History", key="clear_google_history"):
                    st.session_state.google_agent_history = []
                    st.rerun()
            
            with col2:
                if st.button("📥 Export Chat", key="export_google_chat"):
                    chat_data = {
                        "export_time": self.get_server_time(),
                        "model": st.session_state.google_agent_model,
                        "history": st.session_state.google_agent_history
                    }
                    st.download_button(
                        "💾 Download JSON",
                        data=json.dumps(chat_data, indent=2),
                        file_name=f"google_agent_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    def render_info_panel(self):
        """Render information panel about Google Agent capabilities"""
        with st.expander("ℹ️ About Google Agent"):
            st.markdown("""
            **Google Agent** is a web-search powered AI assistant that can:
            
            🔍 **Search Capabilities:**
            - Real-time web search across multiple sources
            - Content extraction and summarization
            - Citation and source tracking
            - Multi-language support
            
            🤖 **AI Features:**
            - Multiple language model support
            - Configurable response parameters
            - Structured output formatting
            - Context-aware responses
            
            📊 **Research Tools:**
            - Comprehensive analysis of search results
            - Source credibility assessment
            - Fact-checking and verification
            - Trend analysis and insights
            
            **Note:** This is a demonstration interface showing the integration framework. 
            Full functionality requires additional API configurations and dependencies.
            """)
    
    def render(self):
        """Render the complete Google Agent page"""
        try:
            # Configuration panel in sidebar
            self.render_configuration_panel()
            
            # Main chat interface
            self.render_chat_interface()
            
            # History management
            self.render_history_management()
            
            # Information panel
            self.render_info_panel()
            
            # Status information
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Status")
            st.sidebar.info(f"Messages: {len(st.session_state.google_agent_history)}")
            st.sidebar.info(f"Model: {st.session_state.google_agent_model}")
            st.sidebar.info(f"Last Update: {self.get_server_time()}")
            
        except Exception as e:
            log_error("Error rendering Google Agent page", "google_agent", e)
            st.error(f"An error occurred while rendering the Google Agent page: {str(e)}")
            st.exception(e)
