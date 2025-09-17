"""
Sidebar Component
Handles the sidebar navigation and configuration
"""

import streamlit as st
from utils.config_manager import ConfigManager
from ui.components.template_manager import TemplateManager

class Sidebar:
    """Sidebar component for navigation and configuration"""
    
    def __init__(self, template_manager: TemplateManager, config_manager :ConfigManager):
        self.template_manager = template_manager
        self.config_manager = config_manager
    
    def render(self):
        """Render the sidebar with navigation and configuration"""
        with st.sidebar:
            # # Page navigation
            # pages = ["Chat", "Tools", "Workflows", "Templates", "Google Agent", "System Monitor", "Settings"]
            # selected_page = st.selectbox(
            #     "Select Page",
            #     pages,
            #     index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
            # )
            # st.session_state.current_page = selected_page
            
            # Model selection
            llm_provider = st.selectbox(
                "LLM Provider",
                ["openai", "openrouter","cohere","nvidia", "anthropic", "google", "deepseek", "groq", "mistral", "perplexity", "huggingface", "tavily", "web_search", "exa", "serpapi", "github"],
                index=0
            )
            # Get models dynamically from config manager
            models = self.config_manager.get_provider_models(llm_provider)
            
            selected_model = st.selectbox("Model", [model.split("/")[1] if "/" in model else model for model in models])
            
            # # Add refresh button for dynamic model loading
            # if self.config_manager and llm_provider in ["openai", "anthropic", "google", "groq"]:
            #     _, col2 = st.columns([3, 1])
            #     with col2:
            #         if st.button("🔄", help="Refresh models from API", key="refresh_models"):
            #             # Clear cached models and refetch
            #             if llm_provider in st.session_state.get("provider_models", {}):
            #                 del st.session_state.provider_models[llm_provider]
            #             # Trigger rerun to fetch fresh models
            #             st.rerun()
            
            # Store selections in session state
            st.session_state.llm_provider = llm_provider
            st.session_state.selected_model = selected_model
            
            # Template selection
            self.render_template_selector()
            
            # Workflow type selection
            self.render_workflow_selector()
            if st.button("🔄 Reset Chat"):
                st.session_state.chat_history = []
                st.session_state.chat_session = None
                st.rerun()
    
    def render_template_selector(self):
        """Render template selection interface"""
        templates = self.template_manager.get_all_templates()
        template_names = [template.get("name", "Unknown") for template in templates]
            
        selected_template_name = st.selectbox(
                "Choose your AI assistant personality:",
                template_names,
                index=template_names.index(st.session_state.selected_template) if st.session_state.selected_template in template_names else 0
        )
            
        if selected_template_name != st.session_state.selected_template:
            st.session_state.selected_template = selected_template_name
            # Reset chat session when template changes
            st.session_state.chat_session = None
            st.rerun()
        
        # Display selected template info
        selected_template = next(
            (t for t in templates if t.get("name") == selected_template_name), 
            None
        )
        
        if selected_template:
            with st.expander("Template Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {selected_template.get('name', 'Unknown')}")
                    st.write(f"**Category:** {selected_template.get('category', 'Unknown')}")
                    st.write(f"**Icon:** {selected_template.get('icon', '🤖')}")
                with col2:
                    st.write(f"**Function Calling:** {'Enabled' if selected_template.get('function_calling_enabled') else 'Disabled'}")
                    st.write(f"**Tools:** {len(selected_template.get('function_calling_tools', []))}")
                
                st.write(f"**Description:** {selected_template.get('description', 'No description available')}")
                
                if selected_template.get('system_instruction'):
                    st.write("**System Instruction:**")
                    st.code(selected_template.get('system_instruction'), language="text")
    
    def render_workflow_selector(self):
        """Render workflow type selection interface"""
        # st.subheader("⚙️ Workflow Configuration")
        
        # Get workflow templates from selected template
        selected_template_name = st.session_state.get("selected_template", "Default Assistant")
        workflow_templates = self.template_manager.get_workflow_templates(selected_template_name)
        
        if workflow_templates:
            workflow_names = [template["name"] for template in workflow_templates.values()]
            workflow_keys = list(workflow_templates.keys())
            
            # Set default workflow type if not set
            if "selected_workflow_type" not in st.session_state:
                st.session_state.selected_workflow_type = workflow_keys[0] if workflow_keys else "basic_chat"
            
            # Ensure selected workflow exists in current template
            if st.session_state.selected_workflow_type not in workflow_keys:
                st.session_state.selected_workflow_type = workflow_keys[0] if workflow_keys else "basic_chat"
            
            # Get current selection index
            current_index = 0
            if st.session_state.selected_workflow_type in workflow_keys:
                current_index = workflow_keys.index(st.session_state.selected_workflow_type)
            
            selected_workflow_name = st.selectbox(
                "Choose workflow type:",
                workflow_names,
                index=current_index,
                help="Select the type of workflow for your AI assistant"
            )
            
            # Find the corresponding workflow key
            selected_workflow_key = workflow_keys[workflow_names.index(selected_workflow_name)]
            
            if selected_workflow_key != st.session_state.get("selected_workflow_type"):
                st.session_state.selected_workflow_type = selected_workflow_key
                # Reset chat session when workflow changes
                st.session_state.chat_session = None
                st.rerun()
            
            # Display workflow details
            workflow_info = workflow_templates[selected_workflow_key]
            with st.expander("Workflow Details"):
                st.write(f"**Name:** {workflow_info['name']}")
                st.write(f"**Description:** {workflow_info['description']}")
                st.write("**Features:**")
                for feature in workflow_info.get('features', []):
                    st.write(f"• {feature}")
        else:
            st.info("No workflow templates available in selected template")
        
        # Advanced workflow options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                max_iterations = st.number_input(
                    "Max Iterations",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Maximum number of workflow iterations"
                )
                st.session_state.max_iterations = max_iterations
            
            with col2:
                error_threshold = st.number_input(
                    "Error Threshold",
                    min_value=0,
                    max_value=10,
                    value=3,
                    help="Maximum number of errors before stopping"
                )
                st.session_state.error_threshold = error_threshold
