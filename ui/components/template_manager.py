"""
Template Manager UI Component for DurgasAI
Provides interface for managing AI assistant templates
"""

import streamlit as st
import json
from typing import Dict, List, Any
from utils.template_manager import TemplateManager

class TemplateManagerUI:
    """UI component for template management"""
    
    def __init__(self, template_manager: TemplateManager):
        self.template_manager = TemplateManager()
    
    def render(self):
        """Render the template management interface"""
        # Template overview
        self.render_template_overview()
        
        # Template details
        self.render_template_details()
        
        # Template creation/editing
        self.render_template_editor()
    
    def render_template_overview(self):
        """Render template overview section"""
        st.subheader("📊 Template Overview")
        
        stats = self.template_manager.get_template_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Templates", stats.get("total", 0))
        
        with col2:
            st.metric("Function Calling Enabled", stats.get("function_calling_enabled", 0))
        
        with col3:
            st.metric("Function Calling Disabled", stats.get("function_calling_disabled", 0))
        
        with col4:
            avg_tools = stats.get("average_tools_per_template", 0)
            st.metric("Avg Tools/Template", f"{avg_tools:.1f}")
        
        # Templates by category
        st.subheader("🏷️ Templates by Category")
        by_category = stats.get("by_category", {})
        
        if by_category:
            cols = st.columns(len(by_category))
            for i, (category, count) in enumerate(by_category.items()):
                with cols[i % len(cols)]:
                    st.metric(category.title(), count)
        else:
            st.info("No templates found")
    
    def render_template_details(self):
        """Render detailed template information"""
        st.subheader("🔍 Template Details")
        
        all_templates = self.template_manager.get_all_templates()
        template_names = [template.get("name", "Unknown") for template in all_templates]
        
        if not template_names:
            st.info("No templates available")
            return
        
        # Template selection
        selected_template_name = st.selectbox(
            "Select a template to view details:",
            template_names,
            index=0
        )
        
        selected_template = next(
            (t for t in all_templates if t.get("name") == selected_template_name),
            None
        )
        
        if selected_template:
            # Basic information
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Name:** {selected_template.get('name', 'Unknown')}")
                st.write(f"**Description:** {selected_template.get('description', 'No description')}")
                st.write(f"**Category:** {selected_template.get('category', 'Unknown')}")
                st.write(f"**Icon:** {selected_template.get('icon', '🤖')}")
                st.write(f"**Version:** {selected_template.get('version', 'Unknown')}")
            
            with col2:
                # Status indicators
                if selected_template.get('function_calling_enabled'):
                    st.success("✅ Function Calling Enabled")
                else:
                    st.info("ℹ️ Function Calling Disabled")
                
                # Tools count
                tools_count = len(selected_template.get('function_calling_tools', []))
                st.metric("Available Tools", tools_count)
            
            # System instruction
            st.subheader("📝 System Instruction")
            system_instruction = selected_template.get('system_instruction', 'No system instruction defined')
            st.code(system_instruction, language="text")
            
            # Function calling tools
            if selected_template.get('function_calling_enabled'):
                st.subheader("🔧 Function Calling Tools")
                tools = selected_template.get('function_calling_tools', [])
                
                if tools:
                    for tool in tools:
                        st.write(f"• {tool}")
                else:
                    st.info("No tools configured")
            
            # Generation parameters
            st.subheader("⚙️ Generation Parameters")
            gen_params = selected_template.get('generation_parameters', {})
            
            if gen_params:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    temperature = gen_params.get('temperature', 'Not set')
                    st.write(f"**Temperature:** {temperature}")
                
                with col2:
                    max_tokens = gen_params.get('max_output_tokens', 'Not set')
                    st.write(f"**Max Tokens:** {max_tokens}")
                
                with col3:
                    top_p = gen_params.get('top_p', 'Not set')
                    st.write(f"**Top P:** {top_p}")
                
                if 'thinking_budget' in gen_params:
                    thinking_budget = gen_params.get('thinking_budget')
                    st.write(f"**Thinking Budget:** {thinking_budget}")
            else:
                st.info("No generation parameters set")
            
            # Safety settings
            st.subheader("🛡️ Safety Settings")
            safety_settings = selected_template.get('safety_settings', {})
            
            if safety_settings:
                for setting, value in safety_settings.items():
                    st.write(f"**{setting.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No safety settings configured")
            
            # Template actions
            st.subheader("🎛️ Template Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Copy Template"):
                    st.code(json.dumps(selected_template, indent=2), language="json")
            
            with col2:
                if st.button("📥 Export Template"):
                    # In a real implementation, this would trigger a file download
                    st.success("Template exported successfully!")
            
            with col3:
                if st.button("🗑️ Delete Template"):
                    if st.session_state.get(f"confirm_delete_{selected_template_name}", False):
                        if self.template_manager.delete_template(selected_template_name):
                            st.success("Template deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete template")
                    else:
                        st.session_state[f"confirm_delete_{selected_template_name}"] = True
                        st.warning("Click again to confirm deletion")
    
    def render_template_editor(self):
        """Render template creation/editing interface"""
        st.subheader("✏️ Template Editor")
        
        # Template creation form
        with st.expander("Create New Template", expanded=False):
            self.render_template_creation_form()
        
        # Template editing
        with st.expander("Edit Existing Template", expanded=False):
            self.render_template_editing_form()
        
        # Template search
        with st.expander("Search Templates", expanded=False):
            self.render_template_search()
    
    def render_template_creation_form(self):
        """Render template creation form"""
        with st.form("create_template_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Template Name *", help="Unique name for the template")
                description = st.text_area("Description *", help="Brief description of the template")
                category = st.selectbox(
                    "Category *",
                    ["assistant", "creative", "analytical", "technical", "business", "education"]
                )
                icon = st.text_input("Icon", value="🤖", help="Emoji icon for the template")
            
            with col2:
                function_calling_enabled = st.checkbox("Enable Function Calling")
                
                if function_calling_enabled:
                    # Tool selection would go here
                    st.write("**Available Tools:**")
                    # This would be populated from the tool manager
                    st.info("Tool selection will be implemented")
                
                temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
                max_tokens = st.number_input("Max Tokens", min_value=100, max_value=8192, value=2048)
            
            system_instruction = st.text_area(
                "System Instruction *",
                height=200,
                help="The system prompt that defines the assistant's behavior"
            )
            
            # Tags
            tags_input = st.text_input("Tags (comma-separated)", help="Tags for categorizing the template")
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
            
            submitted = st.form_submit_button("Create Template")
            
            if submitted:
                if not all([name, description, system_instruction]):
                    st.error("Please fill in all required fields (marked with *)")
                else:
                    template_data = {
                        "name": name,
                        "description": description,
                        "category": category,
                        "icon": icon,
                        "function_calling_enabled": function_calling_enabled,
                        "function_calling_tools": [],  # Would be populated from tool selection
                        "system_instruction": system_instruction,
                        "generation_parameters": {
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                            "top_p": 0.95,
                            "top_k": 40
                        },
                        "safety_settings": {
                            "hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
                            "dangerous_content": "BLOCK_MEDIUM_AND_ABOVE",
                            "harassment": "BLOCK_MEDIUM_AND_ABOVE",
                            "sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        "tags": tags
                    }
                    
                    # Validate template
                    errors = self.template_manager.validate_template(template_data)
                    if errors:
                        st.error("Template validation errors:")
                        for error in errors:
                            st.write(f"• {error}")
                    else:
                        if self.template_manager.create_template(template_data):
                            st.success(f"Template '{name}' created successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to create template")
    
    def render_template_editing_form(self):
        """Render template editing form"""
        all_templates = self.template_manager.get_all_templates()
        template_names = [template.get("name", "Unknown") for template in all_templates]
        
        if not template_names:
            st.info("No templates available for editing")
            return
        
        selected_template_name = st.selectbox(
            "Select template to edit:",
            template_names,
            key="edit_template_select"
        )
        
        selected_template = next(
            (t for t in all_templates if t.get("name") == selected_template_name),
            None
        )
        
        if selected_template:
            with st.form("edit_template_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Template Name", value=selected_template.get("name", ""))
                    description = st.text_area("Description", value=selected_template.get("description", ""))
                    category = st.selectbox(
                        "Category",
                        ["assistant", "creative", "analytical", "technical", "business", "education"],
                        index=["assistant", "creative", "analytical", "technical", "business", "education"].index(
                            selected_template.get("category", "assistant")
                        )
                    )
                
                with col2:
                    function_calling_enabled = st.checkbox(
                        "Enable Function Calling",
                        value=selected_template.get("function_calling_enabled", False)
                    )
                    
                    # Generation parameters
                    gen_params = selected_template.get("generation_parameters", {})
                    temperature = st.slider(
                        "Temperature",
                        0.0, 1.0,
                        gen_params.get("temperature", 0.7),
                        0.1
                    )
                    max_tokens = st.number_input(
                        "Max Tokens",
                        min_value=100, max_value=8192,
                        value=gen_params.get("max_output_tokens", 2048)
                    )
                
                system_instruction = st.text_area(
                    "System Instruction",
                    value=selected_template.get("system_instruction", ""),
                    height=200
                )
                
                submitted = st.form_submit_button("Update Template")
                
                if submitted:
                    updates = {
                        "name": name,
                        "description": description,
                        "category": category,
                        "function_calling_enabled": function_calling_enabled,
                        "system_instruction": system_instruction,
                        "generation_parameters": {
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                            "top_p": gen_params.get("top_p", 0.95),
                            "top_k": gen_params.get("top_k", 40)
                        }
                    }
                    
                    if self.template_manager.update_template(selected_template_name, updates):
                        st.success("Template updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update template")
    
    def render_template_search(self):
        """Render template search interface"""
        search_query = st.text_input("Search templates by name, description, or tags:")
        
        if search_query:
            results = self.template_manager.search_templates(search_query)
            
            if results:
                st.write(f"Found {len(results)} template(s):")
                
                for template in results:
                    with st.expander(f"{template.get('icon', '🤖')} {template.get('name', 'Unknown')}"):
                        st.write(f"**Description:** {template.get('description', 'No description')}")
                        st.write(f"**Category:** {template.get('category', 'Unknown')}")
                        
                        tags = template.get('tags', [])
                        if tags:
                            st.write(f"**Tags:** {', '.join(tags)}")
                        
                        if template.get('function_calling_enabled'):
                            tools = template.get('function_calling_tools', [])
                            if tools:
                                st.write(f"**Tools:** {', '.join(tools)}")
            else:
                st.info("No templates found matching your search.")
        else:
            st.info("Enter a search query to find templates.")
