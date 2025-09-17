"""
Tool Manager UI Component for DurgasAI
Provides interface for managing and configuring tools
"""

import streamlit as st
import json
from typing import Dict, List, Any
from utils.tool_manager import DurgasAIToolManager

class ToolManager:
    """UI component for tool management"""
    
    def __init__(self, tool_manager: DurgasAIToolManager):
        self.tool_manager = tool_manager
    
    def render(self):
        """Render the tool management interface"""
        # Tool overview
        self.render_tool_overview()
        
        # Tool details
        self.render_tool_details()
        
        # Tool configuration
        self.render_tool_configuration()
    
    def render_tool_overview(self):
        """Render tool overview section"""
        st.subheader("📊 Tool Overview")
        
        all_tools = self.tool_manager.get_all_tools()
        enabled_tools = self.tool_manager.get_enabled_tools()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tools", len(all_tools))
        
        with col2:
            st.metric("Enabled Tools", len(enabled_tools))
        
        with col3:
            categories = set()
            for tool in all_tools.values():
                category = tool.get("category")
                if category:
                    categories.add(category)
            st.metric("Categories", len(categories))
        
        with col4:
            tools_with_functions = sum(
                1 for tool_name in all_tools.keys()
                if self.tool_manager.get_tool_function(tool_name) is not None
            )
            st.metric("With Functions", tools_with_functions)
        
        # Tool categories
        st.subheader("🏷️ Tools by Category")
        categories = {}
        for tool_name, tool_def in all_tools.items():
            category = tool_def.get("category", "uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(tool_name)
        
        for category, tools in categories.items():
            with st.expander(f"{category.title()} ({len(tools)} tools)"):
                for tool_name in sorted(tools):
                    tool_def = all_tools[tool_name]
                    status = "✅" if tool_def.get("status") == "enabled" else "❌"
                    st.write(f"{status} **{tool_name}** - {tool_def.get('description', 'No description')}")
    
    def render_tool_details(self):
        """Render detailed tool information"""
        st.subheader("🔍 Tool Details")
        
        all_tools = self.tool_manager.get_all_tools()
        tool_names = list(all_tools.keys())
        
        if not tool_names:
            st.info("No tools available")
            return
        
        selected_tool = st.selectbox(
            "Select a tool to view details:",
            tool_names,
            index=0
        )
        
        if selected_tool:
            tool_info = self.tool_manager.get_tool_info(selected_tool)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Name:** {tool_info.get('name', 'Unknown')}")
                st.write(f"**Description:** {tool_info.get('description', 'No description')}")
                st.write(f"**Category:** {tool_info.get('category', 'Unknown')}")
                st.write(f"**Version:** {tool_info.get('version', 'Unknown')}")
                st.write(f"**Author:** {tool_info.get('author', 'Unknown')}")
                st.write(f"**Status:** {tool_info.get('status', 'Unknown')}")
                st.write(f"**Security Level:** {tool_info.get('security_level', 'Unknown')}")
            
            with col2:
                if tool_info.get('has_function'):
                    st.success("✅ Function Loaded")
                else:
                    st.error("❌ Function Not Available")
                
                # Tool execution test
                if st.button("🧪 Test Tool"):
                    self.test_tool(selected_tool)
            
            # Parameters
            st.subheader("📝 Parameters")
            parameters = tool_info.get('parameters', [])
            
            if parameters:
                for param in parameters:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{param.get('name', 'Unknown')}**")
                    with col2:
                        st.write(f"Type: {param.get('type', 'Unknown')}")
                    with col3:
                        required = "Required" if param.get('required') else "Optional"
                        st.write(required)
                    st.write(f"Description: {param.get('description', 'No description')}")
                    st.write("---")
            else:
                st.info("No parameters defined")
            
            # Output parameters
            st.subheader("📤 Output Parameters")
            output_params = tool_info.get('output_parameters', [])
            
            if output_params:
                for param in output_params:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{param.get('name', 'Unknown')}**")
                    with col2:
                        st.write(f"Type: {param.get('type', 'Unknown')}")
                    st.write(f"Description: {param.get('description', 'No description')}")
                    st.write("---")
            else:
                st.info("No output parameters defined")
            
            # Tags
            tags = tool_info.get('tags', [])
            if tags:
                st.subheader("🏷️ Tags")
                tag_string = " ".join([f"`{tag}`" for tag in tags])
                st.markdown(tag_string)
    
    def render_tool_configuration(self):
        """Render tool configuration interface"""
        st.subheader("⚙️ Tool Configuration")
        
        all_tools = self.tool_manager.get_all_tools()
        
        # Tool status management
        st.write("**Enable/Disable Tools:**")
        
        for tool_name, tool_def in all_tools.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{tool_name}** - {tool_def.get('description', 'No description')}")
            
            with col2:
                current_status = tool_def.get('status', 'disabled')
                new_status = st.selectbox(
                    "Status",
                    ["enabled", "disabled"],
                    index=0 if current_status == "enabled" else 1,
                    key=f"status_{tool_name}"
                )
                
                if new_status != current_status:
                    if st.button(f"Update", key=f"update_{tool_name}"):
                        self.update_tool_status(tool_name, new_status)
        
        # Tool reload
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload All Tools"):
                self.tool_manager.reload_tools()
                st.success("Tools reloaded successfully!")
                st.rerun()
        
        with col2:
            if st.button("📊 Refresh Statistics"):
                st.rerun()
    
    def test_tool(self, tool_name: str = None):
        """Test a tool with sample inputs"""
        if tool_name is None:
            st.error("No tool name provided for testing")
            return
        
        try:
            tool_info = self.tool_manager.get_tool_info(tool_name)
            parameters = tool_info.get('parameters', [])
            
            if not parameters:
                # Test tool with no parameters
                result = self.tool_manager.execute_tool(tool_name)
                if result.success:
                    # st.success("Tool executed successfully!")
                    st.subheader("Execution Details:")
                    st.write(f"**Success:** {result.success}")
                    st.write(f"**Execution Time:** {result.execution_time:.2f}s")
                    st.write(f"**Tool Name:** {result.tool_name}")
                    st.subheader("Result:")
                    
                    # Display formatted result if available
                    if hasattr(result, 'formatted_result') and result.formatted_result:
                        self._display_formatted_result(result.formatted_result)
                    else:
                        st.json(result.result)
                else:
                    st.error("Tool execution failed!")
                    st.subheader("Error Details:")
                    st.write(f"**Success:** {result.success}")
                    st.write(f"**Execution Time:** {result.execution_time:.2f}s")
                    st.write(f"**Tool Name:** {result.tool_name}")
                    st.write(f"**Error:** {result.error}")
                return
            
            # Collect parameters from user
            kwargs = {}
            for param in parameters:
                param_name = param.get('name')
                param_type = param.get('type', 'string')
                param_desc = param.get('description', '')
                required = param.get('required', False)
                item_type = param.get('item_type', 'string')
                
                # Create a container for each parameter
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if param_type == "string":
                            value = st.text_input(
                                f"{param_name} {'(required)' if required else '(optional)'}",
                                key=f"test_{tool_name}_{param_name}",
                                help=param_desc
                            )
                            if value:
                                kwargs[param_name] = value
                        
                        elif param_type == "integer":
                            value = st.number_input(
                                f"{param_name} {'(required)' if required else '(optional)'}",
                                key=f"test_{tool_name}_{param_name}",
                                help=param_desc,
                                step=1
                            )
                            if value is not None:
                                kwargs[param_name] = int(value)
                        
                        elif param_type == "number":
                            value = st.number_input(
                                f"{param_name} {'(required)' if required else '(optional)'}",
                                key=f"test_{tool_name}_{param_name}",
                                help=param_desc,
                                step=0.1
                            )
                            if value is not None:
                                kwargs[param_name] = value
                        
                        elif param_type == "boolean":
                            value = st.checkbox(
                                f"{param_name} {'(required)' if required else '(optional)'}",
                                key=f"test_{tool_name}_{param_name}",
                                help=param_desc
                            )
                            kwargs[param_name] = value
                        
                        elif param_type == "array":
                            if item_type == "string":
                                value = st.text_area(
                                    f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array of strings",
                                    key=f"test_{tool_name}_{param_name}",
                                    help=f"{param_desc} (e.g., [\"item1\", \"item2\", \"item3\"])",
                                    height=100
                                )
                            elif item_type == "number":
                                value = st.text_area(
                                    f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array of numbers",
                                    key=f"test_{tool_name}_{param_name}",
                                    help=f"{param_desc} (e.g., [1, 2, 3, 4, 5])",
                                    height=100
                                )
                            elif item_type == "boolean":
                                value = st.text_area(
                                    f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array of booleans",
                                    key=f"test_{tool_name}_{param_name}",
                                    help=f"{param_desc} (e.g., [true, false, true])",
                                    height=100
                                )
                            else:
                                value = st.text_area(
                                    f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array",
                                    key=f"test_{tool_name}_{param_name}",
                                    help=f"{param_desc} (JSON array format)",
                                    height=100
                                )
                            
                            if value:
                                try:
                                    kwargs[param_name] = json.loads(value)
                                except json.JSONDecodeError as e:
                                    st.error(f"Invalid JSON for {param_name}: {str(e)}")
                                    return
                        
                        elif param_type == "file":
                            value = st.file_uploader(
                                f"{param_name} {'(required)' if required else '(optional)'}",
                                key=f"test_{tool_name}_{param_name}",
                                help=param_desc
                            )
                            if value:
                                # Convert file upload to dict format expected by tools
                                file_content = value.read()
                                kwargs[param_name] = {
                                    "name": value.name,
                                    "content": file_content,
                                    "type": value.type,
                                    "size": value.size
                                }
                        
                        else:
                            value = st.text_input(
                                f"{param_name} {'(required)' if required else '(optional)'}",
                                key=f"test_{tool_name}_{param_name}",
                                help=param_desc
                            )
                            if value:
                                kwargs[param_name] = value
                    
                    with col2:
                        st.write(f"**Type:** {param_type}")
                        if param_type == "array":
                            st.write(f"**Item Type:** {item_type}")
                        st.write(f"**Required:** {'Yes' if required else 'No'}")
                    
                    # Add separator between parameters
                    st.divider()
            
            # Execute tool
            if st.button(f"Execute {tool_name}"):
                try:
                    # Validate inputs
                    if not self.tool_manager.validate_tool_input(tool_name, **kwargs):
                        st.error("Invalid input parameters")
                        return
                    
                    # Execute tool
                    result = self.tool_manager.execute_tool(tool_name, **kwargs)
                    
                    if result.success:
                        # st.success("Tool executed successfully!")
                        st.subheader("Execution Details:")
                        st.write(f"**Success:** {result.success}")
                        st.write(f"**Execution Time:** {result.execution_time:.2f}s")
                        st.write(f"**Tool Name:** {result.tool_name}")
                        st.subheader("Result:")
                        
                        # Display formatted result if available
                        if hasattr(result, 'formatted_result') and result.formatted_result:
                            self._display_formatted_result(result.formatted_result)
                        else:
                            st.json(result.result)
                    else:
                        st.error("Tool execution failed!")
                        st.subheader("Error Details:")
                        st.write(f"**Success:** {result.success}")
                        st.write(f"**Execution Time:** {result.execution_time:.2f}s")
                        st.write(f"**Tool Name:** {result.tool_name}")
                        st.write(f"**Error:** {result.error}")
                    
                except Exception as e:
                    st.error(f"Error executing tool: {str(e)}")
        
        except Exception as e:
            st.error(f"Error testing tool: {str(e)}")
    
    def update_tool_status(self, tool_name: str = None, new_status: str = "enabled"):
        """Update tool status"""
        if tool_name is None:
            st.error("No tool name provided for status update")
            return
        
        try:
            # This would typically update the tool definition file
            # For now, we'll just show a success message
            st.success(f"Tool {tool_name} status updated to {new_status}")
            
            # In a real implementation, you would:
            # 1. Load the tool definition JSON file
            # 2. Update the status field
            # 3. Save the file back
            # 4. Reload the tool manager
            
        except Exception as e:
            st.error(f"Error updating tool status: {str(e)}")
    
    def _display_formatted_result(self, formatted_result: Dict[str, Any]):
        """Display formatted result according to output_parameters structure"""
        for param_name, param_data in formatted_result.items():
            if isinstance(param_data, dict) and 'value' in param_data:
                value = param_data['value']
                param_type = param_data.get('type', 'string')
                description = param_data.get('description', '')
                param_format = param_data.get('format', 'plain_text')
                status = param_data.get('status', 'present')
                
                # Create a container for each output parameter
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if status == 'missing':
                            st.warning(f"**{param_name}**: Missing from result")
                        else:
                            st.write(f"**{param_name}**:")
                            
                            # Display value based on format
                            if param_format == 'json':
                                st.json(value)
                            elif param_format == 'markdown':
                                st.markdown(value)
                            elif param_format == 'plain_text':
                                if isinstance(value, (list, dict)):
                                    st.write(value)
                                else:
                                    st.write(str(value))
                            else:
                                st.write(value)
                        
                        if description:
                            st.caption(f"*{description}*")
                    
                    with col2:
                        st.write(f"**Type:** {param_type}")
                        if param_format != 'plain_text':
                            st.write(f"**Format:** {param_format}")
                        if status == 'missing':
                            st.error("Missing")
                        else:
                            # st.success("Present")
                            pass
                    
                    # st.divider()
            else:
                # Fallback for non-structured data
                st.write(f"**{param_name}**: {param_data}")
