"""
Enhanced Tools Page Module
Handles the tools management functionality with workflow integration
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from ..components.tool_manager import ToolManager


class ToolsPage:
    """Enhanced tools page implementation with workflow integration"""
    
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
    
    def render(self):
        """Render the enhanced tools management page with tabbed interface"""
        # st.title("🔧 Enhanced Tools Management")
        # st.markdown("Manage, monitor, and analyze your AI tools with workflow integration")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analytics", "⚙️ Management", "🧪 Testing"])
        
        with tab1:
            # st.markdown("### 📊 Overview")
            # st.markdown("Quick overview of tool statistics and metrics")
            self.render_tool_overview()
        
        with tab2:
            # st.markdown("### 📈 Analytics")
            # st.markdown("Detailed performance analytics and usage patterns")
            self.render_tool_analytics()
        
        with tab3:
            # st.markdown("### ⚙️ Management")
            # st.markdown("Configure, enable/disable, and manage individual tools")
            self.render_tool_management()
        
        with tab4:
            # st.markdown("### 🧪 Testing")
            # st.markdown("Test individual tools or run batch tests")
            self.render_tool_testing()
    
    def render_tool_overview(self):
        """Render tool overview metrics"""
        
        # Get tool statistics
        tool_stats = self.tool_manager.get_tool_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Tools",
                tool_stats.get("total_tools", 0),
                help="Total number of available tools"
            )
        
        with col2:
            st.metric(
                "Active Tools",
                tool_stats.get("active_tools", 0),
                help="Number of tools that have been used"
            )
        
        with col3:
            st.metric(
                "Total Executions",
                tool_stats.get("total_executions", 0),
                help="Total number of tool executions"
            )
        
        with col4:
            avg_execution_time = tool_stats.get("avg_execution_time", 0)
            st.metric(
                "Avg Execution Time",
                f"{avg_execution_time:.2f}s",
                help="Average tool execution time"
            )
    
    def render_tool_analytics(self):
        """Render tool performance analytics"""
        
        # Tool usage chart
        self.render_tool_usage_chart()
        
        # Tool performance trends
        self.render_tool_performance_trends()
        
        # Tool categories analysis
        self.render_tool_categories()
    
    def render_tool_usage_chart(self):
        """Render tool usage frequency chart"""
        st.subheader("🔧 Tool Usage Frequency")
        
        tool_stats = self.tool_manager.get_tool_statistics()
        tool_usage = tool_stats.get("tool_usage", {})
        
        if not tool_usage:
            st.info("No tool usage data available")
            return
        
        # Create horizontal bar chart
        tools = list(tool_usage.keys())
        usage_counts = list(tool_usage.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=usage_counts,
                y=tools,
                orientation='h',
                marker_color='lightblue',
                text=usage_counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Tool Usage Frequency",
            xaxis_title="Usage Count",
            yaxis_title="Tools",
            height=max(400, len(tools) * 30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_tool_performance_trends(self):
        """Render tool performance trends over time"""
        st.subheader("📊 Tool Performance Trends")
        
        # Get execution history
        execution_history = self.tool_manager.execution_history
        
        if len(execution_history) < 2:
            st.info("Insufficient data for performance trends")
            return
        
        # Prepare data for time series
        timestamps = [execution.timestamp for execution in execution_history]
        execution_times = [execution.execution_time for execution in execution_history]
        tool_names = [execution.tool_name for execution in execution_history]
        
        # Create performance trend chart
        fig = go.Figure()
        
        # Group by tool and create traces
        unique_tools = list(set(tool_names))
        colors = px.colors.qualitative.Set3[:len(unique_tools)]
        
        for i, tool in enumerate(unique_tools):
            tool_timestamps = [ts for j, ts in enumerate(timestamps) if tool_names[j] == tool]
            tool_times = [et for j, et in enumerate(execution_times) if tool_names[j] == tool]
            
            fig.add_trace(go.Scatter(
                x=tool_timestamps,
                y=tool_times,
                mode='lines+markers',
                name=tool,
                line=dict(color=colors[i])
            ))
        
        fig.update_layout(
            title="Tool Execution Time Trends",
            xaxis_title="Time",
            yaxis_title="Execution Time (seconds)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_tool_categories(self):
        """Render tool categories analysis"""
        st.subheader("📂 Tool Categories Analysis")
        
        tools = self.tool_manager.get_all_tools()
        
        if not tools:
            st.info("No tools available for analysis")
            return
        
        # Group tools by category
        categories = {}
        for tool_name, tool_def in tools.items():
            category = tool_def.get("category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(tool_def)
        
        # Create category distribution chart
        category_names = list(categories.keys())
        category_counts = [len(tools) for tools in categories.values()]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=category_names,
                values=category_counts,
                hole=0.3
            )
        ])
        
        fig.update_layout(
            title="Tool Distribution by Category",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_tool_management(self):
        """Render tool management interface"""
        
        tools = self.tool_manager.get_all_tools()
        
        if not tools:
            st.info("No tools available")
            return
        
        # Tool selection and details
        tool_names = [tool_def['name'] for tool_def in tools.values()]
        selected_tool_name = st.selectbox(
            "Select a tool to manage:",
            tool_names,
            help="Choose a tool to view details and manage"
        )
        
        selected_tool = next(tool_def for tool_def in tools.values() if tool_def['name'] == selected_tool_name)
        
        # Tool details
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("**Tool Information:**", expanded=False):
                st.write(f"**Name:** {selected_tool['name']}")
                st.write(f"**Description:** {selected_tool['description']}")
                st.write(f"**Category:** {selected_tool['category']}")
                st.write(f"**Parameters:** {len(selected_tool.get('parameters', {}))}")
        
        with col2:
            
            with st.expander("**Performance Metrics:**", expanded=False):
                # Get tool performance data
                tool_performance = self.tool_manager.get_tool_performance_report(selected_tool_name)
                st.write(f"**Total Executions:** {tool_performance.get('total_executions', 0)}")
                st.write(f"**Success Rate:** {tool_performance.get('success_rate', 0):.1f}%")
                st.write(f"**Average Time:** {tool_performance.get('avg_execution_time', 0):.2f}s")
                st.write(f"**Last Used:** {tool_performance.get('last_used', 'Never')}")
        
        # Tool actions
        # st.subheader("🔧 Tool Actions")
        
        # col2, col3 = st.columns(2)
        
        # with col1:
        #     if st.button("🔄 Reload Tool", help="Reload the selected tool"):
        #         self.tool_manager.reload_tools()
        #         st.success("Tools reloaded successfully!")
        #         st.rerun()
        
        # with col2:
        # if st.button("🧪 Test Tool", help="Test the selected tool"):
            # st.session_state.test_tool = selected_tool_name
        self.render_tool_test_interface(selected_tool_name)
            # st.rerun()
        
        # with col3:
        #     if st.button("📊 View Details", help="View detailed tool information"):
        #         st.session_state.view_tool_details = selected_tool_name
        #         st.rerun()
    
    def render_tool_testing(self):
        """Render tool testing interface"""
        
        # # Test selected tool
        # if st.session_state.get("test_tool"):
        #     self.render_tool_test_interface(st.session_state.test_tool)
        
        # Batch testing
        self.render_batch_testing()
    
    def render_tool_test_interface(self, tool_name: str = None):
        """Render interface for testing a specific tool"""
        if tool_name is None:
            st.error("No tool name provided for testing")
            return
        
        # st.write(f"**Testing Tool: {tool_name}**")
        
        # Get tool info
        tool_info = self.tool_manager.get_tool_info(tool_name)
        
        if not tool_info:
            st.error(f"Tool '{tool_name}' not found")
            return
        
        # Test parameters input
        # st.write("**Test Parameters:**")
        
        # Add example values buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Fill Example Values", help="Fill in example values for testing"):
                st.session_state.fill_examples = True
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Values", help="Clear all input values"):
                st.session_state.fill_examples = False
                # Clear all input fields by rerunning
                st.rerun()
        
        test_params = {}
        parameters = tool_info.get("parameters", [])
        
        # Get example values for this tool
        example_values = self._get_example_values(tool_name, parameters)
        
        for param_info in parameters:
            param_name = param_info.get("name")
            param_type = param_info.get("type", "string")
            param_description = param_info.get("description", "")
            required = param_info.get("required", False)
            item_type = param_info.get("item_type", "string")
            
            # Get example value for this parameter
            example_value = example_values.get(param_name, "")
            
            # Create a container for each parameter
            with st.container():
                # col1, col2 = st.columns([2, 1])
                
                # with col1:
                    if param_type == "string":
                        test_params[param_name] = st.text_input(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            value=example_value if st.session_state.get("fill_examples", False) else "",
                            help=param_description,
                            key=f"test_{tool_name}_{param_name}"
                        )
                    elif param_type == "integer":
                        test_params[param_name] = st.number_input(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            value=int(example_value) if example_value and st.session_state.get("fill_examples", False) else 0,
                            help=param_description,
                            step=1,
                            key=f"test_{tool_name}_{param_name}"
                        )
                    elif param_type == "number":
                        test_params[param_name] = st.number_input(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            value=float(example_value) if example_value and st.session_state.get("fill_examples", False) else 0.0,
                            help=param_description,
                            step=0.1,
                            key=f"test_{tool_name}_{param_name}"
                        )
                    elif param_type == "boolean":
                        test_params[param_name] = st.checkbox(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_description,
                            key=f"test_{tool_name}_{param_name}"
                        )
                    elif param_type == "array":
                        if item_type == "string":
                            test_params[param_name] = st.text_area(
                                f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array of strings",
                                value=example_value if st.session_state.get("fill_examples", False) else "",
                                help=f"{param_description} (e.g., [\"item1\", \"item2\", \"item3\"])",
                                height=100,
                                key=f"test_{tool_name}_{param_name}"
                            )
                        elif item_type == "number":
                            test_params[param_name] = st.text_area(
                                f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array of numbers",
                                value=example_value if st.session_state.get("fill_examples", False) else "",
                                help=f"{param_description} (e.g., [1, 2, 3, 4, 5])",
                                height=100,
                                key=f"test_{tool_name}_{param_name}"
                            )
                        elif item_type == "boolean":
                            test_params[param_name] = st.text_area(
                                f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array of booleans",
                                value=example_value if st.session_state.get("fill_examples", False) else "",
                                help=f"{param_description} (e.g., [true, false, true])",
                                height=100,
                                key=f"test_{tool_name}_{param_name}"
                            )
                        else:
                            test_params[param_name] = st.text_area(
                                f"{param_name} {'(required)' if required else '(optional)'} - Enter JSON array",
                                value=example_value if st.session_state.get("fill_examples", False) else "",
                                help=f"{param_description} (JSON array format)",
                                height=100,
                                key=f"test_{tool_name}_{param_name}"
                            )
                    elif param_type == "file":
                        test_params[param_name] = st.file_uploader(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_description,
                            key=f"test_{tool_name}_{param_name}"
                        )
                    else:
                        test_params[param_name] = st.text_input(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_description,
                            key=f"test_{tool_name}_{param_name}"
                        )
                
                # with col2:
                #     st.write(f"**Type:** {param_type}")
                #     if param_type == "array":
                #         st.write(f"**Item Type:** {item_type}")
                #     st.write(f"**Required:** {'Yes' if required else 'No'}")
                #     if example_value and st.session_state.get("fill_examples", False):
                #         st.write(f"**Example:** {example_value}")
            
            # Add separator between parameters
            # st.divider()
        
        # Test execution
        if st.button("🚀 Run Test", help="Execute the tool with test parameters"):
            with st.spinner("Testing tool..."):
                try:
                    # Process parameters before execution
                    processed_params = {}
                    for param_name, param_value in test_params.items():
                        if param_value is not None and param_value != "":
                            # Find the parameter definition to get type info
                            param_def = next((p for p in parameters if p.get("name") == param_name), {})
                            param_type = param_def.get("type", "string")
                            
                            if param_type == "array":
                                # Parse JSON array
                                try:
                                    processed_params[param_name] = json.loads(param_value)
                                except json.JSONDecodeError as e:
                                    st.error(f"Invalid JSON for {param_name}: {str(e)}")
                                    return
                            elif param_type == "file":
                                # Handle file upload
                                if hasattr(param_value, 'read'):
                                    # Convert file upload to dict format expected by tools
                                    file_content = param_value.read()
                                    processed_params[param_name] = {
                                        "name": param_value.name,
                                        "content": file_content,
                                        "type": param_value.type,
                                        "size": param_value.size
                                    }
                                else:
                                    processed_params[param_name] = param_value
                            else:
                                processed_params[param_name] = param_value
                    
                    # Execute tool
                    result = self.tool_manager.execute_tool(tool_name, **processed_params)
                    # print(result)
                    # Display results
                    # st.success("Tool executed successfully!")
                    with st.expander("**Result:**", expanded=True):
                    # col1, col2 = st.columns(2)
                    
                    # with col1:
                        # st.write("**Execution Details:**")
                        st.write(f"**Success:** {result.success}")
                        st.write(f"**Execution Time:** {result.execution_time:.2f}s")
                        # st.write(f"**Tool Name:** {result.tool_name}")
                    
                    # with col2:
                        # st.write("**Result:**")
                        if result.success:
                            # Display formatted result if available
                            if hasattr(result, 'formatted_result') and result.formatted_result:
                                self._display_formatted_result(result.formatted_result)
                            else:
                                st.json(result.result)
                        else:
                            st.error(f"Error: {result.error}")
                
                except Exception as e:
                    st.error(f"Test failed: {str(e)}")
        
        # Clear test session
        if st.button("❌ Clear Test"):
            st.session_state.test_tool = None
            st.rerun()
    
    def render_batch_testing(self):
        """Render batch testing interface"""
        st.subheader("🔄 Batch Tool Testing")
        
        tools = self.tool_manager.get_all_tools()
        
        if not tools:
            st.info("No tools available for batch testing")
            return
        
        # Select tools for batch testing
        selected_tools = st.multiselect(
            "Select tools for batch testing:",
            [tool_def['name'] for tool_def in tools.values()],
            help="Choose multiple tools to test simultaneously"
        )
        
        if selected_tools and st.button("🚀 Run Batch Test", help="Test all selected tools"):
            with st.spinner("Running batch tests..."):
                try:
                    results = self.tool_manager.batch_test_tools(selected_tools)
                    
                    st.success(f"Batch test completed! Tested {len(selected_tools)} tools.")
                    
                    # Display results
                    for tool_name, result in results.items():
                        with st.expander(f"🔧 {tool_name} - {'✅ Success' if result['success'] else '❌ Failed'}"):
                            st.write(f"**Execution Time:** {result['execution_time']:.2f}s")
                            if result['success']:
                                st.write("**Result:**")
                                st.json(result['result'])
                            else:
                                st.error(f"**Error:** {result['error']}")
                
                except Exception as e:
                    st.error(f"Batch test failed: {str(e)}")
    
    def _get_example_values(self, tool_name: str, parameters: list) -> dict:
        """Get example values for tool parameters based on tool name"""
        examples = {}
        
        # Define example values for common tools
        tool_examples = {
            "add_two_numbers": {"a": "5", "b": "3"},
            "subtract_two_numbers": {"a": "10", "b": "3"},
            "calculate_math": {"expression": "2 + 3 * 4"},
            "get_current_time": {
                "format_type": "iso_8601_extended",
                "timezone_offset": "UTC",
                "include_timezone": "true",
                "precision": "seconds",
                "include_weekday": "true",
                "include_week_number": "false",
                "include_julian_date": "false",
                "include_epoch_variants": "true",
                "include_historical_formats": "false",
                "custom_format_pattern": "%Y-%m-%d %H:%M:%S",
                "include_common_strings": "true"
            },
            "get_current_weather": {"location": "New York, NY", "unit": "celsius"},
            "search_web": {"query": "artificial intelligence", "num_results": "3"},
            "tavily_search": {"query": "latest AI news", "num_results": "5"},
            "text_list_processor": {"text_list": '["Hello", "World", "Test"]', "operation": "uppercase"},
            "number_list_analyzer": {"numbers": "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "include_details": "true"},
            "data_analyzer_enhanced": {"dataset": "[1.5, 2.3, 3.7, 4.1, 5.9]", "analysis_type": "basic", "include_visualization": "false"},
            "task_list_manager": {"task_names": '["Task 1", "Task 2", "Task 3"]', "completion_status": "[true, false, true]", "show_summary": "true"},
            "generate_random_number": {"min_value": "1", "max_value": "100"},
            "browser_search": {"query": "machine learning", "topn": "5"},
            "browser_open": {"id": "-1", "num_lines": "10"},
            "browser_find": {"pattern": "artificial intelligence", "cursor": "-1"},
            "file_operations": {"operation": "list", "filename": "test.txt"},
        }
        
        # Get examples for this specific tool
        if tool_name in tool_examples:
            examples = tool_examples[tool_name]
        
        # Generate generic examples for parameters not in the specific tool examples
        for param in parameters:
            param_name = param.get("name")
            param_type = param.get("type", "string")
            item_type = param.get("item_type", "string")
            
            if param_name not in examples:
                if param_type == "string":
                    examples[param_name] = f"example_{param_name}"
                elif param_type == "integer":
                    examples[param_name] = "42"
                elif param_type == "number":
                    examples[param_name] = "3.14"
                elif param_type == "boolean":
                    examples[param_name] = "true"
                elif param_type == "array":
                    if item_type == "string":
                        examples[param_name] = '["item1", "item2", "item3"]'
                    elif item_type == "number":
                        examples[param_name] = "[1, 2, 3, 4, 5]"
                    elif item_type == "boolean":
                        examples[param_name] = "[true, false, true]"
                    else:
                        examples[param_name] = '["example"]'
        
        return examples
    
    def _display_formatted_result(self, formatted_result: Dict[str, Any]):
        """Display formatted result according to output_parameters structure"""
        for param_name, param_data in formatted_result.items():
            if isinstance(param_data, dict) and 'value' in param_data:
                value = param_data['value']
                param_type = param_data.get('type', 'string')
                # description = param_data.get('description', '')
                param_format = param_data.get('format', 'plain_text')
                status = param_data.get('status', 'present')
                
                # Create a container for each output parameter
                with st.container():
                    # col1, col2 = st.columns([3, 1])
                    
                    # with col1:
                        if status == 'missing':
                            st.warning(f"**{param_name}**: Missing from result")
                        else:
                            # Display value based on format
                            if param_format == 'json':
                                st.write(f"**{param_name}**:")
                                st.json(value)
                            elif param_format == 'markdown':
                                st.write(f"**{param_name}**:")
                                st.markdown(value)
                            elif param_format == 'plain_text':
                                if isinstance(value, (list, dict)):
                                    st.write(f"**{param_name}**: {value}")
                                    # st.write(value)
                                else:
                                    st.write(f"**{param_name}**: {str(value)}")
                                    # st.write(str(value))
                            else:
                                st.write(f"**{param_name}**: {value}")
                                # st.write(value)
                        
                        # if description:
                        #     st.caption(f"*{description}*")
                    
                    # with col2:
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
