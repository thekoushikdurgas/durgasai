"""
Workflows Page Module
Handles workflow management, execution, and monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from utils.workflow_service import WorkflowService, WorkflowStatus, StepStatus


class WorkflowsPage:
    """Workflows page implementation with comprehensive workflow management"""
    
    def __init__(self, workflow_service: WorkflowService):
        self.workflow_service = workflow_service
    
    def render(self):
        """Render the workflows management page with tabbed interface"""
        st.title("🔄 Workflow Management")
        st.markdown("Create, execute, and monitor automated workflows with tool integration")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔄 Execution", "⚙️ Management", "📈 Analytics"])
        
        with tab1:
            st.markdown("### 📊 Overview")
            st.markdown("Workflow statistics and quick access to common operations")
            self.render_workflow_overview()
        
        with tab2:
            st.markdown("### 🔄 Execution")
            st.markdown("Execute workflows and monitor running processes")
            self.render_workflow_execution()
        
        with tab3:
            st.markdown("### ⚙️ Management")
            st.markdown("Create, edit, and manage workflow definitions")
            self.render_workflow_management()
        
        with tab4:
            st.markdown("### 📈 Analytics")
            st.markdown("Workflow performance analytics and execution history")
            self.render_workflow_analytics()
    
    def render_workflow_overview(self):
        """Render workflow overview metrics"""
        # Get workflow statistics
        stats = self.workflow_service.get_workflow_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Workflows",
                stats.get("total_workflows", 0),
                help="Number of available workflow definitions"
            )
        
        with col2:
            st.metric(
                "Total Executions",
                stats.get("total_executions", 0),
                help="Total number of workflow executions"
            )
        
        with col3:
            st.metric(
                "Active Executions",
                stats.get("active_executions", 0),
                help="Currently running workflows"
            )
        
        with col4:
            success_rate = stats.get("success_rate", 0)
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                help="Percentage of successful executions"
            )
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload Workflows", help="Reload workflow definitions from disk"):
                self.workflow_service.reload_workflows()
                st.success("Workflows reloaded successfully!")
                st.rerun()
        
        with col2:
            if st.button("📊 View Statistics", help="View detailed workflow statistics"):
                st.session_state.show_detailed_stats = True
                st.rerun()
        
        with col3:
            if st.button("📋 View History", help="View execution history"):
                st.session_state.show_execution_history = True
                st.rerun()
        
        # Show detailed statistics if requested
        if st.session_state.get("show_detailed_stats"):
            self.render_detailed_statistics(stats)
    
    def render_detailed_statistics(self, stats: Dict[str, Any] = None):
        """Render detailed workflow statistics"""
        if stats is None:
            st.warning("No statistics data available")
            return
        
        st.subheader("📊 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Execution Status Distribution:**")
            status_counts = stats.get("status_counts", {})
            if status_counts:
                for status, count in status_counts.items():
                    st.write(f"- {status.title()}: {count}")
            else:
                st.info("No execution data available")
        
        with col2:
            st.write("**Most Used Workflows:**")
            most_used = stats.get("most_used_workflows", [])
            if most_used:
                for workflow_name, count in most_used:
                    st.write(f"- {workflow_name}: {count} executions")
            else:
                st.info("No usage data available")
        
        # Average execution time
        avg_time = stats.get("average_execution_time", 0)
        st.write(f"**Average Execution Time:** {avg_time:.2f} seconds")
    
    def render_workflow_execution(self):
        """Render workflow execution interface"""
        workflows = self.workflow_service.get_all_workflows()
        
        if not workflows:
            st.info("No workflows available for execution")
            return
        
        # Workflow selection
        workflow_names = list(workflows.keys())
        selected_workflow = st.selectbox(
            "Select a workflow to execute:",
            workflow_names,
            help="Choose a workflow to run"
        )
        
        if selected_workflow:
            workflow_def = workflows[selected_workflow]
            
            # Display workflow info
            st.subheader(f"Workflow: {selected_workflow}")
            st.write(f"**Description:** {workflow_def.get('description', 'No description')}")
            
            # Input parameters
            input_params = workflow_def.get("input_parameters", [])
            if input_params:
                st.subheader("Input Parameters")
                inputs = {}
                
                for param in input_params:
                    param_name = param.get("name")
                    param_type = param.get("type", "string")
                    param_desc = param.get("description", "")
                    required = param.get("required", False)
                    
                    if param_type == "string":
                        inputs[param_name] = st.text_input(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_desc,
                            key=f"exec_{param_name}"
                        )
                    elif param_type == "integer":
                        inputs[param_name] = st.number_input(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_desc,
                            step=1,
                            key=f"exec_{param_name}"
                        )
                    elif param_type == "boolean":
                        inputs[param_name] = st.checkbox(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_desc,
                            key=f"exec_{param_name}"
                        )
                    elif param_type == "array":
                        inputs[param_name] = st.text_area(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=f"{param_desc} (Enter JSON array)",
                            height=100,
                            key=f"exec_{param_name}"
                        )
                        if inputs[param_name]:
                            try:
                                inputs[param_name] = json.loads(inputs[param_name])
                            except json.JSONDecodeError:
                                st.error(f"Invalid JSON for {param_name}")
                                return
                
                # Filter out empty inputs
                inputs = {k: v for k, v in inputs.items() if v != "" and v is not None}
            else:
                inputs = {}
            
            # Execute workflow
            if st.button("🚀 Execute Workflow", help="Run the selected workflow"):
                with st.spinner("Executing workflow..."):
                    try:
                        execution = self.workflow_service.execute_workflow(selected_workflow, inputs)
                        
                        if execution:
                            st.success(f"Workflow execution started! ID: {execution.execution_id}")
                            
                            # Store execution ID for monitoring
                            st.session_state.current_execution_id = execution.execution_id
                            st.rerun()
                        else:
                            st.error("Failed to start workflow execution")
                    
                    except Exception as e:
                        st.error(f"Error executing workflow: {str(e)}")
        
        # Show execution history if requested
        if st.session_state.get("show_execution_history"):
            self.render_execution_history()
    
    def render_execution_history(self):
        """Render workflow execution history"""
        st.subheader("📋 Execution History")
        
        # Get execution history
        history = self.workflow_service.get_execution_history(limit=50)
        
        if not history:
            st.info("No execution history available")
            return
        
        # Display executions in a table format
        for execution in history:
            with st.expander(f"🔄 {execution.workflow_name} - {execution.status.value.title()} ({execution.execution_id[:8]}...)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {execution.status.value.title()}")
                    st.write(f"**Start Time:** {datetime.fromtimestamp(execution.start_time).strftime('%Y-%m-%d %H:%M:%S') if execution.start_time else 'N/A'}")
                    st.write(f"**Duration:** {execution.total_execution_time:.2f}s")
                
                with col2:
                    st.write(f"**Steps:** {len(execution.steps)}")
                    completed_steps = sum(1 for step in execution.steps if step.status == StepStatus.COMPLETED)
                    st.write(f"**Completed:** {completed_steps}/{len(execution.steps)}")
                    
                    if execution.error:
                        st.error(f"**Error:** {execution.error}")
                
                # Show step details
                st.write("**Step Details:**")
                for i, step in enumerate(execution.steps):
                    status_icon = "✅" if step.status == StepStatus.COMPLETED else "❌" if step.status == StepStatus.FAILED else "⏳"
                    st.write(f"{status_icon} Step {i+1}: {step.name} ({step.tool}) - {step.status.value}")
                    
                    if step.error:
                        st.write(f"   Error: {step.error}")
    
    def render_workflow_management(self):
        """Render workflow management interface"""
        workflows = self.workflow_service.get_all_workflows()
        
        # Workflow selection
        if workflows:
            workflow_names = list(workflows.keys())
            selected_workflow = st.selectbox(
                "Select a workflow to manage:",
                workflow_names,
                help="Choose a workflow to view and edit"
            )
            
            if selected_workflow:
                workflow_def = workflows[selected_workflow]
                
                # Display workflow details
                st.subheader(f"Workflow: {selected_workflow}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"**Name:** {workflow_def.get('name', 'Unknown')}")
                    st.write(f"**Description:** {workflow_def.get('description', 'No description')}")
                    st.write(f"**Steps:** {len(workflow_def.get('steps', []))}")
                
                with col2:
                    st.write("**Parameters:**")
                    input_params = workflow_def.get('input_parameters', [])
                    output_params = workflow_def.get('output_parameters', [])
                    st.write(f"**Input Parameters:** {len(input_params)}")
                    st.write(f"**Output Parameters:** {len(output_params)}")
                
                # Show workflow definition
                st.subheader("Workflow Definition")
                st.json(workflow_def)
                
                # Management actions
                st.subheader("Management Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Export Workflow", help="Export workflow definition"):
                        export_data = self.workflow_service.export_workflow(selected_workflow)
                        if export_data:
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"{selected_workflow}.json",
                                mime="application/json"
                            )
                
                with col2:
                    if st.button("🔄 Reload Workflow", help="Reload workflow from disk"):
                        self.workflow_service.reload_workflows()
                        st.success("Workflow reloaded!")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete Workflow", help="Delete workflow definition"):
                        if self.workflow_service.delete_workflow(selected_workflow):
                            st.success("Workflow deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete workflow")
        
        # Create new workflow
        st.subheader("Create New Workflow")
        
        with st.expander("➕ Add New Workflow"):
            workflow_json = st.text_area(
                "Workflow Definition (JSON):",
                height=400,
                help="Enter the workflow definition in JSON format"
            )
            
            if st.button("Create Workflow"):
                try:
                    workflow_def = json.loads(workflow_json)
                    if self.workflow_service.create_workflow(workflow_def):
                        st.success("Workflow created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create workflow")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {str(e)}")
    
    def render_workflow_analytics(self):
        """Render workflow analytics and performance metrics"""
        stats = self.workflow_service.get_workflow_statistics()
        
        # Execution status chart
        st.subheader("📊 Execution Status Distribution")
        
        status_counts = stats.get("status_counts", {})
        if status_counts:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Workflow Execution Status",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No execution data available for analytics")
        
        # Most used workflows chart
        st.subheader("📈 Most Used Workflows")
        
        most_used = stats.get("most_used_workflows", [])
        if most_used:
            workflows = [item[0] for item in most_used]
            counts = [item[1] for item in most_used]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=workflows,
                    y=counts,
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Workflow Usage Frequency",
                xaxis_title="Workflows",
                yaxis_title="Execution Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usage data available for analytics")
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_time = stats.get("average_execution_time", 0)
            st.metric("Average Execution Time", f"{avg_time:.2f}s")
        
        with col2:
            success_rate = stats.get("success_rate", 0)
            st.metric("Success Rate", f"{success_rate:.1f}%")
