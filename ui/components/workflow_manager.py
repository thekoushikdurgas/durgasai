"""
Workflow Manager UI Component for DurgasAI
Provides interface for managing and monitoring workflows
"""

import streamlit as st
import json
from typing import Dict, List, Any
from utils.workflow_service import WorkflowService, WorkflowStatus, StepStatus


class WorkflowManager:
    """UI component for workflow management"""
    
    def __init__(self, workflow_service: WorkflowService):
        self.workflow_service = workflow_service
    
    def render(self):
        """Render the workflow management interface"""
        # Workflow overview
        self.render_workflow_overview()
        
        # Workflow execution
        self.render_workflow_execution()
        
        # Workflow monitoring
        self.render_workflow_monitoring()
    
    def render_workflow_overview(self):
        """Render workflow overview section"""
        st.subheader("📊 Workflow Overview")
        
        workflows = self.workflow_service.get_all_workflows()
        stats = self.workflow_service.get_workflow_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Workflows", len(workflows))
        
        with col2:
            st.metric("Total Executions", stats.get("total_executions", 0))
        
        with col3:
            st.metric("Active Executions", stats.get("active_executions", 0))
        
        with col4:
            success_rate = stats.get("success_rate", 0)
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Available workflows
        st.subheader("🔄 Available Workflows")
        
        if workflows:
            for workflow_name, workflow_def in workflows.items():
                with st.expander(f"📋 {workflow_name}"):
                    st.write(f"**Description:** {workflow_def.get('description', 'No description')}")
                    st.write(f"**Steps:** {len(workflow_def.get('steps', []))}")
                    
                    # Show steps
                    steps = workflow_def.get('steps', [])
                    if steps:
                        st.write("**Workflow Steps:**")
                        for i, step in enumerate(steps):
                            st.write(f"{i+1}. {step.get('name', 'Unknown')} → {step.get('tool', 'Unknown')}")
        else:
            st.info("No workflows available")
    
    def render_workflow_execution(self):
        """Render workflow execution section"""
        st.subheader("🚀 Execute Workflow")
        
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
            st.write(f"**Selected Workflow:** {selected_workflow}")
            st.write(f"**Description:** {workflow_def.get('description', 'No description')}")
            
            # Input parameters
            input_params = workflow_def.get("input_parameters", [])
            if input_params:
                st.write("**Input Parameters:**")
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
                            key=f"workflow_exec_{param_name}"
                        )
                    elif param_type == "integer":
                        inputs[param_name] = st.number_input(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_desc,
                            step=1,
                            key=f"workflow_exec_{param_name}"
                        )
                    elif param_type == "boolean":
                        inputs[param_name] = st.checkbox(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=param_desc,
                            key=f"workflow_exec_{param_name}"
                        )
                    elif param_type == "array":
                        inputs[param_name] = st.text_area(
                            f"{param_name} {'(required)' if required else '(optional)'}",
                            help=f"{param_desc} (Enter JSON array)",
                            height=100,
                            key=f"workflow_exec_{param_name}"
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
                            st.success(f"Workflow execution completed! ID: {execution.execution_id}")
                            
                            # Display results
                            if execution.outputs:
                                st.write("**Workflow Outputs:**")
                                st.json(execution.outputs)
                            
                            # Display step results
                            st.write("**Step Results:**")
                            for i, step in enumerate(execution.steps):
                                status_icon = "✅" if step.status == StepStatus.COMPLETED else "❌"
                                st.write(f"{status_icon} Step {i+1}: {step.name} - {step.status.value}")
                                
                                if step.result:
                                    with st.expander(f"View result for {step.name}"):
                                        st.json(step.result)
                                
                                if step.error:
                                    st.error(f"Error in {step.name}: {step.error}")
                        else:
                            st.error("Failed to execute workflow")
                    
                    except Exception as e:
                        st.error(f"Error executing workflow: {str(e)}")
    
    def render_workflow_monitoring(self):
        """Render workflow monitoring section"""
        st.subheader("📊 Workflow Monitoring")
        
        # Active executions
        active_executions = self.workflow_service.active_executions
        
        if active_executions:
            st.write("**Active Executions:**")
            for execution_id, execution in active_executions.items():
                with st.expander(f"🔄 {execution.workflow_name} - {execution.status.value.title()}"):
                    st.write(f"**Execution ID:** {execution_id}")
                    st.write(f"**Status:** {execution.status.value.title()}")
                    st.write(f"**Start Time:** {execution.start_time}")
                    
                    # Show step progress
                    st.write("**Step Progress:**")
                    for i, step in enumerate(execution.steps):
                        status_icon = "✅" if step.status == StepStatus.COMPLETED else "⏳" if step.status == StepStatus.RUNNING else "⏸️"
                        st.write(f"{status_icon} Step {i+1}: {step.name} - {step.status.value}")
        else:
            st.info("No active executions")
        
        # Recent execution history
        st.subheader("📋 Recent Execution History")
        
        history = self.workflow_service.get_execution_history(limit=10)
        
        if history:
            for execution in history:
                with st.expander(f"🔄 {execution.workflow_name} - {execution.status.value.title()} ({execution.execution_id[:8]}...)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Status:** {execution.status.value.title()}")
                        st.write(f"**Duration:** {execution.total_execution_time:.2f}s")
                    
                    with col2:
                        completed_steps = sum(1 for step in execution.steps if step.status == StepStatus.COMPLETED)
                        st.write(f"**Steps:** {completed_steps}/{len(execution.steps)}")
                        
                        if execution.error:
                            st.error(f"**Error:** {execution.error}")
        else:
            st.info("No execution history available")
    
    def render_workflow_creation(self):
        """Render workflow creation interface"""
        st.subheader("➕ Create New Workflow")
        
        # Workflow name
        workflow_name = st.text_input("Workflow Name", help="Unique name for the workflow")
        
        # Workflow description
        workflow_description = st.text_area("Description", help="Description of what the workflow does")
        
        # Input parameters
        st.write("**Input Parameters:**")
        input_params = []
        
        if st.button("Add Input Parameter"):
            input_params.append({
                "name": "",
                "type": "string",
                "description": "",
                "required": False
            })
        
        for i, param in enumerate(input_params):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                param["name"] = st.text_input(f"Parameter {i+1} Name", value=param.get("name", ""), key=f"param_name_{i}")
            
            with col2:
                param["type"] = st.selectbox("Type", ["string", "integer", "boolean", "array"], key=f"param_type_{i}")
            
            with col3:
                param["required"] = st.checkbox("Required", value=param.get("required", False), key=f"param_required_{i}")
            
            with col4:
                if st.button("Remove", key=f"param_remove_{i}"):
                    input_params.pop(i)
                    st.rerun()
            
            param["description"] = st.text_input(f"Description", value=param.get("description", ""), key=f"param_desc_{i}")
        
        # Workflow steps
        st.write("**Workflow Steps:**")
        steps = []
        
        if st.button("Add Step"):
            steps.append({
                "name": "",
                "tool": "",
                "inputs": {}
            })
        
        for i, step in enumerate(steps):
            with st.expander(f"Step {i+1}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    step["name"] = st.text_input(f"Step Name", value=step.get("name", ""), key=f"step_name_{i}")
                
                with col2:
                    step["tool"] = st.text_input(f"Tool Name", value=step.get("tool", ""), key=f"step_tool_{i}")
                
                # Step inputs
                st.write("**Step Inputs:**")
                step_inputs = step.get("inputs", {})
                
                if st.button(f"Add Input", key=f"step_input_{i}"):
                    step_inputs[f"input_{len(step_inputs)}"] = {
                        "type": "const",
                        "value": ""
                    }
                
                for input_name, input_def in step_inputs.items():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.write(input_name)
                    
                    with col2:
                        input_type = st.selectbox("Type", ["const", "param", "ref"], key=f"step_input_type_{i}_{input_name}")
                        input_def["type"] = input_type
                        
                        if input_type == "const":
                            input_def["value"] = st.text_input("Value", value=input_def.get("value", ""), key=f"step_input_value_{i}_{input_name}")
                        elif input_type == "param":
                            input_def["name"] = st.text_input("Parameter Name", value=input_def.get("name", ""), key=f"step_input_param_{i}_{input_name}")
                        elif input_type == "ref":
                            input_def["step"] = st.number_input("Step Index", value=input_def.get("step", 0), key=f"step_input_step_{i}_{input_name}")
                            input_def["output"] = st.text_input("Output Name", value=input_def.get("output", ""), key=f"step_input_output_{i}_{input_name}")
                    
                    with col3:
                        if st.button("Remove", key=f"step_input_remove_{i}_{input_name}"):
                            del step_inputs[input_name]
                            st.rerun()
        
        # Create workflow
        if st.button("Create Workflow"):
            if workflow_name and workflow_description and steps:
                workflow_def = {
                    "name": workflow_name,
                    "description": workflow_description,
                    "input_parameters": input_params,
                    "output_parameters": [],
                    "steps": steps
                }
                
                if self.workflow_service.create_workflow(workflow_def):
                    st.success("Workflow created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create workflow")
            else:
                st.error("Please fill in all required fields")
