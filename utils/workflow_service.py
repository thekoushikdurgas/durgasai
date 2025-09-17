"""
Workflow Service for DurgasAI
Manages workflow execution, definition, and monitoring
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from utils.logging_utils import log_info, log_error, log_debug, log_warning
from utils.tool_manager import DurgasAIToolManager
from models.workflow_models import WorkflowDefinition, WorkflowExecution, WorkflowStep, WorkflowTemplate, WorkflowSchedule, WorkflowTrigger
from models import create_workflow_definition, create_workflow_template, create_workflow_schedule, create_workflow_trigger


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Individual step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    name: str
    tool: str
    inputs: Dict[str, Any]
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    start_time: float = None
    end_time: float = None


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance"""
    execution_id: str
    workflow_name: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = None
    start_time: float = None
    end_time: float = None
    total_execution_time: float = 0.0
    error: str = None
    created_at: float = None


class WorkflowService:
    """Service for managing and executing workflows"""
    
    def __init__(self, workflows_directory: str = "output/workflows", tool_manager: DurgasAIToolManager = None):
        self.workflows_directory = Path(workflows_directory)
        self.tool_manager = tool_manager
        self.workflow_definitions = {}
        self.execution_history = []
        self.active_executions = {}
        self._load_workflows()
        log_info(f"Workflow service initialized with {len(self.workflow_definitions)} workflows", "workflows")
    
    def _load_workflows(self, workflows_directory: str = None):
        """Load all workflow definitions from the workflows directory"""
        if workflows_directory is not None:
            self.workflows_directory = Path(workflows_directory)
        
        try:
            for workflow_file in self.workflows_directory.glob("*.json"):
                try:
                    with open(workflow_file, 'r', encoding='utf-8') as f:
                        workflow_def = json.load(f)
                    
                    workflow_name = workflow_def.get("name")
                    if workflow_name:
                        # Validate workflow definition
                        validation_errors = self._validate_workflow_definition(workflow_def)
                        if validation_errors:
                            log_warning(f"Workflow {workflow_name} has validation issues: {validation_errors}", "workflows")
                        
                        self.workflow_definitions[workflow_name] = workflow_def
                        log_debug(f"Loaded workflow: {workflow_name}", "workflows")
                        
                except Exception as e:
                    log_error(f"Error loading workflow {workflow_file}: {e}", "workflows", e)
        
        except Exception as e:
            log_error(f"Error loading workflows directory: {e}", "workflows", e)
    
    def _validate_workflow_definition(self, workflow_def: Dict) -> List[str]:
        """Validate workflow definition and return list of errors"""
        errors = []
        
        # Required fields
        required_fields = ["name", "description", "steps"]
        for field in required_fields:
            if not workflow_def.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate steps
        steps = workflow_def.get("steps", [])
        if not isinstance(steps, list):
            errors.append("steps must be a list")
        else:
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    errors.append(f"Step {i} must be a dictionary")
                    continue
                
                if not step.get("name"):
                    errors.append(f"Step {i} missing name")
                if not step.get("tool"):
                    errors.append(f"Step {i} missing tool")
                if not step.get("inputs"):
                    errors.append(f"Step {i} missing inputs")
        
        return errors
    
    def get_workflow_definition(self, workflow_name: str) -> Optional[Dict]:
        """Get workflow definition by name"""
        return self.workflow_definitions.get(workflow_name)
    
    def get_all_workflows(self) -> Dict[str, Dict]:
        """Get all available workflow definitions"""
        return self.workflow_definitions.copy()
    
    def create_workflow_execution(self, workflow_name: str, inputs: Dict[str, Any] = None) -> Optional[WorkflowExecution]:
        """Create a new workflow execution instance"""
        workflow_def = self.get_workflow_definition(workflow_name)
        if not workflow_def:
            log_error(f"Workflow {workflow_name} not found", "workflows")
            return None
        
        if not self.tool_manager:
            log_error("Tool manager not available for workflow execution", "workflows")
            return None
        
        # Create execution ID
        execution_id = str(uuid.uuid4())
        
        # Create workflow steps
        steps = []
        for step_def in workflow_def.get("steps", []):
            step = WorkflowStep(
                name=step_def["name"],
                tool=step_def["tool"],
                inputs=step_def.get("inputs", {})
            )
            steps.append(step)
        
        # Create execution instance
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.PENDING,
            steps=steps,
            inputs=inputs or {},
            created_at=time.time()
        )
        
        self.active_executions[execution_id] = execution
        log_info(f"Created workflow execution {execution_id} for {workflow_name}", "workflows")
        
        return execution
    
    def execute_workflow(self, workflow_name: str, inputs: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute a workflow and return the execution instance"""
        execution = self.create_workflow_execution(workflow_name, inputs)
        if not execution:
            return None
        
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = time.time()
            
            log_info(f"Starting workflow execution {execution.execution_id}", "workflows")
            
            # Execute each step
            for i, step in enumerate(execution.steps):
                try:
                    step.status = StepStatus.RUNNING
                    step.start_time = time.time()
                    
                    log_debug(f"Executing step {i+1}/{len(execution.steps)}: {step.name}", "workflows")
                    
                    # Resolve step inputs
                    resolved_inputs = self._resolve_step_inputs(step, execution, i)
                    
                    # Execute the tool
                    if step.tool in self.tool_manager.tool_functions:
                        result = self.tool_manager.execute_tool(step.tool, **resolved_inputs)
                        
                        if result.success:
                            step.result = result.result
                            step.status = StepStatus.COMPLETED
                            log_debug(f"Step {step.name} completed successfully", "workflows")
                        else:
                            step.error = result.error
                            step.status = StepStatus.FAILED
                            log_error(f"Step {step.name} failed: {result.error}", "workflows")
                            break
                    else:
                        step.error = f"Tool {step.tool} not found"
                        step.status = StepStatus.FAILED
                        log_error(f"Tool {step.tool} not found for step {step.name}", "workflows")
                        break
                    
                    step.end_time = time.time()
                    step.execution_time = step.end_time - step.start_time
                    
                except Exception as e:
                    step.error = str(e)
                    step.status = StepStatus.FAILED
                    step.end_time = time.time()
                    step.execution_time = step.end_time - step.start_time
                    log_error(f"Error executing step {step.name}: {e}", "workflows", e)
                    break
            
            # Determine final status
            execution.end_time = time.time()
            execution.total_execution_time = execution.end_time - execution.start_time
            
            if all(step.status == StepStatus.COMPLETED for step in execution.steps):
                execution.status = WorkflowStatus.COMPLETED
                execution.outputs = self._collect_workflow_outputs(execution)
                log_info(f"Workflow execution {execution.execution_id} completed successfully", "workflows")
            else:
                execution.status = WorkflowStatus.FAILED
                failed_steps = [step.name for step in execution.steps if step.status == StepStatus.FAILED]
                execution.error = f"Failed steps: {', '.join(failed_steps)}"
                log_error(f"Workflow execution {execution.execution_id} failed", "workflows")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = time.time()
            execution.total_execution_time = execution.end_time - execution.start_time
            log_error(f"Error executing workflow {workflow_name}: {e}", "workflows", e)
        
        finally:
            # Move to execution history
            self.execution_history.append(execution)
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
        
        return execution
    
    def _resolve_step_inputs(self, step: WorkflowStep, execution: WorkflowExecution, step_index: int) -> Dict[str, Any]:
        """Resolve step inputs from parameters, constants, and previous step outputs"""
        resolved_inputs = {}
        
        for input_name, input_def in step.inputs.items():
            if isinstance(input_def, dict):
                input_type = input_def.get("type")
                
                if input_type == "param":
                    # Get from workflow inputs
                    param_name = input_def.get("name")
                    resolved_inputs[input_name] = execution.inputs.get(param_name)
                
                elif input_type == "const":
                    # Use constant value
                    resolved_inputs[input_name] = input_def.get("value")
                
                elif input_type == "ref":
                    # Reference previous step output
                    ref_step = input_def.get("step", step_index - 1)
                    ref_output = input_def.get("output")
                    
                    if 0 <= ref_step < len(execution.steps):
                        ref_step_result = execution.steps[ref_step].result
                        if isinstance(ref_step_result, dict) and ref_output in ref_step_result:
                            resolved_inputs[input_name] = ref_step_result[ref_output]
                        else:
                            resolved_inputs[input_name] = ref_step_result
                    else:
                        log_warning(f"Invalid step reference {ref_step} in step {step.name}", "workflows")
                        resolved_inputs[input_name] = None
                
                else:
                    log_warning(f"Unknown input type {input_type} in step {step.name}", "workflows")
                    resolved_inputs[input_name] = None
            else:
                # Direct value
                resolved_inputs[input_name] = input_def
        
        return resolved_inputs
    
    def _collect_workflow_outputs(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Collect workflow outputs from step results"""
        outputs = {}
        
        # Get workflow definition to understand expected outputs
        workflow_def = self.get_workflow_definition(execution.workflow_name)
        if not workflow_def:
            return outputs
        
        output_params = workflow_def.get("output_parameters", [])
        
        # For now, collect all step results
        # In a more sophisticated implementation, you would map specific outputs
        for i, step in enumerate(execution.steps):
            if step.status == StepStatus.COMPLETED and step.result:
                outputs[f"step_{i}_{step.name}"] = step.result
        
        return outputs
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status by ID"""
        # Check active executions first
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check execution history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution
        
        return None
    
    def get_execution_history(self, workflow_name: str = None, limit: int = 100) -> List[WorkflowExecution]:
        """Get execution history, optionally filtered by workflow name"""
        history = self.execution_history.copy()
        
        if workflow_name:
            history = [exec for exec in history if exec.workflow_name == workflow_name]
        
        # Sort by creation time (newest first)
        history.sort(key=lambda x: x.created_at, reverse=True)
        
        return history[:limit]
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics"""
        total_workflows = len(self.workflow_definitions)
        total_executions = len(self.execution_history)
        active_executions = len(self.active_executions)
        
        # Execution status counts
        status_counts = {}
        for execution in self.execution_history:
            status = execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Most used workflows
        workflow_usage = {}
        for execution in self.execution_history:
            workflow_name = execution.workflow_name
            workflow_usage[workflow_name] = workflow_usage.get(workflow_name, 0) + 1
        
        most_used = sorted(workflow_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Average execution times
        completed_executions = [exec for exec in self.execution_history if exec.status == WorkflowStatus.COMPLETED]
        avg_execution_time = 0
        if completed_executions:
            total_time = sum(exec.total_execution_time for exec in completed_executions)
            avg_execution_time = total_time / len(completed_executions)
        
        return {
            "total_workflows": total_workflows,
            "total_executions": total_executions,
            "active_executions": active_executions,
            "status_counts": status_counts,
            "most_used_workflows": most_used,
            "average_execution_time": avg_execution_time,
            "success_rate": (status_counts.get("completed", 0) / total_executions * 100) if total_executions > 0 else 0
        }
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active workflow execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = time.time()
            execution.total_execution_time = execution.end_time - execution.start_time
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            log_info(f"Cancelled workflow execution {execution_id}", "workflows")
            return True
        
        return False
    
    def reload_workflows(self):
        """Reload all workflow definitions from disk"""
        log_info("Reloading workflows from disk", "workflows")
        self.workflow_definitions.clear()
        self._load_workflows()
        log_info(f"Reloaded {len(self.workflow_definitions)} workflows", "workflows")
    
    def create_workflow(self, workflow_def: Dict) -> bool:
        """Create a new workflow definition"""
        try:
            workflow_name = workflow_def.get("name")
            if not workflow_name:
                log_error("Workflow name is required", "workflows")
                return False
            
            # Validate workflow definition
            validation_errors = self._validate_workflow_definition(workflow_def)
            if validation_errors:
                log_error(f"Workflow validation failed: {validation_errors}", "workflows")
                return False
            
            # Save to file
            workflow_file = self.workflows_directory / f"{workflow_name}.json"
            with open(workflow_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_def, f, indent=2)
            
            # Add to definitions
            self.workflow_definitions[workflow_name] = workflow_def
            
            log_info(f"Created workflow: {workflow_name}", "workflows")
            return True
            
        except Exception as e:
            log_error(f"Error creating workflow: {e}", "workflows", e)
            return False
    
    def delete_workflow(self, workflow_name: str) -> bool:
        """Delete a workflow definition"""
        try:
            # Remove from definitions
            if workflow_name in self.workflow_definitions:
                del self.workflow_definitions[workflow_name]
            
            # Delete file
            workflow_file = self.workflows_directory / f"{workflow_name}.json"
            if workflow_file.exists():
                workflow_file.unlink()
            
            log_info(f"Deleted workflow: {workflow_name}", "workflows")
            return True
            
        except Exception as e:
            log_error(f"Error deleting workflow {workflow_name}: {e}", "workflows", e)
            return False
    
    def export_workflow(self, workflow_name: str) -> Optional[Dict]:
        """Export workflow definition as JSON"""
        return self.get_workflow_definition(workflow_name)
    
    def import_workflow(self, workflow_def: Dict) -> bool:
        """Import workflow definition from JSON"""
        return self.create_workflow(workflow_def)
