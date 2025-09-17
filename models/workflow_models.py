"""
Workflow Models and Data Structures for AI Agent Dashboard

This module provides Pydantic models and data structures for the workflow management
system, including workflow definitions, execution records, and configuration models.

Features:
- Workflow definition schemas
- Execution tracking models
- Step dependency models
- Configuration and settings
- Type definitions for workflow operations
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from datetime import datetime, timedelta
from enum import Enum
import uuid
import re
import json


class WorkflowStatus(str, Enum):
    """Workflow status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepInputType(str, Enum):
    """Step input type enumeration"""
    CONSTANT = "const"
    PARAMETER = "param"
    REFERENCE = "ref"


class WorkflowStep(BaseModel):
    """Individual workflow step definition"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(description="Step name")
    tool: str = Field(description="Tool to execute")
    inputs: Dict[str, Any] = Field(default={}, description="Step input configuration")
    outputs: List[str] = Field(default=[], description="Expected output names")
    description: Optional[str] = Field(default=None, description="Step description")
    timeout: Optional[int] = Field(default=None, description="Step timeout in seconds")
    retry_count: int = Field(default=0, description="Number of retries on failure")
    enabled: bool = Field(default=True, description="Whether step is enabled")
    
    # Enhanced step properties
    dependencies: List[str] = Field(default=[], description="Step dependencies (step IDs)")
    conditions: Dict[str, Any] = Field(default={}, description="Execution conditions")
    parallel_execution: bool = Field(default=False, description="Can run in parallel")
    priority: int = Field(default=0, description="Step priority (higher = more important)")
    resource_requirements: Dict[str, Any] = Field(default={}, description="Resource requirements")
    error_handling: Dict[str, Any] = Field(default={}, description="Error handling configuration")
    notifications: List[Dict[str, Any]] = Field(default=[], description="Notification settings")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Step name cannot be empty")
        return v.strip()
    
    @validator('tool')
    def validate_tool(cls, v):
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @validator('retry_count')
    def validate_retry_count(cls, v):
        if v < 0:
            raise ValueError("Retry count cannot be negative")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Priority must be between 0 and 100")
        return v


class WorkflowDefinition(BaseModel):
    """Complete workflow definition schema"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Workflow name")
    description: str = Field(description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    author: str = Field(default="Unknown", description="Workflow author")
    category: str = Field(default="General", description="Workflow category")
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT, description="Workflow status")
    
    # Workflow structure
    steps: List[WorkflowStep] = Field(default=[], description="Workflow steps")
    input_parameters: List[Dict[str, Any]] = Field(default=[], description="Input parameters")
    output_parameters: List[Dict[str, Any]] = Field(default=[], description="Output parameters")
    
    # Metadata
    tags: List[str] = Field(default=[], description="Workflow tags")
    dependencies: List[str] = Field(default=[], description="Required dependencies")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_executed: Optional[datetime] = Field(default=None)
    
    # Execution statistics
    execution_count: int = Field(default=0)
    success_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    average_execution_time: float = Field(default=0.0)
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Workflow name cannot be empty")
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError("Workflow description cannot be empty")
        return v.strip()
    
    @validator('steps')
    def validate_steps(cls, v):
        if not v:
            raise ValueError("Workflow must have at least one step")
        return v


class WorkflowExecution(BaseModel):
    """Workflow execution record"""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(description="Workflow identifier")
    workflow_name: str = Field(description="Workflow name")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    
    # Input and output
    inputs: Dict[str, Any] = Field(default={}, description="Execution inputs")
    outputs: Optional[Dict[str, Any]] = Field(default=None, description="Execution outputs")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Timing and performance
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(default=None)
    duration: Optional[float] = Field(default=None)
    
    # Step execution details
    step_results: List[Dict[str, Any]] = Field(default=[], description="Individual step results")
    step_executions: List[Dict[str, Any]] = Field(default=[], description="Step execution details")
    completed_steps: int = Field(default=0)
    total_steps: int = Field(default=0)
    
    # Performance metrics
    performance_metrics: List[Dict[str, Any]] = Field(default=[], description="Performance data")
    memory_usage_mb: Optional[float] = Field(default=None)
    cpu_usage_percent: Optional[float] = Field(default=None)
    
    # Context
    executed_by: str = Field(default="system", description="Who executed the workflow")
    session_id: Optional[str] = Field(default=None, description="Session context")
    
    # Logs and debugging
    logs: List[str] = Field(default=[], description="Execution logs")
    warnings: List[str] = Field(default=[], description="Execution warnings")
    errors: List[str] = Field(default=[], description="Execution errors")
    
    def calculate_duration(self):
        """Calculate execution duration"""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def add_log(self, message: str):
        """Add log message"""
        self.logs.append(f"[{datetime.now().isoformat()}] {message}")
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(f"[{datetime.now().isoformat()}] {message}")
    
    def add_error(self, message: str):
        """Add error message"""
        self.errors.append(f"[{datetime.now().isoformat()}] {message}")


class WorkflowExecutionHistory(BaseModel):
    """Workflow execution history container"""
    executions: List[WorkflowExecution] = Field(default=[])
    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    average_duration: float = Field(default=0.0)
    
    def add_execution(self, execution: WorkflowExecution = None):
        """Add execution to history"""
        if execution is None:
            return
        
        self.executions.append(execution)
        self.total_executions += 1
        
        if execution.status == ExecutionStatus.COMPLETED:
            self.successful_executions += 1
        elif execution.status == ExecutionStatus.FAILED:
            self.failed_executions += 1
        
        # Update average duration
        if execution.duration is not None:
            total_duration = sum(e.duration or 0 for e in self.executions)
            self.average_duration = total_duration / len(self.executions)
    
    def get_recent_executions(self, limit: int = 10) -> List[WorkflowExecution]:
        """Get recent executions"""
        return sorted(self.executions, key=lambda x: x.start_time, reverse=True)[:limit]
    
    def get_executions_by_workflow(self, workflow_id: str) -> List[WorkflowExecution]:
        """Get executions for specific workflow"""
        return [e for e in self.executions if e.workflow_id == workflow_id]


class StepDependency(BaseModel):
    """Step dependency definition"""
    from_step: int = Field(description="Source step index")
    to_step: int = Field(description="Target step index")
    output_name: str = Field(description="Output parameter name")
    input_name: str = Field(description="Input parameter name")
    
    @validator('from_step', 'to_step')
    def validate_step_indices(cls, v):
        if v < 0:
            raise ValueError("Step index must be non-negative")
        return v


class WorkflowConfiguration(BaseModel):
    """Workflow system configuration"""
    # Directories
    workflows_directory: str = Field(default="output/workflows")
    
    # Execution settings
    max_execution_time: int = Field(default=3600, description="Max execution time in seconds")
    max_concurrent_executions: int = Field(default=5, description="Max concurrent executions")
    enable_parallel_execution: bool = Field(default=False, description="Enable parallel step execution")
    
    # Step settings
    default_step_timeout: int = Field(default=300, description="Default step timeout in seconds")
    max_retry_attempts: int = Field(default=3, description="Max retry attempts per step")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")
    
    # History settings
    max_history_size: int = Field(default=1000, description="Maximum history entries")
    history_retention_days: int = Field(default=30, description="History retention in days")
    
    # UI settings
    default_page_size: int = Field(default=20, description="Default pagination size")
    show_advanced_options: bool = Field(default=False, description="Show advanced options by default")


class WorkflowRegistry(BaseModel):
    """Workflow registry container"""
    workflows: Dict[str, WorkflowDefinition] = Field(default={})
    categories: Dict[str, List[str]] = Field(default={})
    tags: Dict[str, List[str]] = Field(default={})
    
    def add_workflow(self, workflow: WorkflowDefinition):
        """Add workflow to registry"""
        self.workflows[workflow.id] = workflow
        
        # Update categories
        if workflow.category not in self.categories:
            self.categories[workflow.category] = []
        if workflow.id not in self.categories[workflow.category]:
            self.categories[workflow.category].append(workflow.id)
        
        # Update tags
        for tag in workflow.tags:
            if tag not in self.tags:
                self.tags[tag] = []
            if workflow.id not in self.tags[tag]:
                self.tags[tag].append(workflow.id)
    
    def remove_workflow(self, workflow_id: str):
        """Remove workflow from registry"""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            
            # Remove from categories
            if workflow.category in self.categories:
                if workflow_id in self.categories[workflow.category]:
                    self.categories[workflow.category].remove(workflow_id)
            
            # Remove from tags
            for tag in workflow.tags:
                if tag in self.tags and workflow_id in self.tags[tag]:
                    self.tags[tag].remove(workflow_id)
            
            del self.workflows[workflow_id]
    
    def get_workflows_by_category(self, category: str) -> List[WorkflowDefinition]:
        """Get workflows by category"""
        workflow_ids = self.categories.get(category, [])
        return [self.workflows[wid] for wid in workflow_ids if wid in self.workflows]
    
    def get_workflows_by_tag(self, tag: str) -> List[WorkflowDefinition]:
        """Get workflows by tag"""
        workflow_ids = self.tags.get(tag, [])
        return [self.workflows[wid] for wid in workflow_ids if wid in self.workflows]
    
    def search_workflows(self, query: str) -> List[WorkflowDefinition]:
        """Search workflows by name or description"""
        query_lower = query.lower()
        results = []
        
        for workflow in self.workflows.values():
            if (query_lower in workflow.name.lower() or 
                query_lower in workflow.description.lower() or
                any(query_lower in tag.lower() for tag in workflow.tags)):
                results.append(workflow)
        
        return results


class WorkflowExecutionResult(BaseModel):
    """Workflow execution result wrapper"""
    success: bool = Field(description="Whether execution was successful")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(description="Execution time in seconds")
    steps_completed: int = Field(description="Number of steps completed")
    total_steps: int = Field(description="Total number of steps")
    performance_metrics: Dict[str, Any] = Field(default={}, description="Performance data")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class StepExecutionResult(BaseModel):
    """Individual step execution result"""
    step_id: str = Field(description="Step identifier")
    step_name: str = Field(description="Step name")
    tool: str = Field(description="Tool executed")
    success: bool = Field(description="Whether step was successful")
    result: Optional[Any] = Field(default=None, description="Step result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(description="Step execution time in seconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    inputs: Dict[str, Any] = Field(default={}, description="Step inputs")
    outputs: Dict[str, Any] = Field(default={}, description="Step outputs")


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_workflow_definition(name: str, description: str, steps: List[WorkflowStep],
                              input_parameters: List[Dict[str, Any]] = None,
                              output_parameters: List[Dict[str, Any]] = None,
                              category: str = "General", author: str = "Unknown") -> WorkflowDefinition:
    """Create a new workflow definition"""
    return WorkflowDefinition(
        name=name,
        description=description,
        steps=steps,
        input_parameters=input_parameters or [],
        output_parameters=output_parameters or [],
        category=category,
        author=author
    )


def create_workflow_step(name: str, tool: str, inputs: Dict[str, Any] = None,
                        outputs: List[str] = None, description: str = None) -> WorkflowStep:
    """Create a new workflow step"""
    return WorkflowStep(
        name=name,
        tool=tool,
        inputs=inputs or {},
        outputs=outputs or [],
        description=description
    )


def create_execution_record(workflow_id: str, workflow_name: str, inputs: Dict[str, Any],
                          executed_by: str = "system", session_id: str = None) -> WorkflowExecution:
    """Create a new execution record"""
    return WorkflowExecution(
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        inputs=inputs,
        executed_by=executed_by,
        session_id=session_id
    )


def validate_workflow_structure(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Validate workflow structure and dependencies"""
    issues = []
    warnings = []
    
    # Check for empty steps
    if not workflow.steps:
        issues.append("Workflow must have at least one step")
    
    # Check step dependencies
    for i, step in enumerate(workflow.steps):
        if not step.tool:
            issues.append(f"Step {i+1} has no tool specified")
        
        if not step.name:
            issues.append(f"Step {i+1} has no name")
    
    # Check for circular dependencies
    step_dependencies = {}
    for i, step in enumerate(workflow.steps):
        dependencies = []
        for input_name, input_def in step.inputs.items():
            if isinstance(input_def, dict) and input_def.get('type') == 'ref':
                ref_step = input_def.get('step', -1)
                if 0 <= ref_step < i:
                    dependencies.append(ref_step)
        step_dependencies[i] = dependencies
    
    # Check for circular references
    for step_idx, deps in step_dependencies.items():
        for dep in deps:
            if step_idx in step_dependencies.get(dep, []):
                issues.append(f"Circular dependency detected between steps {step_idx+1} and {dep+1}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }


def get_default_workflow_configuration() -> WorkflowConfiguration:
    """Get default workflow configuration"""
    return WorkflowConfiguration()


def create_workflow_registry() -> WorkflowRegistry:
    """Create empty workflow registry"""
    return WorkflowRegistry()


# ==============================================================================
# ENHANCED WORKFLOW MODELS
# ==============================================================================

class WorkflowTemplate(BaseModel):
    """Workflow template for reusable workflow patterns"""
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Template ID")
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    category: str = Field(description="Template category")
    version: str = Field(default="1.0.0", description="Template version")
    author: str = Field(description="Template author")
    tags: List[str] = Field(default=[], description="Template tags")
    workflow_definition: WorkflowDefinition = Field(description="Base workflow definition")
    parameters: List[Dict[str, Any]] = Field(default=[], description="Template parameters")
    usage_count: int = Field(default=0, description="Number of times used")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_public: bool = Field(default=False, description="Whether template is public")
    
    @validator('version')
    def validate_version(cls, v):
        version_pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(version_pattern, v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.0.0)")
        return v


class WorkflowSchedule(BaseModel):
    """Workflow scheduling configuration"""
    schedule_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Schedule ID")
    workflow_id: str = Field(description="Workflow ID")
    name: str = Field(description="Schedule name")
    cron_expression: str = Field(description="Cron expression for scheduling")
    timezone: str = Field(default="UTC", description="Timezone for scheduling")
    is_active: bool = Field(default=True, description="Whether schedule is active")
    next_run: Optional[datetime] = Field(default=None, description="Next scheduled run")
    last_run: Optional[datetime] = Field(default=None, description="Last run timestamp")
    run_count: int = Field(default=0, description="Number of times executed")
    max_runs: Optional[int] = Field(default=None, description="Maximum number of runs")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(description="Creator user ID")
    
    @validator('cron_expression')
    def validate_cron_expression(cls, v):
        # Basic cron validation (5 fields: minute hour day month weekday)
        cron_parts = v.strip().split()
        if len(cron_parts) != 5:
            raise ValueError("Cron expression must have 5 parts: minute hour day month weekday")
        return v


class WorkflowTrigger(BaseModel):
    """Workflow trigger configuration"""
    trigger_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Trigger ID")
    workflow_id: str = Field(description="Workflow ID")
    trigger_type: Literal["webhook", "file", "database", "api", "manual"] = Field(description="Trigger type")
    name: str = Field(description="Trigger name")
    configuration: Dict[str, Any] = Field(description="Trigger configuration")
    is_active: bool = Field(default=True, description="Whether trigger is active")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(description="Creator user ID")
    
    @validator('trigger_type')
    def validate_trigger_type(cls, v):
        valid_types = ["webhook", "file", "database", "api", "manual"]
        if v not in valid_types:
            raise ValueError(f"Trigger type must be one of: {valid_types}")
        return v


class WorkflowMetrics(BaseModel):
    """Workflow performance metrics"""
    workflow_id: str = Field(description="Workflow ID")
    total_executions: int = Field(default=0, description="Total executions")
    successful_executions: int = Field(default=0, description="Successful executions")
    failed_executions: int = Field(default=0, description="Failed executions")
    average_execution_time: float = Field(default=0.0, description="Average execution time in seconds")
    total_execution_time: float = Field(default=0.0, description="Total execution time in seconds")
    last_execution: Optional[datetime] = Field(default=None, description="Last execution timestamp")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    most_common_errors: List[Dict[str, Any]] = Field(default=[], description="Most common errors")
    performance_trends: List[Dict[str, Any]] = Field(default=[], description="Performance trends")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def calculate_rates(self):
        """Calculate success and error rates"""
        if self.total_executions > 0:
            self.success_rate = (self.successful_executions / self.total_executions) * 100
            self.error_rate = (self.failed_executions / self.total_executions) * 100
        else:
            self.success_rate = 0.0
            self.error_rate = 0.0


class WorkflowDependencyGraph(BaseModel):
    """Workflow dependency graph for execution planning"""
    workflow_id: str = Field(description="Workflow ID")
    nodes: List[Dict[str, Any]] = Field(description="Graph nodes (steps)")
    edges: List[Dict[str, Any]] = Field(description="Graph edges (dependencies)")
    execution_order: List[str] = Field(description="Topological execution order")
    parallel_groups: List[List[str]] = Field(description="Steps that can run in parallel")
    critical_path: List[str] = Field(description="Critical path for execution")
    estimated_duration: float = Field(default=0.0, description="Estimated total duration")
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_dependency(self, from_step: str, to_step: str, dependency_type: str = "data"):
        """Add a dependency between steps"""
        edge = {
            "from": from_step,
            "to": to_step,
            "type": dependency_type,
            "id": f"{from_step}_{to_step}"
        }
        if edge not in self.edges:
            self.edges.append(edge)
    
    def remove_dependency(self, from_step: str, to_step: str):
        """Remove a dependency between steps"""
        self.edges = [e for e in self.edges if not (e["from"] == from_step and e["to"] == to_step)]
    
    def get_dependencies(self, step_id: str) -> List[str]:
        """Get all dependencies for a step"""
        return [e["from"] for e in self.edges if e["to"] == step_id]
    
    def get_dependents(self, step_id: str) -> List[str]:
        """Get all steps that depend on this step"""
        return [e["to"] for e in self.edges if e["from"] == step_id]


class WorkflowExecutionPlan(BaseModel):
    """Workflow execution plan with resource allocation"""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Plan ID")
    workflow_id: str = Field(description="Workflow ID")
    execution_id: str = Field(description="Execution ID")
    steps: List[Dict[str, Any]] = Field(description="Planned step executions")
    resource_allocation: Dict[str, Any] = Field(description="Resource allocation plan")
    estimated_duration: float = Field(description="Estimated total duration")
    estimated_cost: float = Field(default=0.0, description="Estimated execution cost")
    priority: int = Field(default=0, description="Execution priority")
    created_at: datetime = Field(default_factory=datetime.now)
    scheduled_start: Optional[datetime] = Field(default=None, description="Scheduled start time")
    actual_start: Optional[datetime] = Field(default=None, description="Actual start time")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="Plan status")


# ==============================================================================
# ENHANCED UTILITY FUNCTIONS
# ==============================================================================

def create_workflow_template(name: str, description: str, workflow_definition: WorkflowDefinition,
                           category: str = "General", author: str = "Unknown") -> WorkflowTemplate:
    """Create a new workflow template"""
    return WorkflowTemplate(
        name=name,
        description=description,
        category=category,
        author=author,
        workflow_definition=workflow_definition
    )


def create_workflow_schedule(workflow_id: str, name: str, cron_expression: str,
                           timezone: str = "UTC", created_by: str = "system") -> WorkflowSchedule:
    """Create a new workflow schedule"""
    return WorkflowSchedule(
        workflow_id=workflow_id,
        name=name,
        cron_expression=cron_expression,
        timezone=timezone,
        created_by=created_by
    )


def create_workflow_trigger(workflow_id: str, trigger_type: str, name: str,
                          configuration: Dict[str, Any], created_by: str = "system") -> WorkflowTrigger:
    """Create a new workflow trigger"""
    return WorkflowTrigger(
        workflow_id=workflow_id,
        trigger_type=trigger_type,
        name=name,
        configuration=configuration,
        created_by=created_by
    )


def build_dependency_graph(workflow: WorkflowDefinition) -> WorkflowDependencyGraph:
    """Build dependency graph from workflow definition"""
    graph = WorkflowDependencyGraph(workflow_id=workflow.id)
    
    # Add nodes (steps)
    for step in workflow.steps:
        node = {
            "id": step.step_id,
            "name": step.name,
            "tool": step.tool,
            "enabled": step.enabled,
            "priority": step.priority,
            "parallel_execution": step.parallel_execution
        }
        graph.nodes.append(node)
    
    # Add edges (dependencies)
    for step in workflow.steps:
        for dep_id in step.dependencies:
            graph.add_dependency(dep_id, step.step_id)
    
    # Calculate execution order (topological sort)
    graph.execution_order = topological_sort(graph)
    
    # Find parallel groups
    graph.parallel_groups = find_parallel_groups(graph)
    
    # Calculate critical path
    graph.critical_path = calculate_critical_path(graph)
    
    return graph


def topological_sort(graph: WorkflowDependencyGraph) -> List[str]:
    """Perform topological sort to determine execution order"""
    # Kahn's algorithm for topological sorting
    in_degree = {node["id"]: 0 for node in graph.nodes}
    
    # Calculate in-degrees
    for edge in graph.edges:
        in_degree[edge["to"]] += 1
    
    # Find nodes with no incoming edges
    queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        # Sort by priority (higher priority first)
        queue.sort(key=lambda x: next((n["priority"] for n in graph.nodes if n["id"] == x), 0), reverse=True)
        
        current = queue.pop(0)
        result.append(current)
        
        # Remove current node and update in-degrees
        for edge in graph.edges:
            if edge["from"] == current:
                in_degree[edge["to"]] -= 1
                if in_degree[edge["to"]] == 0:
                    queue.append(edge["to"])
    
    return result


def find_parallel_groups(graph: WorkflowDependencyGraph) -> List[List[str]]:
    """Find groups of steps that can run in parallel"""
    parallel_groups = []
    remaining_steps = set(node["id"] for node in graph.nodes)
    
    while remaining_steps:
        # Find steps with no dependencies or all dependencies satisfied
        current_group = []
        for step_id in list(remaining_steps):
            dependencies = graph.get_dependencies(step_id)
            if not dependencies or all(dep not in remaining_steps for dep in dependencies):
                current_group.append(step_id)
        
        if not current_group:
            # Circular dependency or error
            break
        
        parallel_groups.append(current_group)
        remaining_steps -= set(current_group)
    
    return parallel_groups


def calculate_critical_path(graph: WorkflowDependencyGraph) -> List[str]:
    """Calculate critical path for workflow execution"""
    # Simple implementation - longest path through the graph
    # In a real implementation, this would consider step execution times
    if not graph.nodes:
        return []
    
    # For now, return the execution order as critical path
    return graph.execution_order


def validate_workflow_dependencies(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Validate workflow dependencies for circular references and other issues"""
    issues = []
    warnings = []
    
    # Build dependency graph
    graph = build_dependency_graph(workflow)
    
    # Check for circular dependencies
    visited = set()
    rec_stack = set()
    
    def has_cycle(node_id):
        visited.add(node_id)
        rec_stack.add(node_id)
        
        for edge in graph.edges:
            if edge["from"] == node_id:
                neighbor = edge["to"]
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
        
        rec_stack.remove(node_id)
        return False
    
    for node in graph.nodes:
        if node["id"] not in visited:
            if has_cycle(node["id"]):
                issues.append(f"Circular dependency detected involving step: {node['name']}")
    
    # Check for orphaned steps
    all_step_ids = {step.step_id for step in workflow.steps}
    referenced_steps = set()
    
    for step in workflow.steps:
        for dep_id in step.dependencies:
            if dep_id not in all_step_ids:
                issues.append(f"Step '{step.name}' references non-existent dependency: {dep_id}")
            referenced_steps.add(dep_id)
    
    # Check for unused steps
    for step in workflow.steps:
        if step.step_id not in referenced_steps and len(workflow.steps) > 1:
            warnings.append(f"Step '{step.name}' is not referenced by any other step")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "dependency_graph": graph
    }


def optimize_workflow_execution(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Optimize workflow for better execution performance"""
    graph = build_dependency_graph(workflow)
    
    optimizations = []
    
    # Identify steps that can run in parallel
    parallel_groups = find_parallel_groups(graph)
    if len(parallel_groups) > 1:
        optimizations.append({
            "type": "parallel_execution",
            "description": f"Found {len(parallel_groups)} groups of steps that can run in parallel",
            "groups": parallel_groups
        })
    
    # Identify high-priority steps
    high_priority_steps = [step for step in workflow.steps if step.priority > 50]
    if high_priority_steps:
        optimizations.append({
            "type": "priority_optimization",
            "description": f"Found {len(high_priority_steps)} high-priority steps",
            "steps": [step.name for step in high_priority_steps]
        })
    
    # Identify steps with long timeouts
    long_timeout_steps = [step for step in workflow.steps if step.timeout and step.timeout > 300]
    if long_timeout_steps:
        optimizations.append({
            "type": "timeout_optimization",
            "description": f"Found {len(long_timeout_steps)} steps with long timeouts",
            "steps": [step.name for step in long_timeout_steps]
        })
    
    return {
        "optimizations": optimizations,
        "estimated_improvement": len(optimizations) * 10,  # Placeholder
        "recommendations": [
            "Consider enabling parallel execution for independent steps",
            "Review timeout settings for better resource utilization",
            "Optimize step priorities for critical path execution"
        ]
    }
