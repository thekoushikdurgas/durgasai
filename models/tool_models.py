"""
Tool Models and Data Structures for AI Agent Dashboard

This module provides Pydantic models and data structures for the tool execution
system, including tool definitions, execution records, and configuration models.

Features:
- Tool definition schemas
- Execution tracking models
- Security and validation models
- Configuration and settings
- Type definitions for tool operations
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Any, Optional, Union, Callable, Literal
from datetime import datetime, timedelta
from enum import Enum
import uuid
import re
import ast
import hashlib
import json
import time

import csv
import io
import xml.etree.ElementTree as ET
import yaml

class ToolStatus(str, Enum):
    """Tool status enumeration"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    LOADING = "loading"


class ExecutionStatus(str, Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SecurityLevel(str, Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    item_type: Optional[str] = Field(default=None, description="Item type for arrays")
    format: Optional[str] = Field(default=None, description="Parameter format")
    default_value: Optional[Any] = Field(default=None, description="Default value for the parameter")


class ToolDefinitionJson(BaseModel):
    """Tool definition schema for JSON files"""
    id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    author: str = Field(default="Unknown", description="Tool author")
    category: str = Field(default="utility", description="Tool category")
    status: str = Field(default="enabled", description="Tool status (enabled/disabled)")
    security_level: str = Field(default="medium", description="Security level (low/medium/high/critical)")
    function_name: str = Field(description="Python function name")
    input_parameters: List[ToolParameter] = Field(default=[], description="Input parameters")
    output_parameters: List[ToolParameter] = Field(default=[], description="Output parameters")
    code: str = Field(description="Code filename")
    dependencies: List[str] = Field(default=[], description="Required dependencies")
    tags: List[str] = Field(default=[], description="Tool tags")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last update timestamp")
    last_executed: Optional[str] = Field(default=None, description="Last execution timestamp")
    execution_count: int = Field(default=0, description="Number of executions")
    success_count: int = Field(default=0, description="Successful executions")
    failure_count: int = Field(default=0, description="Failed executions")
    average_execution_time: float = Field(default=0.0, description="Average execution time in milliseconds")


class ToolDefinition(BaseModel):
    """Tool definition schema"""
    id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    author: str = Field(default="Unknown", description="Tool author")
    category: str = Field(default="General", description="Tool category")
    status: ToolStatus = Field(default=ToolStatus.ENABLED, description="Tool status")
    security_level: SecurityLevel = Field(default=SecurityLevel.MEDIUM, description="Security level")
    
    # Function details
    function_name: str = Field(description="Python function name")
    parameters: Dict[str, Any] = Field(default={}, description="Function parameters schema")
    return_type: str = Field(default="Any", description="Expected return type")
    
    # Code and metadata
    code: str = Field(description="Tool implementation code")
    dependencies: List[str] = Field(default=[], description="Required dependencies")
    tags: List[str] = Field(default=[], description="Tool tags")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_executed: Optional[datetime] = Field(default=None)
    
    # Execution statistics
    execution_count: int = Field(default=0)
    success_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    average_execution_time: float = Field(default=0.0)
    
    @validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            return str(uuid.uuid4())
        return v
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()
    
    @validator('code')
    def validate_code(cls, v):
        if not v or not v.strip():
            raise ValueError("Tool code cannot be empty")
        return v.strip()


class ToolExecution(BaseModel):
    """Tool execution record"""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_id: str = Field(description="Tool identifier")
    tool_name: str = Field(description="Tool name")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    
    # Input and output
    inputs: Dict[str, Any] = Field(default={}, description="Execution inputs")
    outputs: Optional[Any] = Field(default=None, description="Execution outputs")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Timing and performance
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(default=None)
    duration_ms: Optional[float] = Field(default=None)
    
    # Resource usage
    memory_usage_mb: Optional[float] = Field(default=None)
    cpu_usage_percent: Optional[float] = Field(default=None)
    
    # Security and validation
    security_checks_passed: bool = Field(default=True)
    validation_errors: List[str] = Field(default=[])
    
    # Context
    executed_by: str = Field(default="system", description="Who executed the tool")
    session_id: Optional[str] = Field(default=None, description="Session context")
    
    def calculate_duration(self):
        """Calculate execution duration"""
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class ToolExecutionHistory(BaseModel):
    """Tool execution history container"""
    executions: List[ToolExecution] = Field(default=[])
    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    average_duration_ms: float = Field(default=0.0)
    
    def add_execution(self, execution: ToolExecution = None):
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
        if execution.duration_ms is not None:
            total_duration = sum(e.duration_ms or 0 for e in self.executions)
            self.average_duration_ms = total_duration / len(self.executions)
    
    def get_recent_executions(self, limit: int = 10) -> List[ToolExecution]:
        """Get recent executions"""
        return sorted(self.executions, key=lambda x: x.start_time, reverse=True)[:limit]
    
    def get_executions_by_tool(self, tool_id: str) -> List[ToolExecution]:
        """Get executions for specific tool"""
        return [e for e in self.executions if e.tool_id == tool_id]


class ToolSecurityReport(BaseModel):
    """Tool security analysis report"""
    tool_id: str = Field(description="Tool identifier")
    security_level: SecurityLevel = Field(description="Assessed security level")
    risk_score: float = Field(ge=0, le=10, description="Risk score (0-10)")
    
    # Security checks
    has_dangerous_imports: bool = Field(default=False)
    has_file_operations: bool = Field(default=False)
    has_network_operations: bool = Field(default=False)
    has_system_commands: bool = Field(default=False)
    has_eval_usage: bool = Field(default=False)
    has_subprocess_calls: bool = Field(default=False)
    has_os_calls: bool = Field(default=False)
    has_socket_operations: bool = Field(default=False)
    has_database_operations: bool = Field(default=False)
    has_crypto_operations: bool = Field(default=False)
    
    # Code analysis
    code_complexity: Literal["low", "medium", "high"] = Field(default="low", description="Code complexity")
    function_count: int = Field(default=0, description="Number of functions")
    class_count: int = Field(default=0, description="Number of classes")
    line_count: int = Field(default=0, description="Number of lines")
    cyclomatic_complexity: int = Field(default=0, description="Cyclomatic complexity")
    
    # Vulnerabilities
    vulnerabilities: List[str] = Field(default=[], description="Identified vulnerabilities")
    suspicious_patterns: List[str] = Field(default=[], description="Suspicious code patterns")
    code_smells: List[str] = Field(default=[], description="Code quality issues")
    
    # Recommendations
    recommendations: List[str] = Field(default=[], description="Security recommendations")
    warnings: List[str] = Field(default=[], description="Security warnings")
    
    # Validation
    is_safe: bool = Field(default=True, description="Overall safety assessment")
    requires_review: bool = Field(default=False, description="Requires manual review")
    auto_approval: bool = Field(default=False, description="Can be auto-approved")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    analyzer_version: str = Field(default="1.0.0", description="Security analyzer version")
    code_hash: Optional[str] = Field(default=None, description="Code hash for change detection")


class ToolConfiguration(BaseModel):
    """Tool system configuration"""
    # Directories
    tools_directory: str = Field(default="output/tools")
    code_directory: str = Field(default="output/tools/code")
    
    # Execution settings
    max_execution_time: int = Field(default=300, description="Max execution time in seconds")
    max_memory_mb: int = Field(default=512, description="Max memory usage in MB")
    enable_sandboxing: bool = Field(default=True, description="Enable execution sandboxing")
    
    # Security settings
    require_security_review: bool = Field(default=True, description="Require security review for new tools")
    allowed_imports: List[str] = Field(default=[], description="Allowed import modules")
    blocked_imports: List[str] = Field(default=["os", "subprocess", "sys"], description="Blocked import modules")
    
    # History settings
    max_history_size: int = Field(default=1000, description="Maximum history entries")
    history_retention_days: int = Field(default=30, description="History retention in days")
    
    # UI settings
    default_page_size: int = Field(default=20, description="Default pagination size")
    show_advanced_options: bool = Field(default=False, description="Show advanced options by default")


class ToolRegistry(BaseModel):
    """Tool registry container"""
    tools: Dict[str, ToolDefinition] = Field(default={})
    categories: Dict[str, List[str]] = Field(default={})
    tags: Dict[str, List[str]] = Field(default={})  
    
    def get_tools(self) -> Dict[str, ToolDefinition]:
        """Get all tools"""
        return self.tools
    
    def add_tool(self, tool: ToolDefinition):
        """Add tool to registry"""
        self.tools[tool.id] = tool
        
        # Update categories
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        if tool.id not in self.categories[tool.category]:
            self.categories[tool.category].append(tool.id)
        
        # Update tags
        for tag in tool.tags:
            if tag not in self.tags:
                self.tags[tag] = []
            if tool.id not in self.tags[tag]:
                self.tags[tag].append(tool.id)
    
    def remove_tool(self, tool_id: str):
        """Remove tool from registry"""
        if tool_id in self.tools:
            tool = self.tools[tool_id]
            
            # Remove from categories
            if tool.category in self.categories:
                if tool_id in self.categories[tool.category]:
                    self.categories[tool.category].remove(tool_id)
            
            # Remove from tags
            for tag in tool.tags:
                if tag in self.tags and tool_id in self.tags[tag]:
                    self.tags[tag].remove(tool_id)
            
            del self.tools[tool_id]
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get tools by category"""
        tool_ids = self.categories.get(category, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def get_tools_by_tag(self, tag: str) -> List[ToolDefinition]:
        """Get tools by tag"""
        tool_ids = self.tags.get(tag, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by name or description"""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower() or
                any(query_lower in tag.lower() for tag in tool.tags)):
                results.append(tool)
        
        return results


class ToolExecutionResult(BaseModel):
    """Enhanced tool execution result with comprehensive metadata and formatting capabilities"""
    success: bool = Field(description="Whether execution was successful")
    result: Optional[Any] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(description="Execution time in seconds")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    tool_name: Optional[str] = Field(default=None, description="Name of the executed tool")
    timestamp: float = Field(default_factory=time.time, description="Execution timestamp")
    formatted_result: Dict[str, Any] = Field(default_factory=dict, description="Formatted result according to output parameters")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        json_encoders = {
            # Add custom encoders for special types if needed
        }
    
    def __str__(self) -> str:
        """String representation of the result"""
        return (f"ToolExecutionResult(success={self.success}, result={self.result}, "
                f"error={self.error}, execution_time={self.execution_time}, "
                f"tool_name={self.tool_name}, formatted_result={self.formatted_result})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "execution_time_ms": self.execution_time_ms,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "formatted_result": self.formatted_result,
            "memory_usage_mb": self.memory_usage_mb,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    def to_pretty_json(self) -> str:
        """Convert to pretty-printed JSON string"""
        return json.dumps(self.to_dict(), indent=4, default=str)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        return (f"**Success:** {self.success}\n"
                f"**Result:** {self.result}\n"
                f"**Error:** {self.error}\n"
                f"**Execution Time:** {self.execution_time:.3f}s\n"
                f"**Tool Name:** {self.tool_name}\n"
                f"**Formatted Result:** {self.formatted_result}")
    
    def to_html(self) -> str:
        """Convert to HTML format"""
        return (f"<strong>Success:</strong> {self.success}<br>"
                f"<strong>Result:</strong> {self.result}<br>"
                f"<strong>Error:</strong> {self.error}<br>"
                f"<strong>Execution Time:</strong> {self.execution_time:.3f}s<br>"
                f"<strong>Tool Name:</strong> {self.tool_name}<br>"
                f"<strong>Formatted Result:</strong> {self.formatted_result}")
    
    def to_text(self) -> str:
        """Convert to plain text format"""
        return (f"Success: {self.success}\n"
                f"Result: {self.result}\n"
                f"Error: {self.error}\n"
                f"Execution Time: {self.execution_time:.3f}s\n"
                f"Tool Name: {self.tool_name}\n"
                f"Formatted Result: {self.formatted_result}")
    
    def to_yaml(self) -> str:
        """Convert to YAML format"""
        try:
            return yaml.dump(self.to_dict(), default_flow_style=False)
        except ImportError:
            return "YAML module not available"
    
    def to_xml(self) -> str:
        """Convert to XML format"""
        try:
            root = ET.Element("ToolExecutionResult")
            
            for key, value in self.to_dict().items():
                elem = ET.SubElement(root, key)
                elem.text = str(value)
            
            return ET.tostring(root, encoding='unicode')
        except Exception:
            return "XML conversion failed"
    
    def to_csv(self) -> str:
        """Convert to CSV format"""
        try:
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(self.to_dict().keys())
            # Write data
            writer.writerow(self.to_dict().values())
            
            return output.getvalue()
        except Exception:
            return "CSV conversion failed"
    
    @classmethod
    def create_success(cls, result: Any, execution_time: float, tool_name: str = None, 
                      formatted_result: Dict[str, Any] = None, **kwargs) -> 'ToolExecutionResult':
        """Create a successful execution result"""
        return cls(
            success=True,
            result=result,
            execution_time=execution_time,
            execution_time_ms=execution_time * 1000,
            tool_name=tool_name,
            formatted_result=formatted_result or {},
            **kwargs
        )
    
    @classmethod
    def create_error(cls, error: str, execution_time: float = 0.0, tool_name: str = None, **kwargs) -> 'ToolExecutionResult':
        """Create an error execution result"""
        return cls(
            success=False,
            error=error,
            execution_time=execution_time,
            execution_time_ms=execution_time * 1000,
            tool_name=tool_name,
            **kwargs
        )
    
    def is_successful(self) -> bool:
        """Check if execution was successful"""
        return self.success
    
    def has_error(self) -> bool:
        """Check if execution had an error"""
        return not self.success and self.error is not None
    
    def get_result_value(self) -> Any:
        """Get the result value, handling different formats"""
        if self.formatted_result:
            # If we have formatted result, try to extract the main value
            for key, value in self.formatted_result.items():
                if isinstance(value, dict) and 'value' in value:
                    return value['value']
                return value
        return self.result
    
    def get_execution_summary(self) -> str:
        """Get a brief execution summary"""
        if self.success:
            return f"✅ {self.tool_name or 'Tool'} executed successfully in {self.execution_time:.3f}s"
        else:
            return f"❌ {self.tool_name or 'Tool'} failed: {self.error}"


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def convert_json_to_tool_definition(json_tool: ToolDefinitionJson, code_content: str) -> ToolDefinition:
    """Convert JSON tool definition to internal ToolDefinition format with unified parameter handling"""
    
    # Convert input parameters to the internal format with enhanced validation
    parameters = {}
    for param in json_tool.input_parameters:
        if not param.name or not param.name.strip():
            continue  # Skip invalid parameters
            
        param_info = {
            "type": param.type or "string",
            "description": param.description or f"Parameter {param.name}",
            "required": param.required if param.required is not None else False
        }
        
        # Add optional fields only if they have values
        if param.item_type and param.item_type.strip():
            param_info["item_type"] = param.item_type
        if param.format and param.format.strip():
            param_info["format"] = param.format
        if param.default_value is not None:
            param_info["default_value"] = param.default_value
            
        parameters[param.name] = param_info
    
    # Determine return type from output parameters
    return_type = "Any"
    if json_tool.output_parameters:
        # Use the first output parameter type as the main return type
        return_type = json_tool.output_parameters[0].type
    
    # Map status string to ToolStatus enum
    status_map = {
        "enabled": ToolStatus.ENABLED,
        "disabled": ToolStatus.DISABLED,
        "error": ToolStatus.ERROR,
        "loading": ToolStatus.LOADING
    }
    status = status_map.get(json_tool.status.lower(), ToolStatus.ENABLED)
    
    # Map security level string to SecurityLevel enum
    security_map = {
        "low": SecurityLevel.LOW,
        "medium": SecurityLevel.MEDIUM,
        "high": SecurityLevel.HIGH,
        "critical": SecurityLevel.CRITICAL
    }
    security_level = security_map.get(json_tool.security_level.lower(), SecurityLevel.MEDIUM)
    
    # Parse timestamps
    created_at = datetime.fromisoformat(json_tool.created_at) if json_tool.created_at else datetime.now()
    updated_at = datetime.fromisoformat(json_tool.updated_at) if json_tool.updated_at else datetime.now()
    last_executed = datetime.fromisoformat(json_tool.last_executed) if json_tool.last_executed else None
    
    return ToolDefinition(
        id=json_tool.id,
        name=json_tool.name,
        description=json_tool.description,
        version=json_tool.version,
        author=json_tool.author,
        category=json_tool.category,
        status=status,
        security_level=security_level,
        function_name=json_tool.function_name,
        parameters=parameters,
        return_type=return_type,
        code=code_content,
        dependencies=json_tool.dependencies,
        tags=json_tool.tags,
        created_at=created_at,
        updated_at=updated_at,
        last_executed=last_executed,
        execution_count=json_tool.execution_count,
        success_count=json_tool.success_count,
        failure_count=json_tool.failure_count,
        average_execution_time=json_tool.average_execution_time
    )


def create_tool_definition(name: str, description: str, code: str, 
                          function_name: str, parameters: Dict[str, Any] = None,
                          category: str = "General", author: str = "Unknown",
                          version: str = "1.0.0", return_type: str = "Any",
                          security_level: str = "medium", dependencies: List[str] = None,
                          tags: List[str] = None) -> ToolDefinition:
    """Create a new tool definition"""
    
    # Map security level string to enum
    security_map = {
        "low": SecurityLevel.LOW,
        "medium": SecurityLevel.MEDIUM,
        "high": SecurityLevel.HIGH,
        "critical": SecurityLevel.CRITICAL
    }
    security_level_enum = security_map.get(security_level.lower(), SecurityLevel.MEDIUM)
    
    return ToolDefinition(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        version=version,
        code=code,
        function_name=function_name,
        parameters=parameters or {},
        category=category,
        author=author,
        return_type=return_type,
        security_level=security_level_enum,
        dependencies=dependencies or [],
        tags=tags or []
    )


def create_execution_record(tool_id: str, tool_name: str, inputs: Dict[str, Any],
                          executed_by: str = "system", session_id: str = None) -> ToolExecution:
    """Create a new execution record"""
    return ToolExecution(
        tool_id=tool_id,
        tool_name=tool_name,
        inputs=inputs,
        executed_by=executed_by,
        session_id=session_id
    )


def validate_tool_code(code: str) -> Dict[str, Any]:
    """Validate tool code for security and syntax with comprehensive analysis"""
    issues = []
    warnings = []
    vulnerabilities = []
    suspicious_patterns = []
    code_smells = []
    
    try:
        # Parse code for syntax errors
        tree = ast.parse(code)
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")
        return {"valid": False, "issues": issues, "warnings": warnings}
    
    # Basic code metrics
    line_count = len(code.splitlines())
    function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
    class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    
    # Check for dangerous imports
    dangerous_imports = [
        'os', 'subprocess', 'sys', 'shutil', 'glob', 'tempfile',
        'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib',
        'sqlite3', 'psycopg2', 'pymongo', 'redis',
        'pickle', 'marshal', 'shelve',
        'ctypes', 'cffi', 'cryptography'
    ]
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in dangerous_imports:
                    warnings.append(f"Dangerous import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module in dangerous_imports:
                warnings.append(f"Dangerous import from: {node.module}")
    
    # Check for dangerous function calls
    dangerous_functions = [
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'execfile', 'reload',
        'os.system', 'os.popen', 'os.spawn', 'os.exec',
        'subprocess.call', 'subprocess.run', 'subprocess.Popen',
        'pickle.loads', 'pickle.load', 'marshal.loads', 'marshal.load'
    ]
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = get_function_name(node)
            if func_name in dangerous_functions:
                vulnerabilities.append(f"Dangerous function call: {func_name}")
    
    # Check for file operations
    file_operations = ['open', 'file', 'os.path', 'glob.glob', 'os.listdir']
    for pattern in file_operations:
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(f"File operation detected: {pattern}")
    
    # Check for network operations
    network_patterns = [
        r'requests\.', r'urllib\.', r'socket\.', r'http\.',
        r'ftplib\.', r'smtplib\.', r'poplib\.', r'imap'
    ]
    for pattern in network_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(f"Network operation detected: {pattern}")
    
    # Check for database operations
    db_patterns = [
        r'sqlite3\.', r'psycopg2\.', r'pymongo\.', r'redis\.',
        r'mysql\.', r'pymysql\.', r'sqlalchemy\.'
    ]
    for pattern in db_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(f"Database operation detected: {pattern}")
    
    # Check for crypto operations
    crypto_patterns = [
        r'hashlib\.', r'hmac\.', r'cryptography\.', r'Crypto\.',
        r'base64\.', r'binascii\.', r'secrets\.'
    ]
    for pattern in crypto_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(f"Crypto operation detected: {pattern}")
    
    # Check for suspicious patterns
    suspicious_patterns_list = [
        (r'eval\s*\(', "Eval function usage"),
        (r'exec\s*\(', "Exec function usage"),
        (r'__import__\s*\(', "Dynamic import usage"),
        (r'compile\s*\(', "Compile function usage"),
        (r'getattr\s*\(', "Dynamic attribute access"),
        (r'setattr\s*\(', "Dynamic attribute setting"),
        (r'delattr\s*\(', "Dynamic attribute deletion"),
        (r'globals\s*\(', "Global variables access"),
        (r'locals\s*\(', "Local variables access"),
        (r'vars\s*\(', "Variables access"),
        (r'dir\s*\(', "Directory listing"),
        (r'help\s*\(', "Help function usage"),
    ]
    
    for pattern, description in suspicious_patterns_list:
        if re.search(pattern, code, re.IGNORECASE):
            suspicious_patterns.append(description)
    
    # Check for code smells
    if line_count > 100:
        code_smells.append("Function is too long (>100 lines)")
    
    if function_count > 10:
        code_smells.append("Too many functions in single file (>10)")
    
    # Check for hardcoded secrets
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']'
    ]
    
    for pattern in secret_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            code_smells.append("Potential hardcoded secret detected")
    
    # Calculate risk score
    risk_score = 0
    risk_score += len(vulnerabilities) * 3
    risk_score += len(warnings) * 1
    risk_score += len(suspicious_patterns) * 2
    risk_score += len(code_smells) * 1
    
    # Determine security level
    if risk_score >= 10:
        security_level = "critical"
    elif risk_score >= 6:
        security_level = "high"
    elif risk_score >= 3:
        security_level = "medium"
    else:
        security_level = "low"
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "vulnerabilities": vulnerabilities,
        "suspicious_patterns": suspicious_patterns,
        "code_smells": code_smells,
        "metrics": {
            "line_count": line_count,
            "function_count": function_count,
            "class_count": class_count
        },
        "risk_score": min(risk_score, 10),
        "security_level": security_level,
        "requires_review": risk_score >= 3
    }


def get_function_name(node: ast.Call) -> str:
    """Extract function name from AST call node"""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return f"{get_attr_name(node.func.value)}.{node.func.attr}"
    return "unknown"


def get_attr_name(node: ast.AST) -> str:
    """Extract attribute name from AST node"""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{get_attr_name(node.value)}.{node.attr}"
    return "unknown"


def generate_security_report(tool_id: str, code: str) -> ToolSecurityReport:
    """Generate comprehensive security report for a tool"""
    validation_result = validate_tool_code(code)
    
    # Calculate code hash
    code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
    
    # Map security level
    security_level_map = {
        "low": SecurityLevel.LOW,
        "medium": SecurityLevel.MEDIUM,
        "high": SecurityLevel.HIGH,
        "critical": SecurityLevel.CRITICAL
    }
    
    security_level = security_level_map.get(validation_result.get("security_level", "low"), SecurityLevel.LOW)
    
    # Generate recommendations
    recommendations = []
    if validation_result.get("vulnerabilities"):
        recommendations.append("Review and remove dangerous function calls")
    if validation_result.get("suspicious_patterns"):
        recommendations.append("Replace dynamic code execution with safer alternatives")
    if validation_result.get("code_smells"):
        recommendations.append("Refactor code to improve maintainability")
    if validation_result.get("warnings"):
        recommendations.append("Review imports and operations for security implications")
    
    return ToolSecurityReport(
        tool_id=tool_id,
        security_level=security_level,
        risk_score=validation_result.get("risk_score", 0),
        has_dangerous_imports=bool(validation_result.get("warnings")),
        has_file_operations=any("file" in w.lower() for w in validation_result.get("warnings", [])),
        has_network_operations=any("network" in w.lower() for w in validation_result.get("warnings", [])),
        has_system_commands=any("system" in w.lower() for w in validation_result.get("warnings", [])),
        has_eval_usage=any("eval" in w.lower() for w in validation_result.get("suspicious_patterns", [])),
        has_subprocess_calls=any("subprocess" in w.lower() for w in validation_result.get("warnings", [])),
        has_os_calls=any("os" in w.lower() for w in validation_result.get("warnings", [])),
        has_socket_operations=any("socket" in w.lower() for w in validation_result.get("warnings", [])),
        has_database_operations=any("database" in w.lower() for w in validation_result.get("warnings", [])),
        has_crypto_operations=any("crypto" in w.lower() for w in validation_result.get("warnings", [])),
        code_complexity="high" if validation_result.get("metrics", {}).get("line_count", 0) > 100 else "medium" if validation_result.get("metrics", {}).get("line_count", 0) > 50 else "low",
        function_count=validation_result.get("metrics", {}).get("function_count", 0),
        class_count=validation_result.get("metrics", {}).get("class_count", 0),
        line_count=validation_result.get("metrics", {}).get("line_count", 0),
        vulnerabilities=validation_result.get("vulnerabilities", []),
        suspicious_patterns=validation_result.get("suspicious_patterns", []),
        code_smells=validation_result.get("code_smells", []),
        recommendations=recommendations,
        warnings=validation_result.get("warnings", []),
        is_safe=validation_result.get("risk_score", 0) < 3,
        requires_review=validation_result.get("requires_review", False),
        auto_approval=validation_result.get("risk_score", 0) < 1,
        code_hash=code_hash
    )


def get_default_tool_configuration() -> ToolConfiguration:
    """Get default tool configuration"""
    return ToolConfiguration()


def validate_parameter_structure(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize parameter structure for consistency.
    
    Args:
        parameters: Raw parameter dictionary
        
    Returns:
        Validated and normalized parameter dictionary
    """
    validated = {}
    
    for param_name, param_info in parameters.items():
        if not isinstance(param_info, dict):
            continue
            
        # Create normalized parameter structure
        normalized_param = {
            "type": param_info.get("type", "string"),
            "description": param_info.get("description", f"Parameter {param_name}"),
            "required": param_info.get("required", False)
        }
        
        # Add optional fields if they exist and are valid
        if "item_type" in param_info and param_info["item_type"]:
            normalized_param["item_type"] = param_info["item_type"]
        if "format" in param_info and param_info["format"]:
            normalized_param["format"] = param_info["format"]
        if "default_value" in param_info and param_info["default_value"] is not None:
            normalized_param["default_value"] = param_info["default_value"]
            
        validated[param_name] = normalized_param
    
    return validated


def convert_legacy_parameters(legacy_params: Dict[str, Any]) -> List[ToolParameter]:
    """
    Convert legacy parameter format to new ToolParameter list format.
    
    Args:
        legacy_params: Legacy parameter dictionary
        
    Returns:
        List of ToolParameter objects
    """
    parameters = []
    
    for param_name, param_info in legacy_params.items():
        if not isinstance(param_info, dict):
            continue
            
        param = ToolParameter(
            name=param_name,
            type=param_info.get("type", "string"),
            description=param_info.get("description", f"Parameter {param_name}"),
            required=param_info.get("required", False),
            item_type=param_info.get("item_type"),
            format=param_info.get("format"),
            default_value=param_info.get("default_value")
        )
        parameters.append(param)
    
    return parameters


def convert_parameters_to_legacy(parameters: List[ToolParameter]) -> Dict[str, Any]:
    """
    Convert ToolParameter list to legacy dictionary format.
    
    Args:
        parameters: List of ToolParameter objects
        
    Returns:
        Legacy parameter dictionary
    """
    legacy_params = {}
    
    for param in parameters:
        param_info = {
            "type": param.type,
            "description": param.description,
            "required": param.required
        }
        
        if param.item_type:
            param_info["item_type"] = param.item_type
        if param.format:
            param_info["format"] = param.format
        if param.default_value is not None:
            param_info["default_value"] = param.default_value
            
        legacy_params[param.name] = param_info
    
    return legacy_params


# def create_tool_registry() -> ToolRegistry:
#     """Create empty tool registry"""
#     return 
