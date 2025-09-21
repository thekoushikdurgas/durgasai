"""
Documentation generation utilities for DurgasAI custom models.

This module provides tools for generating consistent, professional documentation
for custom models using HuggingFace's @auto_docstring patterns.
"""

import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

from .logger import debug, info, warning, error


@dataclass
class DocstringInfo:
    """Information about a docstring for a class or function."""
    name: str
    type: str  # 'class' or 'function'
    signature: str
    arguments: List[Dict[str, Any]]
    return_type: Optional[str] = None
    custom_docstring: Optional[str] = None
    auto_docstring_applied: bool = False


class DocstringGenerator:
    """Generator for consistent model documentation using @auto_docstring patterns."""
    
    def __init__(self):
        self.standard_arguments = {
            'input_ids': '`torch.LongTensor` of shape `(batch_size, sequence_length)`',
            'attention_mask': '`torch.Tensor` of shape `(batch_size, sequence_length)`',
            'token_type_ids': '`torch.LongTensor` of shape `(batch_size, sequence_length)`',
            'position_ids': '`torch.LongTensor` of shape `(batch_size, sequence_length)`',
            'head_mask': '`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`',
            'inputs_embeds': '`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`',
            'output_attentions': '`bool`, *optional*',
            'output_hidden_states': '`bool`, *optional*',
            'return_dict': '`bool`, *optional*',
            'labels': '`torch.LongTensor` of shape `(batch_size,)`',
            'pixel_values': '`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`',
            'use_cache': '`bool`, *optional*',
            'past_key_values': '`tuple(tuple(torch.FloatTensor))`, *optional*',
        }
        
        self.standard_descriptions = {
            'input_ids': 'Indices of input sequence tokens in the vocabulary.',
            'attention_mask': 'Mask to avoid performing attention on padding token indices.',
            'token_type_ids': 'Segment token indices to indicate first and second portions of the inputs.',
            'position_ids': 'Indices of positions of each input sequence tokens in the position embeddings.',
            'head_mask': 'Mask to nullify selected heads of the self-attention modules.',
            'inputs_embeds': 'Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.',
            'output_attentions': 'Whether or not to return the attentions tensors of all attention layers.',
            'output_hidden_states': 'Whether or not to return the hidden states of all layers.',
            'return_dict': 'Whether or not to return a `ModelOutput` instead of a plain tuple.',
            'labels': 'Labels for computing the (language modeling, classification, etc.) loss.',
            'pixel_values': 'Pixel values.',
            'use_cache': 'If set to `True`, `past_key_values` key value states are returned.',
            'past_key_values': 'Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`.',
        }
        
        debug("DocstringGenerator initialized", "documentation")
    
    def generate_class_docstring(self, class_info: DocstringInfo) -> str:
        """
        Generate docstring for a model class.
        
        Args:
            class_info: Information about the class
            
        Returns:
            Generated docstring content
        """
        try:
            debug(f"Generating docstring for class {class_info.name}", "documentation")
            
            docstring_parts = []
            
            # Add class description
            if class_info.custom_docstring:
                docstring_parts.append(class_info.custom_docstring)
            else:
                docstring_parts.append(f"{class_info.name} model implementation.")
            
            # Add arguments section
            if class_info.arguments:
                docstring_parts.append("")
                docstring_parts.append("Args:")
                
                for arg in class_info.arguments:
                    arg_doc = self._format_argument_docstring(arg)
                    docstring_parts.append(f"    {arg_doc}")
            
            return "\n".join(docstring_parts)
            
        except Exception as e:
            error(f"Error generating class docstring: {str(e)}", "documentation", e)
            return f"{class_info.name} model implementation."
    
    def generate_function_docstring(self, func_info: DocstringInfo) -> str:
        """
        Generate docstring for a function.
        
        Args:
            func_info: Information about the function
            
        Returns:
            Generated docstring content
        """
        try:
            debug(f"Generating docstring for function {func_info.name}", "documentation")
            
            docstring_parts = []
            
            # Add function description
            if func_info.custom_docstring:
                docstring_parts.append(func_info.custom_docstring)
            else:
                docstring_parts.append(f"{func_info.name} method.")
            
            # Add arguments section
            if func_info.arguments:
                docstring_parts.append("")
                docstring_parts.append("Args:")
                
                for arg in func_info.arguments:
                    arg_doc = self._format_argument_docstring(arg)
                    docstring_parts.append(f"    {arg_doc}")
            
            # Add returns section
            if func_info.return_type:
                docstring_parts.append("")
                docstring_parts.append("Returns:")
                docstring_parts.append(f"    `{func_info.return_type}`: {self._get_return_description(func_info.return_type)}")
            
            return "\n".join(docstring_parts)
            
        except Exception as e:
            error(f"Error generating function docstring: {str(e)}", "documentation", e)
            return f"{func_info.name} method."
    
    def _format_argument_docstring(self, arg: Dict[str, Any]) -> str:
        """Format a single argument for docstring."""
        name = arg['name']
        arg_type = arg.get('type', 'Any')
        default = arg.get('default')
        description = arg.get('description', '')
        
        # Check if it's a standard argument
        if name in self.standard_arguments:
            type_str = self.standard_arguments[name]
            if not description:
                description = self.standard_descriptions[name]
        else:
            type_str = f"`{arg_type}`"
        
        # Build the argument line
        parts = [f"{name} ({type_str})"]
        
        # Add optional marker
        if default is not None or arg.get('optional', False):
            parts.append("*optional*")
        
        # Add default value
        if default is not None and default != 'None':
            if isinstance(default, str):
                parts.append(f"defaults to `\"{default}\"`")
            else:
                parts.append(f"defaults to `{default}`")
        
        # Add description
        if description:
            parts.append(f": {description}")
        
        return ": ".join(parts)
    
    def _get_return_description(self, return_type: str) -> str:
        """Get description for return type."""
        descriptions = {
            'torch.FloatTensor': 'Model outputs',
            'CausalLMOutput': 'Causal language modeling output',
            'ModelOutput': 'Model output',
            'Tuple': 'Tuple of outputs',
            'None': 'No return value'
        }
        
        return descriptions.get(return_type, 'Model outputs')
    
    def analyze_python_file(self, file_path: Path) -> List[DocstringInfo]:
        """
        Analyze a Python file to extract docstring information.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of DocstringInfo objects
        """
        try:
            debug(f"Analyzing Python file: {file_path}", "documentation")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            docstring_infos = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    info = self._extract_docstring_info(node, content)
                    if info:
                        docstring_infos.append(info)
            
            info(f"Analyzed {file_path.name}: found {len(docstring_infos)} classes/functions", "documentation")
            return docstring_infos
            
        except Exception as e:
            error(f"Error analyzing file {file_path}: {str(e)}", "documentation", e)
            return []
    
    def _extract_docstring_info(self, node: ast.AST, content: str) -> Optional[DocstringInfo]:
        """
        Extract docstring information from an AST node.
        
        This method performs comprehensive analysis of Python AST nodes to extract
        all information needed for generating proper docstrings. It handles both
        class and function definitions and extracts metadata for documentation.
        
        Args:
            node: AST node (ClassDef or FunctionDef)
            content: Source code content for context
            
        Returns:
            DocstringInfo object with extracted information, or None if extraction fails
        """
        try:
            debug(f"Extracting docstring info from AST node: {node.name}", "documentation",
                  node_type=type(node).__name__,
                  line_number=getattr(node, 'lineno', 'unknown'))
            
            # Step 1: Get basic node information
            name = node.name
            node_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
            debug(f"Node basic info extracted", "documentation",
                  name=name, type=node_type)
            
            # Step 2: Extract function/class signature
            # This provides the complete method signature for documentation
            debug("Extracting signature information", "documentation")
            signature = self._get_signature(node, content)
            debug(f"Signature extracted: {signature[:100]}...", "documentation")
            
            # Step 3: Extract argument information for functions
            # This includes parameter names, types, defaults, and descriptions
            arguments = []
            if hasattr(node, 'args'):
                debug("Extracting argument information", "documentation")
                arguments = self._extract_arguments(node)
                debug(f"Extracted {len(arguments)} arguments", "documentation",
                      argument_names=[arg['name'] for arg in arguments])
            
            # Step 4: Extract return type annotation for functions
            # This helps generate proper return type documentation
            return_type = None
            if isinstance(node, ast.FunctionDef) and node.returns:
                debug("Extracting return type annotation", "documentation")
                return_type = self._get_type_annotation(node.returns)
                debug(f"Return type extracted: {return_type}", "documentation")
            
            # Step 5: Check for @auto_docstring decorator
            # This determines if HuggingFace auto-documentation is already applied
            debug("Checking for @auto_docstring decorator", "documentation")
            auto_docstring_applied = self._has_auto_docstring_decorator(node)
            debug(f"Auto docstring decorator present: {auto_docstring_applied}", "documentation")
            
            # Step 6: Extract existing docstring content
            # This preserves any existing documentation
            debug("Extracting existing docstring", "documentation")
            custom_docstring = self._extract_existing_docstring(node)
            debug(f"Existing docstring found: {custom_docstring is not None}", "documentation",
                  docstring_length=len(custom_docstring) if custom_docstring else 0)
            
            # Step 7: Create comprehensive DocstringInfo object
            docstring_info = DocstringInfo(
                name=name,
                type=node_type,
                signature=signature,
                arguments=arguments,
                return_type=return_type,
                custom_docstring=custom_docstring,
                auto_docstring_applied=auto_docstring_applied
            )
            
            debug(f"DocstringInfo created successfully for {name}", "documentation",
                  has_arguments=len(arguments) > 0,
                  has_return_type=return_type is not None,
                  has_existing_docstring=custom_docstring is not None)
            
            return docstring_info
            
        except Exception as e:
            warning(f"Error extracting docstring info for {node.name}: {str(e)}", "documentation",
                   node_name=getattr(node, 'name', 'unknown'),
                   node_type=type(node).__name__,
                   error_details=str(e))
            return None
    
    def _get_signature(self, node: ast.AST, content: str) -> str:
        """Get the signature string for a node."""
        # This is a simplified implementation
        # In practice, you'd use inspect or ast parsing to get the full signature
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
        elif isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            return f"def {node.name}({', '.join(args)})"
        return f"{node.name}()"
    
    def _extract_arguments(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract argument information from a function definition."""
        arguments = []
        
        for i, arg in enumerate(node.args.args):
            arg_info = {
                'name': arg.arg,
                'type': 'Any',
                'optional': False,
                'default': None,
                'description': ''
            }
            
            # Get type annotation
            if arg.annotation:
                arg_info['type'] = self._get_type_annotation(arg.annotation)
            
            # Check if optional (has default or is after *args)
            if i >= len(node.args.args) - len(node.args.defaults):
                arg_info['optional'] = True
                default_idx = i - (len(node.args.args) - len(node.args.defaults))
                if default_idx >= 0 and default_idx < len(node.args.defaults):
                    arg_info['default'] = self._get_default_value(node.args.defaults[default_idx])
            
            arguments.append(arg_info)
        
        return arguments
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Convert AST type annotation to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Str):  # Python < 3.8
            return annotation.s
        else:
            return 'Any'
    
    def _get_default_value(self, default: ast.AST) -> Any:
        """Get default value from AST node."""
        if isinstance(default, ast.Constant):
            return default.value
        elif isinstance(default, ast.Str):  # Python < 3.8
            return default.s
        elif isinstance(default, ast.Num):  # Python < 3.8
            return default.n
        elif isinstance(default, ast.NameConstant):  # Python < 3.8
            return default.value
        else:
            return None
    
    def _has_auto_docstring_decorator(self, node: ast.AST) -> bool:
        """Check if node has @auto_docstring decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'auto_docstring':
                return True
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id == 'auto_docstring':
                    return True
        return False
    
    def _extract_existing_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract existing docstring from a node."""
        if not node.body or not isinstance(node.body[0], ast.Expr):
            return None
        
        expr = node.body[0]
        if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
            return expr.value.value
        elif isinstance(expr.value, ast.Str):  # Python < 3.8
            return expr.value.s
        
        return None


class DocstringValidator:
    """Validator for model documentation completeness and correctness."""
    
    def __init__(self):
        self.generator = DocstringGenerator()
        debug("DocstringValidator initialized", "documentation")
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate documentation in a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Validation results
        """
        try:
            debug(f"Validating documentation in {file_path}", "documentation")
            
            docstring_infos = self.generator.analyze_python_file(file_path)
            
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'checks': {},
                'suggestions': []
            }
            
            for info in docstring_infos:
                self._validate_docstring_info(info, validation_results)
            
            # Overall validation
            validation_results['valid'] = len(validation_results['errors']) == 0
            
            info(f"Validation complete for {file_path.name}: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings", "documentation")
            return validation_results
            
        except Exception as e:
            error(f"Error validating file {file_path}: {str(e)}", "documentation", e)
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'checks': {},
                'suggestions': []
            }
    
    def _validate_docstring_info(self, info: DocstringInfo, results: Dict[str, Any]):
        """Validate a single docstring info object."""
        # Check if @auto_docstring is applied
        if not info.auto_docstring_applied:
            if info.type == 'class' and 'Model' in info.name:
                results['warnings'].append(f"Class {info.name} should have @auto_docstring decorator")
        
        # Check for missing custom argument documentation
        for arg in info.arguments:
            if arg['name'] not in self.generator.standard_arguments and not arg.get('description'):
                results['warnings'].append(f"Custom argument '{arg['name']}' in {info.name} lacks documentation")
        
        # Check for type annotations
        for arg in info.arguments:
            if arg['type'] == 'Any':
                results['suggestions'].append(f"Consider adding type annotation for '{arg['name']}' in {info.name}")
        
        # Check return type documentation
        if info.type == 'function' and info.return_type and not info.return_type == 'None':
            if not info.custom_docstring or 'Returns:' not in info.custom_docstring:
                results['suggestions'].append(f"Function {info.name} should document return type")


# Global instances
docstring_generator = DocstringGenerator()
docstring_validator = DocstringValidator()
