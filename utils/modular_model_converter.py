"""
Modular Model Converter Utility.

This utility converts modular model files to single-file implementations,
following the HuggingFace Transformers modular approach.

Usage:
    python utils/modular_model_converter.py model_name
"""

import os
import sys
import ast
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, warning, error, log_session_event


class ModularModelConverter:
    """
    Advanced converter for modular model files to single-file implementations.
    
    This class handles the complex process of converting modular Transformers
    model files into the traditional single-file format required by the library.
    """
    
    def __init__(self):
        """Initialize the modular model converter."""
        self.transformers_path = None
        self._find_transformers_path()
        debug("ModularModelConverter initialized", "converter")
    
    def _find_transformers_path(self):
        """Find the transformers library path."""
        try:
            import transformers
            self.transformers_path = Path(transformers.__file__).parent
            debug(f"Found transformers at: {self.transformers_path}", "converter")
        except ImportError:
            warning("Transformers library not found", "converter")
    
    def convert_model(self, model_name: str, input_dir: str = None, output_dir: str = None) -> bool:
        """
        Convert a modular model to single-file implementation.
        
        Args:
            model_name: Name of the model to convert (snake_case)
            input_dir: Directory containing modular files (optional)
            output_dir: Output directory for converted files (optional)
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            info(f"Starting conversion of modular model: {model_name}", "converter",
                 input_dir=str(input_dir) if input_dir else "default",
                 output_dir=str(output_dir) if output_dir else "default")
            
            # Step 1: Set up directory paths with defaults
            # Input directory contains the modular model files to convert
            # Output directory will contain the generated single-file implementations
            debug("Setting up directory paths", "converter")
            
            if input_dir is None:
                input_dir = Path("output/modular_models")
                debug("Using default input directory", "converter", path=str(input_dir))
            else:
                input_dir = Path(input_dir)
                debug("Using provided input directory", "converter", path=str(input_dir))
            
            if output_dir is None:
                output_dir = Path("output/models")
                debug("Using default output directory", "converter", path=str(output_dir))
            else:
                output_dir = Path(output_dir)
                debug("Using provided output directory", "converter", path=str(output_dir))
            
            # Step 2: Ensure directories exist
            # Create directories if they don't exist to prevent file operation errors
            debug("Creating directories if needed", "converter")
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            debug("Directories prepared", "converter",
                  input_dir_exists=input_dir.exists(),
                  output_dir_exists=output_dir.exists())
            
            # Step 3: Locate and validate modular file
            # The modular file contains the model definition to be converted
            modular_file = input_dir / f"modular_{model_name}.py"
            debug(f"Looking for modular file: {modular_file}", "converter")
            
            if not modular_file.exists():
                error(f"Modular file not found: {modular_file}", "converter",
                      expected_path=str(modular_file),
                      input_dir_contents=list(input_dir.glob("*.py")))
                return False
            
            debug("Modular file found", "converter",
                  file_size=modular_file.stat().st_size,
                  file_path=str(modular_file))
            
            # Step 4: Parse modular file content
            # Extract classes, functions, and metadata from the modular file
            debug("Parsing modular file content", "converter")
            modular_content = self._parse_modular_file(modular_file)
            
            if not modular_content:
                error("Failed to parse modular file", "converter",
                      file_path=str(modular_file))
                return False
            
            debug("Modular file parsed successfully", "converter",
                  content_keys=list(modular_content.keys()) if modular_content else [])
            
            # Step 5: Convert to single-file format
            # Transform modular structure into standard HuggingFace single-file format
            debug("Converting to single-file format", "converter")
            converted_files = self._convert_to_single_files(modular_content, model_name)
            debug("Conversion completed", "converter",
                  files_generated=list(converted_files.keys()) if converted_files else [])
            
            # Step 6: Write converted files to output directory
            debug("Writing converted files", "converter", output_dir=str(output_dir))
            success = self._write_converted_files(converted_files, output_dir, model_name)
            
            if success:
                info(f"Successfully converted modular model: {model_name}", "converter")
                log_session_event("modular_model_converted", 
                                model_name=model_name,
                                files_generated=len(converted_files))
                return True
            else:
                error("Failed to write converted files", "converter")
                return False
                
        except Exception as e:
            error(f"Error converting model: {str(e)}", "converter", e)
            return False
    
    def _parse_modular_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a modular file and extract its components.
        
        Args:
            file_path: Path to the modular file
            
        Returns:
            Dict containing parsed components or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            parsed = {
                "imports": [],
                "classes": [],
                "functions": [],
                "raw_content": content
            }
            
            # Extract imports, classes, and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        parsed["imports"].append({
                            "type": "import",
                            "name": alias.name,
                            "alias": alias.asname
                        })
                elif isinstance(node, ast.ImportFrom):
                    parsed["imports"].append({
                        "type": "from_import",
                        "module": node.module,
                        "names": [alias.name for alias in node.names],
                        "level": node.level
                    })
                elif isinstance(node, ast.ClassDef):
                    parsed["classes"].append({
                        "name": node.name,
                        "bases": [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
                        "body": ast.unparse(node)
                    })
                elif isinstance(node, ast.FunctionDef):
                    parsed["functions"].append({
                        "name": node.name,
                        "body": ast.unparse(node)
                    })
            
            debug(f"Parsed modular file: {len(parsed['classes'])} classes, {len(parsed['functions'])} functions", "converter")
            return parsed
            
        except Exception as e:
            error(f"Error parsing modular file: {str(e)}", "converter", e)
            return None
    
    def _convert_to_single_files(self, parsed_content: Dict[str, Any], model_name: str) -> Dict[str, str]:
        """
        Convert parsed modular content to single-file implementations.
        
        Args:
            parsed_content: Parsed modular file content
            model_name: Name of the model
            
        Returns:
            Dict of filename -> content mappings
        """
        converted_files = {}
        
        # Convert imports to absolute imports
        converted_imports = self._convert_imports(parsed_content["imports"])
        
        # Generate configuration file
        config_content = self._generate_config_file(parsed_content, model_name, converted_imports)
        converted_files[f"configuration_{model_name}.py"] = config_content
        
        # Generate modeling file
        modeling_content = self._generate_modeling_file(parsed_content, model_name, converted_imports)
        converted_files[f"modeling_{model_name}.py"] = modeling_content
        
        # Generate tokenizer file (if needed)
        tokenizer_content = self._generate_tokenizer_file(parsed_content, model_name, converted_imports)
        converted_files[f"tokenization_{model_name}.py"] = tokenizer_content
        
        # Generate __init__.py file
        init_content = self._generate_init_file(model_name)
        converted_files["__init__.py"] = init_content
        
        return converted_files
    
    def _convert_imports(self, imports: List[Dict[str, Any]]) -> str:
        """Convert relative imports to absolute imports."""
        converted_imports = []
        
        for imp in imports:
            if imp["type"] == "from_import" and imp["level"] > 0:
                # Convert relative import to absolute
                if imp["module"]:
                    # Replace relative import with absolute
                    module_path = imp["module"].replace("..", "transformers.models")
                    converted_imports.append(f"from {module_path} import {', '.join(imp['names'])}")
                else:
                    # Handle imports like "from .. import something"
                    converted_imports.append(f"from transformers.models import {', '.join(imp['names'])}")
            else:
                # Keep absolute imports as-is
                if imp["type"] == "import":
                    converted_imports.append(f"import {imp['name']}")
                elif imp["type"] == "from_import":
                    converted_imports.append(f"from {imp['module']} import {', '.join(imp['names'])}")
        
        # Add standard imports
        standard_imports = [
            "import torch",
            "import torch.nn as nn",
            "from transformers import PretrainedConfig, PreTrainedModel",
            "from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput",
        ]
        
        return "\n".join(standard_imports + converted_imports)
    
    def _generate_config_file(self, parsed_content: Dict[str, Any], model_name: str, imports: str) -> str:
        """Generate configuration file content."""
        config_classes = [cls for cls in parsed_content["classes"] if "Config" in cls["name"]]
        
        if not config_classes:
            # Generate default config if none found
            return f'''"""
Configuration for {model_name.title()} model.
"""

{imports}


class {model_name.title()}Config(PretrainedConfig):
    """
    Configuration class for {model_name.title()}.
    """
    
    model_type = "{model_name}"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
'''
        
        # Use the first config class found
        config_class = config_classes[0]
        return f'''"""
Configuration for {model_name.title()} model.
"""

{imports}


{config_class["body"]}
'''
    
    def _generate_modeling_file(self, parsed_content: Dict[str, Any], model_name: str, imports: str) -> str:
        """Generate modeling file content."""
        # Combine all classes and functions
        all_content = []
        
        # Add imports
        all_content.append(imports)
        all_content.append("")
        
        # Add all classes
        for cls in parsed_content["classes"]:
            all_content.append(cls["body"])
            all_content.append("")
        
        # Add all functions
        for func in parsed_content["functions"]:
            all_content.append(func["body"])
            all_content.append("")
        
        return "\n".join(all_content)
    
    def _generate_tokenizer_file(self, parsed_content: Dict[str, Any], model_name: str, imports: str) -> str:
        """Generate tokenizer file content."""
        tokenizer_classes = [cls for cls in parsed_content["classes"] if "Tokenizer" in cls["name"]]
        
        if not tokenizer_classes:
            # Generate default tokenizer
            return f'''"""
Tokenizer for {model_name.title()} model.
"""

{imports}


class {model_name.title()}Tokenizer:
    """
    Tokenizer for {model_name.title()}.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
'''
        
        # Use the first tokenizer class found
        tokenizer_class = tokenizer_classes[0]
        return f'''"""
Tokenizer for {model_name.title()} model.
"""

{imports}


{tokenizer_class["body"]}
'''
    
    def _generate_init_file(self, model_name: str) -> str:
        """Generate __init__.py file content."""
        return f'''"""
{model_name.title()} model package.
"""

from .configuration_{model_name} import {model_name.title()}Config

try:
    from .modeling_{model_name} import (
        {model_name.title()}Model,
        {model_name.title()}ForCausalLM,
        {model_name.title()}ForSequenceClassification,
    )
except ImportError:
    pass

try:
    from .tokenization_{model_name} import {model_name.title()}Tokenizer
except ImportError:
    pass

__all__ = [
    "{model_name.title()}Config",
]

# Add model classes if available
try:
    __all__.extend([
        "{model_name.title()}Model",
        "{model_name.title()}ForCausalLM", 
        "{model_name.title()}ForSequenceClassification",
    ])
except NameError:
    pass

# Add tokenizer if available
try:
    __all__.append("{model_name.title()}Tokenizer")
except NameError:
    pass
'''
    
    def _write_converted_files(self, converted_files: Dict[str, str], output_dir: Path, model_name: str) -> bool:
        """Write converted files to the output directory."""
        try:
            # Create model directory
            model_dir = output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Write each file
            for filename, content in converted_files.items():
                file_path = model_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                debug(f"Written file: {file_path}", "converter")
            
            info(f"Successfully wrote {len(converted_files)} files to {model_dir}", "converter")
            return True
            
        except Exception as e:
            error(f"Error writing converted files: {str(e)}", "converter", e)
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Convert modular model files to single-file implementations")
    parser.add_argument("model_name", help="Name of the model to convert (snake_case)")
    parser.add_argument("--input-dir", help="Input directory containing modular files")
    parser.add_argument("--output-dir", help="Output directory for converted files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        # Enable debug logging
        pass
    
    # Create converter and run conversion
    converter = ModularModelConverter()
    success = converter.convert_model(
        model_name=args.model_name,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    if success:
        print(f"✅ Successfully converted modular model: {args.model_name}")
        sys.exit(0)
    else:
        print(f"❌ Failed to convert modular model: {args.model_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
