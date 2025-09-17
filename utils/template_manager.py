"""
Enhanced Template Manager for DurgasAI
Manages AI assistant templates and personalities with LangGraph workflow support
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from utils.logging_utils import log_info, log_error, log_debug, log_warning

class TemplateManager:
    """Enhanced DurgasAI assistant template manager with LangGraph workflow support"""
    
    def __init__(self, templates_directory: str = "output/templates"):
        self.templates_directory = Path(templates_directory)
        self.templates = {}
        self.template_metadata = {}
        self._load_templates()
        log_info(f"Template manager initialized with {len(self.templates)} templates", "templates")
    

    def _load_templates(self):
        """Load all templates from the templates directory with enhanced error handling"""
        try:
            for template_file in self.templates_directory.glob("*.json"):
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template = json.load(f)
                    
                    template_name = template.get("name")
                    if template_name:
                        # Validate template
                        validation_errors = self.validate_template(template)
                        if validation_errors:
                            log_warning(f"Template {template_name} has validation issues: {validation_errors}", "templates")
                        
                        self.templates[template_name] = template
                        
                        # Initialize metadata
                        self.template_metadata[template_name] = {
                            "load_time": time.time(),
                            "usage_count": 0,
                            "last_used": None,
                            "workflow_type": template.get("workflow_type", "basic_chat"),
                            "tool_count": len(template.get("function_calling_tools", [])),
                            "is_enhanced": "workflow_type" in template
                        }
                        
                        log_debug(f"Loaded template: {template_name}", "templates")
                        
                except Exception as e:
                    log_error(f"Error loading template {template_file}: {e}", "templates", e)
        
        except Exception as e:
            log_error(f"Error loading templates directory: {e}", "templates", e)
    
    def get_all_templates(self) -> List[Dict[str, Any]]:
        """Get all available templates with enhanced metadata"""
        templates = []
        for template_name, template in self.templates.items():
            enhanced_template = template.copy()
            enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
            templates.append(enhanced_template)
        return templates
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific template by name with enhanced metadata"""
        template = self.templates.get(template_name)
        if template:
            enhanced_template = template.copy()
            enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
            return enhanced_template
        return None

    def get_workflow_templates(self, template_name: str = None) -> Dict[str, Dict[str, Any]]:
        """Get workflow templates from a specific template or all templates"""
        if template_name:
            template = self.get_template(template_name)
            if template and "workflow_templates" in template:
                return template["workflow_templates"]
            return {}
        
        # If no template specified, get workflow templates from all templates
        all_workflow_types = {}
        for template in self.templates.values():
            if "workflow_templates" in template:
                all_workflow_types.update(template["workflow_templates"])
        return all_workflow_types

    def get_templates_by_workflow_type(self, workflow_type: str) -> List[Dict[str, Any]]:
        """Get templates that use a specific workflow type"""
        templates = []
        for template_name, template in self.templates.items():
            if template.get("workflow_type") == workflow_type:
                enhanced_template = template.copy()
                enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
                templates.append(enhanced_template)
        return templates

    def create_enhanced_template(self, template_data: Dict[str, Any], workflow_type: str = "basic_chat") -> bool:
        """Create an enhanced template with LangGraph workflow support"""
        try:
            # Add workflow configuration
            template_data["workflow_type"] = workflow_type
            template_data["workflow_config"] = self._get_workflow_config(workflow_type)
            
            # Add enhanced features - get from template's own workflow_templates if available
            if "enhanced_features" not in template_data:
                if "workflow_templates" in template_data and workflow_type in template_data["workflow_templates"]:
                    template_data["enhanced_features"] = template_data["workflow_templates"][workflow_type].get("features", [])
                else:
                    # Default features for workflow types
                    default_features = {
                        "basic_chat": ["conversation", "basic_tools"],
                        "advanced_tools": ["conversation", "tool_planning", "multi_step_reasoning"],
                        "research_assistant": ["web_search", "information_synthesis", "source_citation"],
                        "code_assistant": ["code_generation", "debugging", "file_operations", "calculations"],
                        "conditional_workflow": ["conversation", "content_analysis", "conditional_routing", "specialized_processing"]
                    }
                    template_data["enhanced_features"] = default_features.get(workflow_type, [])
            
            # Add creation timestamp
            template_data["created_at"] = self._get_current_timestamp()
            template_data["updated_at"] = template_data["created_at"]
            template_data["version"] = template_data.get("version", "1.0")
            
            # Create the template
            success = self.create_template(template_data)
            if success:
                log_info(f"Created enhanced template {template_data.get('name')} with workflow {workflow_type}", "templates")
            return success
            
        except Exception as e:
            log_error(f"Error creating enhanced template: {e}", "templates", e)
            return False

    def _get_workflow_config(self, workflow_type: str) -> Dict[str, Any]:
        """Get workflow configuration for a specific workflow type"""
        workflow_templates = {
            "basic_chat": {
                "max_iterations": 5,
                "enable_tool_retry": False,
                "enable_error_recovery": True,
                "conversation_memory": True
            },
            "advanced_tools": {
                "max_iterations": 10,
                "enable_tool_retry": True,
                "enable_error_recovery": True,
                "enable_planning": True,
                "conversation_memory": True,
                "tool_validation": True
            },
            "research_assistant": {
                "max_iterations": 15,
                "enable_tool_retry": True,
                "enable_error_recovery": True,
                "enable_web_search": True,
                "enable_source_tracking": True,
                "conversation_memory": True
            },
            "code_assistant": {
                "max_iterations": 12,
                "enable_tool_retry": True,
                "enable_error_recovery": True,
                "enable_code_validation": True,
                "enable_file_operations": True,
                "conversation_memory": True
            }
        }
        return workflow_templates.get(workflow_type, workflow_templates["basic_chat"])

    def update_template_workflow(self, template_name: str, workflow_type: str) -> bool:
        """Update a template's workflow type"""
        try:
            template = self.get_template(template_name)
            if not template:
                return False
            
            # Update workflow configuration
            template["workflow_type"] = workflow_type
            template["workflow_config"] = self._get_workflow_config(workflow_type)
            
            # Update enhanced features from template's own workflow_templates if available
            if "workflow_templates" in template and workflow_type in template["workflow_templates"]:
                template["enhanced_features"] = template["workflow_templates"][workflow_type].get("features", [])
            else:
                # Default features for workflow types
                default_features = {
                    "basic_chat": ["conversation", "basic_tools"],
                    "advanced_tools": ["conversation", "tool_planning", "multi_step_reasoning"],
                    "research_assistant": ["web_search", "information_synthesis", "source_citation"],
                    "code_assistant": ["code_generation", "debugging", "file_operations", "calculations"],
                    "conditional_workflow": ["conversation", "content_analysis", "conditional_routing", "specialized_processing"]
                }
                template["enhanced_features"] = default_features.get(workflow_type, [])
            
            template["updated_at"] = self._get_current_timestamp()
            
            # Update metadata
            if template_name in self.template_metadata:
                self.template_metadata[template_name]["workflow_type"] = workflow_type
                self.template_metadata[template_name]["is_enhanced"] = True
            
            # Save template
            return self.update_template(template_name, template)
            
        except Exception as e:
            log_error(f"Error updating template workflow: {e}", "templates", e)
            return False
    
    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get templates by category"""
        return [
            template for template in self.templates.values()
            if template.get("category") == category
        ]
    
    def get_template_categories(self) -> List[str]:
        """Get all available template categories"""
        categories = set()
        for template in self.templates.values():
            category = template.get("category")
            if category:
                categories.add(category)
        return list(categories)
    
    def create_template(self, template_data: Dict[str, Any]) -> bool:
        """Create a new template"""
        try:
            template_name = template_data.get("name")
            if not template_name:
                return False
            
            # Add default values
            if "version" not in template_data:
                template_data["version"] = "1.0"
            
            if "created_at" not in template_data:
                template_data["created_at"] = datetime.now().isoformat()
            
            template_data["updated_at"] = template_data["created_at"]
            
            # Save to file
            template_file = self.templates_directory / f"{template_name.lower().replace(' ', '_')}.json"
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self.templates[template_name] = template_data
            
            return True
            
        except Exception as e:
            print(f"Error creating template: {e}")
            return False
    
    def update_template(self, template_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing template"""
        try:
            template = self.get_template(template_name)
            if not template:
                return False
            
            # Update the template data
            template.update(updates)
            template["updated_at"] = self._get_current_timestamp()
            
            # Save to file
            template_file = self.templates_directory / f"{template_name.lower().replace(' ', '_')}.json"
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self.templates[template_name] = template
            
            return True
            
        except Exception as e:
            print(f"Error updating template: {e}")
            return False
    
    def delete_template(self, template_name: str) -> bool:
        """Delete a template"""
        try:
            template = self.get_template(template_name)
            if not template:
                return False
            
            # Remove file
            template_file = self.templates_directory / f"{template_name.lower().replace(' ', '_')}.json"
            if template_file.exists():
                template_file.unlink()
            
            # Remove from cache
            if template_name in self.templates:
                del self.templates[template_name]
            
            return True
            
        except Exception as e:
            print(f"Error deleting template: {e}")
            return False
    
    def validate_template(self, template_data: Dict[str, Any]) -> List[str]:
        """Validate template data and return list of errors"""
        errors = []
        
        # Required fields
        required_fields = ["name", "description", "category", "system_instruction"]
        for field in required_fields:
            if not template_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate function_calling_tools if enabled
        if template_data.get("function_calling_enabled"):
            tools = template_data.get("function_calling_tools", [])
            if not isinstance(tools, list):
                errors.append("function_calling_tools must be a list")
        
        # Validate generation_parameters
        gen_params = template_data.get("generation_parameters", {})
        if gen_params:
            temperature = gen_params.get("temperature")
            if temperature is not None and (temperature < 0 or temperature > 1):
                errors.append("temperature must be between 0 and 1")
            
            max_tokens = gen_params.get("max_output_tokens")
            if max_tokens is not None and max_tokens <= 0:
                errors.append("max_output_tokens must be positive")
        
        return errors
    
    def export_template(self, template_name: str, export_path: str) -> bool:
        """Export a template to a file"""
        try:
            template = self.get_template(template_name)
            if not template:
                return False
            
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error exporting template: {e}")
            return False
    
    def import_template(self, import_path: str) -> bool:
        """Import a template from a file"""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Validate template
            errors = self.validate_template(template_data)
            if errors:
                print(f"Template validation errors: {errors}")
                return False
            
            # Create template
            return self.create_template(template_data)
            
        except Exception as e:
            print(f"Error importing template: {e}")
            return False
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics about templates"""
        templates = list(self.templates.values())
        
        if not templates:
            return {"total": 0}
        
        # Basic statistics
        categories = {}
        workflow_types = {}
        enhanced_templates = 0
        
        for template in templates:
            category = template.get("category", "uncategorized")
            categories[category] = categories.get(category, 0) + 1
            
            workflow_type = template.get("workflow_type", "basic_chat")
            workflow_types[workflow_type] = workflow_types.get(workflow_type, 0) + 1
            
            if template.get("workflow_type") and template.get("workflow_type") != "basic_chat":
                enhanced_templates += 1
        
        function_calling_enabled = sum(
            1 for template in templates 
            if template.get("function_calling_enabled", False)
        )
        
        # Usage statistics
        total_usage = sum(meta.get("usage_count", 0) for meta in self.template_metadata.values())
        most_used = sorted(
            [(name, meta.get("usage_count", 0)) for name, meta in self.template_metadata.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total": len(templates),
            "by_category": categories,
            "by_workflow_type": workflow_types,
            "enhanced_templates": enhanced_templates,
            "basic_templates": len(templates) - enhanced_templates,
            "function_calling_enabled": function_calling_enabled,
            "function_calling_disabled": len(templates) - function_calling_enabled,
            "average_tools_per_template": sum(
                len(template.get("function_calling_tools", [])) 
                for template in templates
            ) / len(templates) if templates else 0,
            "total_usage": total_usage,
            "most_used_templates": most_used,
            "workflow_templates_available": len(self.get_workflow_templates())
        }

    def record_template_usage(self, template_name: str = None):
        """Record usage of a template"""
        if template_name is None:
            return
        
        if template_name in self.template_metadata:
            self.template_metadata[template_name]["usage_count"] += 1
            self.template_metadata[template_name]["last_used"] = time.time()
            log_debug(f"Recorded usage for template {template_name}", "templates")

    def get_template_performance_report(self, template_name: str) -> Dict[str, Any]:
        """Get detailed performance report for a specific template"""
        template = self.get_template(template_name)
        metadata = self.template_metadata.get(template_name, {})
        
        if not template:
            return {"error": f"Template {template_name} not found"}
        
        return {
            "template_name": template_name,
            "description": template.get("description", ""),
            "category": template.get("category", ""),
            "workflow_type": template.get("workflow_type", "basic_chat"),
            "is_enhanced": metadata.get("is_enhanced", False),
            "usage_count": metadata.get("usage_count", 0),
            "last_used": metadata.get("last_used"),
            "tool_count": metadata.get("tool_count", 0),
            "function_calling_enabled": template.get("function_calling_enabled", False),
            "enhanced_features": template.get("enhanced_features", []),
            "workflow_config": template.get("workflow_config", {}),
            "created_at": template.get("created_at"),
            "updated_at": template.get("updated_at"),
            "version": template.get("version", "1.0")
        }

    def get_workflow_recommendations(self, use_case: str) -> List[Dict[str, Any]]:
        """Get workflow recommendations based on use case"""
        recommendations = {
            "general_chat": ["basic_chat"],
            "research": ["research_assistant", "advanced_tools"],
            "programming": ["code_assistant", "advanced_tools"],
            "data_analysis": ["code_assistant", "advanced_tools"],
            "content_creation": ["basic_chat", "advanced_tools"],
            "customer_support": ["basic_chat", "advanced_tools"],
            "education": ["research_assistant", "code_assistant"],
            "business_analysis": ["research_assistant", "advanced_tools"]
        }
        
        recommended_workflows = recommendations.get(use_case.lower(), ["basic_chat"])
        results = []
        
        # Get all available workflow templates
        all_workflow_templates = self.get_workflow_templates()
        
        for workflow_type in recommended_workflows:
            if workflow_type in all_workflow_templates:
                workflow_info = all_workflow_templates[workflow_type].copy()
                workflow_info["workflow_type"] = workflow_type
                workflow_info["config"] = self._get_workflow_config(workflow_type)
                results.append(workflow_info)
        
        return results
    
    def reload_templates(self):
        """Reload all templates from disk with enhanced logging"""
        log_info("Reloading templates from disk", "templates")
        self.templates.clear()
        self.template_metadata.clear()
        self._load_templates()
        log_info(f"Reloaded {len(self.templates)} templates", "templates")
    
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced template search with workflow type and feature filtering"""
        query = query.lower()
        results = []
        
        for template_name, template in self.templates.items():
            # Search in name
            if query in template.get("name", "").lower():
                enhanced_template = template.copy()
                enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
                results.append(enhanced_template)
                continue
            
            # Search in description
            if query in template.get("description", "").lower():
                enhanced_template = template.copy()
                enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
                results.append(enhanced_template)
                continue
            
            # Search in tags
            tags = template.get("tags", [])
            if any(query in tag.lower() for tag in tags):
                enhanced_template = template.copy()
                enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
                results.append(enhanced_template)
                continue
            
            # Search in workflow type
            workflow_type = template.get("workflow_type", "")
            if query in workflow_type.lower():
                enhanced_template = template.copy()
                enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
                results.append(enhanced_template)
                continue
            
            # Search in enhanced features
            features = template.get("enhanced_features", [])
            if any(query in feature.lower() for feature in features):
                enhanced_template = template.copy()
                enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
                results.append(enhanced_template)
        
        return results

    def get_templates_by_features(self, features: List[str]) -> List[Dict[str, Any]]:
        """Get templates that have specific features"""
        results = []
        
        for template_name, template in self.templates.items():
            template_features = template.get("enhanced_features", [])
            if any(feature in template_features for feature in features):
                enhanced_template = template.copy()
                enhanced_template["metadata"] = self.template_metadata.get(template_name, {})
                results.append(enhanced_template)
        
        return results

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
