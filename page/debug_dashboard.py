"""
Debug Dashboard for DurgasAI
Provides comprehensive debugging and monitoring information for troubleshooting.
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Dict, Any, List
import platform

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger, debug, info
from utils.config import Config
from utils.ui_helpers import SessionManager


class DebugDashboard:
    """Debug dashboard for monitoring and troubleshooting DurgasAI."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def render(self):
        """Render the complete debug dashboard."""
        st.markdown('<h1 class="main-header">🔧 Debug Dashboard</h1>', unsafe_allow_html=True)
        
        # Create tabs for different debug sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 System Info", "🗂️ Session State", "📝 Logs", 
            "🤖 Model Status", "⚡ Performance", "🛠️ Tools"
        ])
        
        with tab1:
            self.render_system_info()
        
        with tab2:
            self.render_session_state_info()
        
        with tab3:
            self.render_logs_info()
        
        with tab4:
            self.render_model_status()
        
        with tab5:
            self.render_performance_metrics()
        
        with tab6:
            self.render_tools_status()
    
    def render_system_info(self):
        """Render system information section."""
        st.markdown("### 💻 System Information")
        
        # Python and environment info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Python Environment")
            st.code(f"""
Python Version: {sys.version}
Platform: {platform.platform()}
Architecture: {platform.architecture()}
Processor: {platform.processor()}
Python Executable: {sys.executable}
Working Directory: {os.getcwd()}
            """)
        
        with col2:
            st.markdown("#### System Resources")
            if PSUTIL_AVAILABLE:
                try:
                    # System resource information
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    st.code(f"""
CPU Usage: {cpu_percent}%
Memory Total: {memory.total / (1024**3):.1f} GB
Memory Available: {memory.available / (1024**3):.1f} GB
Memory Usage: {memory.percent}%
Disk Total: {disk.total / (1024**3):.1f} GB
Disk Free: {disk.free / (1024**3):.1f} GB
Disk Usage: {(disk.used/disk.total)*100:.1f}%
                    """)
                except Exception as e:
                    st.error(f"Error getting system info: {e}")
            else:
                st.info("Install psutil for detailed system information: `pip install psutil`")
        
        # Package versions
        st.markdown("#### 📦 Package Versions")
        
        packages_to_check = [
            'streamlit', 'langchain', 'transformers', 'torch', 
            'requests', 'numpy', 'pandas'
        ]
        
        package_info = {}
        for package in packages_to_check:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                package_info[package] = version
            except ImportError:
                package_info[package] = 'Not installed'
        
        st.json(package_info)
    
    def render_session_state_info(self):
        """Render session state debugging information."""
        st.markdown("### 🗂️ Session State Debug")
        
        # Session state overview
        if hasattr(st, 'session_state'):
            st.markdown("#### Current Session State")
            
            # Filter sensitive information
            safe_session_state = {}
            for key, value in st.session_state.items():
                if 'token' in key.lower() or 'key' in key.lower():
                    safe_session_state[key] = f"[HIDDEN - Length: {len(str(value))}]"
                elif isinstance(value, (list, dict)) and len(str(value)) > 1000:
                    safe_session_state[key] = f"[LARGE OBJECT - Type: {type(value).__name__}, Size: {len(str(value))}]"
                else:
                    safe_session_state[key] = value
            
            st.json(safe_session_state)
            
            # Session metrics
            st.markdown("#### Session Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Keys", len(st.session_state.keys()))
            
            with col2:
                messages = st.session_state.get("messages", [])
                st.metric("Messages", len(messages))
            
            with col3:
                model_loaded = st.session_state.get("model_loaded", False)
                st.metric("Model Loaded", "Yes" if model_loaded else "No")
            
            with col4:
                session_start = st.session_state.get("session_start_time")
                if session_start:
                    duration = datetime.now() - session_start
                    st.metric("Session Duration", str(duration).split('.')[0])
            
            # Message history analysis
            if messages:
                st.markdown("#### Message History Analysis")
                
                user_msgs = [m for m in messages if m.get("role") == "user"]
                assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("User Messages", len(user_msgs))
                    if user_msgs:
                        avg_user_length = sum(len(m.get("content", "")) for m in user_msgs) / len(user_msgs)
                        st.metric("Avg User Message Length", f"{avg_user_length:.0f} chars")
                
                with col2:
                    st.metric("Assistant Messages", len(assistant_msgs))
                    if assistant_msgs:
                        avg_assistant_length = sum(len(m.get("content", "")) for m in assistant_msgs) / len(assistant_msgs)
                        st.metric("Avg Assistant Message Length", f"{avg_assistant_length:.0f} chars")
        
        else:
            st.warning("Session state not available")
    
    def render_logs_info(self):
        """Render logging information and recent logs."""
        st.markdown("### 📝 Logging Information")
        
        # Log file status
        log_files = {
            "Application Log": "logs/app.log",
            "Debug Log": "logs/debug/debug.log",
            "Model Log": "logs/debug/models.log",
            "Performance Log": "logs/performance/performance.log",
            "Error Log": "logs/errors.log",
            "Session Log": f"logs/sessions/session_{self.logger.session_id}.log"
        }
        
        st.markdown("#### 📁 Log Files Status")
        
        for log_name, log_path in log_files.items():
            if os.path.exists(log_path):
                file_size = os.path.getsize(log_path)
                mod_time = datetime.fromtimestamp(os.path.getmtime(log_path))
                st.success(f"✅ **{log_name}**: {file_size} bytes, modified {mod_time.strftime('%H:%M:%S')}")
            else:
                st.warning(f"⚠️ **{log_name}**: File not found")
        
        # Recent log entries
        st.markdown("#### 🕐 Recent Log Entries")
        
        log_type = st.selectbox("Select log type:", list(log_files.keys()))
        selected_log_path = log_files[log_type]
        
        if os.path.exists(selected_log_path):
            try:
                with open(selected_log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Show last N lines
                num_lines = st.slider("Number of recent lines:", 10, 100, 50)
                recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
                
                st.code('\n'.join(recent_lines), language='text')
                
            except Exception as e:
                st.error(f"Error reading log file: {e}")
        else:
            st.info(f"Log file {selected_log_path} does not exist yet")
    
    def render_model_status(self):
        """Render model status and configuration."""
        st.markdown("### 🤖 Model Status")
        
        # Current model information
        current_model = st.session_state.get("current_model")
        model_loaded = st.session_state.get("model_loaded", False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Model")
            if model_loaded and current_model:
                st.success(f"✅ **{current_model}** is loaded")
            else:
                st.warning("⚠️ No model currently loaded")
        
        with col2:
            st.markdown("#### Model Configuration")
            if current_model:
                # Find model config
                for key, config in Config.AVAILABLE_MODELS.items():
                    if config.name == current_model:
                        st.json({
                            "model_id": config.model_id,
                            "provider": config.provider.value,
                            "max_tokens": config.max_tokens,
                            "temperature": config.temperature,
                            "top_p": config.top_p,
                            "repetition_penalty": config.repetition_penalty
                        })
                        break
            else:
                st.info("No model configuration to display")
        
        # Available models
        st.markdown("#### 📋 Available Models")
        
        models_data = []
        for key, config in Config.AVAILABLE_MODELS.items():
            models_data.append({
                "Key": key,
                "Name": config.name,
                "Provider": config.provider.value,
                "Model ID": config.model_id,
                "Max Tokens": config.max_tokens,
                "Temperature": config.temperature
            })
        
        st.dataframe(models_data, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render performance monitoring information."""
        st.markdown("### ⚡ Performance Metrics")
        
        # Response time tracking
        last_response_time = st.session_state.get("last_model_response_time")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if last_response_time:
                st.metric("Last Response Time", f"{last_response_time:.2f}s")
            else:
                st.metric("Last Response Time", "N/A")
        
        with col2:
            total_messages = st.session_state.get("total_messages", 0)
            st.metric("Total Messages", total_messages)
        
        with col3:
            session_start = st.session_state.get("session_start_time")
            if session_start:
                duration = datetime.now() - session_start
                st.metric("Session Duration", str(duration).split('.')[0])
            else:
                st.metric("Session Duration", "Unknown")
        
        # Performance log analysis
        st.markdown("#### 📈 Performance Trends")
        
        perf_log_path = "logs/performance/performance.log"
        if os.path.exists(perf_log_path):
            try:
                with open(perf_log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Parse performance data
                performance_data = []
                for line in lines[-50:]:  # Last 50 entries
                    if "PERFORMANCE:" in line:
                        try:
                            json_part = line.split("PERFORMANCE: ")[1]
                            perf_data = json.loads(json_part)
                            performance_data.append(perf_data)
                        except:
                            continue
                
                if performance_data:
                    st.dataframe(performance_data, use_container_width=True)
                else:
                    st.info("No performance data available yet")
            
            except Exception as e:
                st.error(f"Error reading performance log: {e}")
        else:
            st.info("Performance log not found")
    
    def render_tools_status(self):
        """Render tools and configuration status."""
        st.markdown("### 🛠️ Tools & Configuration")
        
        # Configuration file status
        st.markdown("#### ⚙️ Configuration Status")
        
        try:
            # Use ConfigService to get configuration status
            from services import ConfigService
            config_service = ConfigService()
            
            # Get API configuration
            api_config = config_service.get_api_config()
            app_config = config_service.get_app_config()
            comprehensive_config = config_service.get_comprehensive_config()
            
            total_sections = len(api_config) + len(app_config) + len(comprehensive_config)
            st.success(f"✅ Configuration loaded: {total_sections} sections from distributed config files")
            
            # Show API key status (without revealing keys)
            api_status = {}
            for provider, config in api_config.items():
                if isinstance(config, dict) and "api_keys" in config:
                    key = config["api_keys"]
                    if key and key.strip():
                        api_status[provider] = f"✅ Configured ({len(key)} chars)"
                    else:
                        api_status[provider] = "❌ Not configured"
            
            if api_status:
                st.markdown("**API Keys Status:**")
                for provider, status in api_status.items():
                    st.write(f"- **{provider}**: {status}")
            
            # Show config file status
            st.markdown("**Config Files Status:**")
            config_files = {
                'API Config': 'config/api_config.json',
                'App Config': 'config/app_config.json', 
                'Model Config': 'config/model_config.json',
                'Comprehensive Config': 'config/comprehensive_config.json'
            }
            
            for name, path in config_files.items():
                if os.path.exists(path):
                    st.write(f"- **{name}**: ✅ Found")
                else:
                    st.write(f"- **{name}**: ❌ Missing")
                
        except Exception as e:
            st.error(f"Error reading configuration: {e}")
            st.warning("Configuration service not available")
        
        # Tools status
        st.markdown("#### 🔧 Available Tools")
        
        tools_dir = Path("output/tools")
        if tools_dir.exists():
            tool_files = list(tools_dir.glob("*.json"))
            
            st.write(f"Found {len(tool_files)} tool configuration files:")
            
            tools_data = []
            for tool_file in tool_files:
                try:
                    with open(tool_file, 'r') as f:
                        tool_config = json.load(f)
                    
                    # Check if corresponding Python file exists
                    code_file = tools_dir / "code" / tool_config.get("code", "")
                    code_exists = code_file.exists() if tool_config.get("code") else False
                    
                    tools_data.append({
                        "Name": tool_config.get("name", "Unknown"),
                        "Status": tool_config.get("status", "Unknown"),
                        "Category": tool_config.get("category", "Unknown"),
                        "Code File": "✅" if code_exists else "❌",
                        "Security Level": tool_config.get("security_level", "Unknown")
                    })
                
                except Exception as e:
                    tools_data.append({
                        "Name": tool_file.stem,
                        "Status": "Error",
                        "Category": "Unknown",
                        "Code File": "❌",
                        "Security Level": f"Error: {e}"
                    })
            
            if tools_data:
                st.dataframe(tools_data, use_container_width=True)
        else:
            st.warning("Tools directory not found")
        
        # Workflow status
        st.markdown("#### 🔄 Workflows")
        
        workflows_dir = Path("output/workflows")
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.json"))
            st.write(f"Found {len(workflow_files)} workflow files:")
            
            for workflow_file in workflow_files:
                try:
                    with open(workflow_file, 'r') as f:
                        workflow_config = json.load(f)
                    
                    with st.expander(f"📋 {workflow_config.get('name', workflow_file.stem)}"):
                        st.write(f"**Description:** {workflow_config.get('description', 'No description')}")
                        st.write(f"**Steps:** {len(workflow_config.get('steps', []))}")
                        st.json(workflow_config)
                
                except Exception as e:
                    st.error(f"Error reading workflow {workflow_file.name}: {e}")
        else:
            st.warning("Workflows directory not found")
    
    def render_debug_controls(self):
        """Render debug control panel."""
        st.markdown("### 🎛️ Debug Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Debug Info"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session state cleared!")
                st.rerun()
        
        with col3:
            if st.button("📊 Export Debug Report"):
                self.export_debug_report()
    
    def export_debug_report(self):
        """Export comprehensive debug report."""
        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.logger.session_id,
            "system_info": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "working_directory": os.getcwd()
            },
            "session_state": {
                key: str(value)[:200] if len(str(value)) > 200 else value
                for key, value in st.session_state.items()
                if 'token' not in key.lower()
            },
            "configuration": {
                "available_models": len(Config.AVAILABLE_MODELS),
                "max_message_history": Config.MAX_MESSAGE_HISTORY,
                "api_timeout": Config.API_TIMEOUT
            }
        }
        
        debug_json = json.dumps(debug_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Debug Report",
            data=debug_json,
            file_name=f"durgasai_debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def render_debug_dashboard():
    """Render the debug dashboard."""
    dashboard = DebugDashboard()
    dashboard.render()
