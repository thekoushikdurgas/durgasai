"""
System Monitor UI Component for DurgasAI
Provides system performance monitoring and usage statistics
"""

import streamlit as st
import psutil
import time
import sys
import platform
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any
from utils.langgraph_integration import LangGraphManager
from utils.config_manager import ConfigManager
from utils.tool_manager import DurgasAIToolManager

class SystemMonitor:
    """UI component for system monitoring"""
    
    def __init__(self):
        self.monitoring_data = []
    
    def render(self):
        """Render the system monitoring interface"""
        # Real-time metrics
        self.render_realtime_metrics()
        
        # LangGraph specific monitoring
        self.render_langgraph_metrics()
        
        # Performance charts
        self.render_performance_charts()
        
        # Usage statistics
        self.render_usage_statistics()
        
        # System information
        self.render_system_info()
    
    def render_realtime_metrics(self):
        """Render real-time system metrics"""
        st.subheader("📊 Real-time Metrics")
        
        # Get current system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{cpu_percent}%",
                delta=None,
                help="Current CPU utilization"
            )
        
        with col2:
            st.metric(
                "Memory Usage",
                f"{memory.percent}%",
                delta=None,
                help=f"Used: {memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB"
            )
        
        with col3:
            st.metric(
                "Disk Usage",
                f"{disk.percent}%",
                delta=None,
                help=f"Used: {disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB"
            )
        
        with col4:
            # Get network stats
            network = psutil.net_io_counters()
            st.metric(
                "Network I/O",
                f"↑{network.bytes_sent // (1024**2)}MB",
                delta=None,
                help=f"↓{network.bytes_recv // (1024**2)}MB"
            )
        
        # Store monitoring data
        self.monitoring_data.append({
            "timestamp": datetime.now(),
            "cpu": cpu_percent,
            "memory": memory.percent,
            "disk": disk.percent
        })
        
        # Keep only last 100 data points
        if len(self.monitoring_data) > 100:
            self.monitoring_data = self.monitoring_data[-100:]
    
    def render_langgraph_metrics(self):
        """Render LangGraph specific monitoring metrics"""
        st.subheader("🔄 LangGraph Monitoring")
        
        # Check if we have LangGraph data in session state
        if "workflow_metrics" not in st.session_state:
            st.info("No LangGraph session data available")
            return
        
        metrics = st.session_state.workflow_metrics
        
        # LangGraph session metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Sessions",
                len(st.session_state.get("chat_session", [])),
                help="Number of active LangGraph sessions"
            )
        
        with col2:
            st.metric(
                "Messages Processed",
                metrics.get("message_count", 0),
                help="Total messages processed in current session"
            )
        
        with col3:
            st.metric(
                "Tool Calls",
                metrics.get("tool_calls", 0),
                help="Total tool calls made in current session"
            )
        
        with col4:
            workflow_steps = metrics.get("workflow_steps", [])
            avg_step_time = sum(step.get("duration", 0) for step in workflow_steps) / len(workflow_steps) if workflow_steps else 0
            st.metric(
                "Avg Step Time",
                f"{avg_step_time:.2f}s",
                help="Average time per workflow step"
            )
        
        # Workflow performance chart
        self.render_workflow_performance_chart()
        
        # Session details
        self.render_session_details()
    
    def render_workflow_performance_chart(self):
        """Render workflow performance chart"""
        st.subheader("📈 Workflow Performance")
        
        metrics = st.session_state.workflow_metrics
        performance_data = metrics.get("performance_data", [])
        
        if len(performance_data) < 2:
            st.info("Collecting workflow performance data...")
            return
        
        # Prepare data for chart
        timestamps = [data["timestamp"] for data in performance_data]
        response_times = [data["response_time"] for data in performance_data]
        token_counts = [data.get("tokens", 0) for data in performance_data]
        
        # Create performance chart
        fig = go.Figure()
        
        # Add response time trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=response_times,
            mode='lines+markers',
            name='Response Time (s)',
            yaxis='y',
            line=dict(color='blue')
        ))
        
        # Add token count trace on secondary y-axis
        if any(token_counts):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=token_counts,
                mode='lines+markers',
                name='Token Count',
                yaxis='y2',
                line=dict(color='green')
            ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            title="Workflow Performance Over Time",
            xaxis_title="Time",
            yaxis=dict(
                title="Response Time (seconds)",
                side="left",
                color="blue"
            ),
            yaxis2=dict(
                title="Token Count",
                side="right",
                overlaying="y",
                color="green"
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_session_details(self):
        """Render detailed session information"""
        st.subheader("📋 Session Details")
        
        if "chat_session" not in st.session_state or not st.session_state.chat_session:
            st.info("No active session")
            return
        
        # Get session info
        try:
            
            # Initialize managers (this is a bit hacky but necessary for the UI)
            config_manager = ConfigManager()
            tool_manager = DurgasAIToolManager()
            langgraph_manager = LangGraphManager(config_manager, tool_manager)
            
            session_info = langgraph_manager.get_session_info(st.session_state.chat_session)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session Information:**")
                st.write(f"• **Session ID:** {st.session_state.chat_session}")
                if session_info:
                    st.write(f"• **Workflow Type:** {session_info.get('workflow_type', 'Unknown')}")
                    st.write(f"• **Created:** {session_info.get('created_at', 'Unknown')}")
                    st.write(f"• **Last Activity:** {session_info.get('last_activity', 'Unknown')}")
                    st.write(f"• **Message Count:** {session_info.get('message_count', 0)}")
                else:
                    st.write("• **Status:** Session info not available")
            
            with col2:
                st.write("**Performance Metrics:**")
                if session_info:
                    st.write(f"• **Total Tool Calls:** {session_info.get('total_tool_calls', 0)}")
                    st.write(f"• **Average Response Time:** {session_info.get('avg_response_time', 0):.2f}s")
                    st.write(f"• **Error Count:** {session_info.get('error_count', 0)}")
                    st.write(f"• **Success Rate:** {session_info.get('success_rate', 0):.1f}%")
                else:
                    st.write("• **Status:** Performance metrics not available")
            
            # Workflow steps timeline
            workflow_steps = st.session_state.workflow_metrics.get("workflow_steps", [])
            if workflow_steps:
                st.write("**Recent Workflow Steps:**")
                for i, step in enumerate(workflow_steps[-5:]):  # Show last 5 steps
                    status_icon = "✅" if step.get("status") == "success" else "❌"
                    st.write(f"{status_icon} **{step.get('action', 'Unknown')}** - {step.get('duration', 0):.2f}s")
        
        except Exception as e:
            st.error(f"Error retrieving session details: {str(e)}")
    
    def render_performance_charts(self):
        """Render performance charts"""
        st.subheader("📈 Performance Trends")
        
        if len(self.monitoring_data) < 2:
            st.info("Collecting performance data... Please wait a moment.")
            return
        
        # Prepare data for charts
        timestamps = [data["timestamp"] for data in self.monitoring_data]
        cpu_data = [data["cpu"] for data in self.monitoring_data]
        memory_data = [data["memory"] for data in self.monitoring_data]
        disk_data = [data["disk"] for data in self.monitoring_data]
        
        # Create charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**CPU Usage Over Time**")
            
            fig_cpu = go.Figure()
            fig_cpu.add_trace(go.Scatter(
                x=timestamps,
                y=cpu_data,
                mode='lines',
                name='CPU %',
                line=dict(color='blue')
            ))
            fig_cpu.update_layout(
                title="CPU Usage",
                xaxis_title="Time",
                yaxis_title="CPU %",
                height=300
            )
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            st.write("**Memory Usage Over Time**")
            fig_memory = go.Figure()
            fig_memory.add_trace(go.Scatter(
                x=timestamps,
                y=memory_data,
                mode='lines',
                name='Memory %',
                line=dict(color='green')
            ))
            fig_memory.update_layout(
                title="Memory Usage",
                xaxis_title="Time",
                yaxis_title="Memory %",
                height=300
            )
            st.plotly_chart(fig_memory, use_container_width=True)
        
        # Combined chart
        st.write("**System Resources Overview**")
        fig_combined = go.Figure()
        
        fig_combined.add_trace(go.Scatter(
            x=timestamps,
            y=cpu_data,
            mode='lines',
            name='CPU %',
            line=dict(color='blue')
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=timestamps,
            y=memory_data,
            mode='lines',
            name='Memory %',
            line=dict(color='green')
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=timestamps,
            y=disk_data,
            mode='lines',
            name='Disk %',
            line=dict(color='red')
        ))
        
        fig_combined.update_layout(
            title="System Resources Overview",
            xaxis_title="Time",
            yaxis_title="Usage %",
            height=400
        )
        st.plotly_chart(fig_combined, use_container_width=True)
    
    def render_usage_statistics(self):
        """Render usage statistics"""
        st.subheader("📋 Usage Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Process Information**")
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
            
            # Get top 5 processes by CPU usage
            top_cpu_processes = sorted(
                processes,
                key=lambda x: x.info.get('cpu_percent', 0),
                reverse=True
            )[:5]
            
            st.write("Top 5 Processes by CPU:")
            for proc in top_cpu_processes:
                info = proc.info
                st.write(f"• {info.get('name', 'Unknown')}: {info.get('cpu_percent', 0):.1f}%")
            
            # Get top 5 processes by memory usage
            top_memory_processes = sorted(
                processes,
                key=lambda x: x.info.get('memory_percent', 0),
                reverse=True
            )[:5]
            
            st.write("Top 5 Processes by Memory:")
            for proc in top_memory_processes:
                info = proc.info
                st.write(f"• {info.get('name', 'Unknown')}: {info.get('memory_percent', 0):.1f}%")
        
        with col2:
            st.write("**Network Statistics**")
            network = psutil.net_io_counters()
            
            st.write(f"**Bytes Sent:** {network.bytes_sent:,}")
            st.write(f"**Bytes Received:** {network.bytes_recv:,}")
            st.write(f"**Packets Sent:** {network.packets_sent:,}")
            st.write(f"**Packets Received:** {network.packets_recv:,}")
            st.write(f"**Errors In:** {network.errin:,}")
            st.write(f"**Errors Out:** {network.errout:,}")
            st.write(f"**Dropped In:** {network.dropin:,}")
            st.write(f"**Dropped Out:** {network.dropout:,}")
    
    def render_system_info(self):
        """Render system information"""
        st.subheader("💻 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Hardware Information**")
            
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_count_logical = psutil.cpu_count(logical=True)
            st.write(f"**CPU Cores:** {cpu_count} physical, {cpu_count_logical} logical")
            
            # Memory information
            memory = psutil.virtual_memory()
            st.write(f"**Total Memory:** {memory.total // (1024**3)}GB")
            st.write(f"**Available Memory:** {memory.available // (1024**3)}GB")
            
            # Disk information
            disk = psutil.disk_usage('/')
            st.write(f"**Total Disk Space:** {disk.total // (1024**3)}GB")
            st.write(f"**Free Disk Space:** {disk.free // (1024**3)}GB")
        
        with col2:
            st.write("**Software Information**")
            
            # Boot time
            boot_time = psutil.boot_time()
            boot_datetime = datetime.fromtimestamp(boot_time)
            uptime = datetime.now() - boot_datetime
            st.write(f"**System Uptime:** {uptime}")
            
            # Python version
            st.write(f"**Python Version:** {sys.version}")
            
            # Streamlit version
            try:
                import streamlit as st_lib
                st.write(f"**Streamlit Version:** {st_lib.__version__}")
            except:
                st.write("**Streamlit Version:** Unknown")
            
            # Platform information
            st.write(f"**Platform:** {platform.platform()}")
            st.write(f"**Architecture:** {platform.architecture()[0]}")
        
        # Performance alerts
        st.subheader("⚠️ Performance Alerts")
        
        current_data = self.monitoring_data[-1] if self.monitoring_data else None
        
        if current_data:
            alerts = []
            
            if current_data["cpu"] > 80:
                alerts.append(f"🚨 High CPU usage: {current_data['cpu']:.1f}%")
            
            if current_data["memory"] > 85:
                alerts.append(f"🚨 High memory usage: {current_data['memory']:.1f}%")
            
            if current_data["disk"] > 90:
                alerts.append(f"🚨 High disk usage: {current_data['disk']:.1f}%")
            
            if alerts:
                for alert in alerts:
                    st.error(alert)
            else:
                st.success("✅ All system metrics are within normal ranges")
        else:
            st.info("Collecting performance data...")
        
        # Refresh button
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
