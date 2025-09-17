"""
Monitor Page Module
Handles the system monitoring functionality
"""

import streamlit as st
from ..components.system_monitor import SystemMonitor


class MonitorPage:
    """Monitor page implementation"""
    
    def __init__(self):
        pass
    
    def render(self):
        """Render the system monitoring page"""
        st.title("📊 System Monitor")
        st.markdown("Monitor system performance and usage")
        
        system_monitor = SystemMonitor()
        system_monitor.render()
