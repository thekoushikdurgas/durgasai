"""
DurgasAI Chatbot Interface Component
Integrates LangGraph chatbot with DurgasAI tools and templates
Enhanced with workflow visualization and monitoring capabilities
"""

import streamlit as st
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from utils.config_manager import ConfigManager
from utils.langgraph_integration import LangGraphManager
from utils.template_manager import TemplateManager
from utils.tool_manager import DurgasAIToolManager

class ChatbotInterface:
    """Main chatbot interface component with enhanced workflow visualization"""
    
    def __init__(self, config_manager: ConfigManager, tool_manager: DurgasAIToolManager, template_manager: TemplateManager):
        self.config_manager = config_manager
        self.tool_manager = tool_manager
        self.template_manager = template_manager
        self.langgraph_manager = LangGraphManager(config_manager, tool_manager)
        
        # # Initialize chat session if not exists
        # if "chat_session" not in st.session_state:
        #     st.session_state.chat_session = None
        
        # Initialize workflow monitoring data
        if "workflow_metrics" not in st.session_state:
            st.session_state.workflow_metrics = {
                "session_start": datetime.now(),
                "message_count": 0,
                "tool_calls": 0,
                "workflow_steps": [],
                "performance_data": []
            }
        
        # Initialize conversation analytics
        if "conversation_analytics" not in st.session_state:
            st.session_state.conversation_analytics = {
                "total_tokens": 0,
                "avg_response_time": 0,
                "workflow_efficiency": 0,
                "tool_usage_stats": {}
            }
    
    # def render(self):
    #     """Render the main chat interface with workflow visualization"""
        
    #     # Workflow status and controls
    #     self.render_workflow_status()


    
    def render_chat_messages(self):
        """Render chat message history"""
        # st.subheader("💬 Chat History")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Hello! I'm your DurgasAI assistant. How can I help you today?")
                return
            
            for message in st.session_state.chat_history:
                if message["type"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                elif message["type"] == "assistant":
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Show tool calls if any
                        if "tool_calls" in message and message["tool_calls"]:
                            with st.expander("🔧 Tool Calls"):
                                for tool_call in message["tool_calls"]:
                                    st.json(tool_call)
                elif message["type"] == "tool":
                    with st.chat_message("assistant"):
                        st.write("🔧 **Tool Execution:**")
                        st.write(message["content"])
    
    def render_chat_input(self):
        """Render chat input interface"""
        # st.subheader("✍️ Send Message")
        
        # Chat input
        user_input = st.chat_input(
            "Type your message here...",
            key="chat_input"
        )
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "type": "user",
                "content": user_input
            })
            
            # Process message with LangGraph
            self.process_user_message(user_input)
    
    def process_user_message(self, user_input: str = None):
        """Process user message through LangGraph chatbot with enhanced monitoring"""
        if user_input is None or not user_input.strip():
            st.warning("No message provided")
            return
        
        start_time = time.time()
        
        try:
            # Show processing indicator
            with st.spinner("🤖 Thinking..."):
                # Get selected template
                template = self.template_manager.get_template(st.session_state.selected_template)
                
                # Initialize or get existing chat session
                if not st.session_state.chat_session:
                    # Use selected workflow type from sidebar, fallback to template default
                    workflow_type = st.session_state.get("selected_workflow_type", template.get("workflow_type", "basic_chat"))
                    st.session_state.chat_session = self.langgraph_manager.create_chat_session(
                        template=template,
                        llm_provider=st.session_state.get("llm_provider", "OpenAI"),
                        model=st.session_state.get("selected_model", "gpt-4o"),
                        api_keys=st.session_state.get("api_keys", {}),
                        workflow_type=workflow_type
                    )
                
                # Process message through LangGraph
                response = self.langgraph_manager.process_message(
                    session=st.session_state.chat_session,
                    message=user_input,
                    template=template
                )
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Update workflow metrics
                step_data = {
                    "action": "message_processed",
                    "duration": response_time,
                    "status": "success",
                    "tool_calls": response.get("tool_calls", []),
                    "response_time": response_time,
                    "tokens": response.get("tokens", 0)
                }
                self.update_workflow_metrics(step_data)
                
                # Add assistant response to history
                if response:
                    assistant_message = {
                        "type": "assistant",
                        "content": response.get("content", "I'm sorry, I couldn't generate a response."),
                        "tool_calls": response.get("tool_calls", [])
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Add tool responses if any
                    if "tool_responses" in response:
                        for tool_response in response["tool_responses"]:
                            st.session_state.chat_history.append({
                                "type": "tool",
                                "content": tool_response
                            })
                
                # Rerun to update the interface
                st.rerun()
                
        except Exception as e:
            response_time = time.time() - start_time
            
            # Update workflow metrics with error
            step_data = {
                "action": "message_processed",
                "duration": response_time,
                "status": "error",
                "error": str(e),
                "response_time": response_time
            }
            self.update_workflow_metrics(step_data)
            
            st.error(f"Error processing message: {str(e)}")
            st.exception(e)
    
    def export_chat_history(self):
        """Export chat history to JSON"""
        if st.session_state.chat_history:
            chat_data = {
                "template": st.session_state.selected_template,
                "timestamp": st.session_state.get("session_start_time"),
                "messages": st.session_state.chat_history
            }
            
            return json.dumps(chat_data, indent=2)
        return None
    
    def import_chat_history(self, chat_data: str = None):
        """Import chat history from JSON"""
        if chat_data is None or not chat_data.strip():
            st.warning("No chat data provided for import")
            return
        
        try:
            data = json.loads(chat_data)
            st.session_state.chat_history = data.get("messages", [])
            st.session_state.selected_template = data.get("template", "default_assistant")
            st.rerun()
        except Exception as e:
            st.error(f"Error importing chat history: {str(e)}")
    
    def render(self):
        """Render workflow status and controls"""
        template = self.template_manager.get_template(st.session_state.selected_template)
        tab1, tab2, tab3, tab4 = st.tabs(["🔄 Workflow Status", "📊 Analytics", "🔍 Workflow Viz", "📈 Stats"])
        
        with tab1:
            if st.session_state.chat_session:
                workflow_type = st.session_state.get("selected_workflow_type", template.get("workflow_type", "basic_chat"))
                st.info(f"🔄 **Active Workflow:** {workflow_type.replace('_', ' ').title()}")
            else:
                st.info("💤 **No active session**")
            
            self.render_chat_messages()
            self.render_chat_input()
            
        with tab2:
            self.render_conversation_analytics()
        
        with tab3:
            self.render_workflow_visualization(self.template_manager.get_template(st.session_state.selected_template))
        
        with tab4:
            self.show_performance_stats()
                
                
        
        # # # Workflow visualization (if enabled)
        # # if st.session_state.get("show_workflow_viz", False):
            
        
        
        # # Analytics panel (if enabled)
        # if st.session_state.get("show_analytics", False):
            
    
    def render_workflow_visualization(self, template: Dict[str, Any]):
        """Render workflow visualization"""
        st.subheader("🔍 Workflow Visualization")
        
        if not st.session_state.chat_session:
            st.info("No active session to visualize")
            return
        workflow_type = st.session_state.get("selected_workflow_type", template.get("workflow_type", "basic_chat"))
        
        # Create workflow diagram
        self.create_workflow_diagram(workflow_type)
        
        # Show workflow steps
        self.render_workflow_steps()
    
    def create_workflow_diagram(self, workflow_type: str = "basic_chat"):
        """Create a visual diagram of the current workflow"""
        # Define workflow nodes and connections
        workflow_configs = {
            "basic_chat": {
                "nodes": ["Start", "Chatbot", "End"],
                "edges": [("Start", "Chatbot"), ("Chatbot", "End")],
                "colors": {"Start": "green", "Chatbot": "blue", "End": "red"}
            },
            "advanced_tools": {
                "nodes": ["Start", "Chatbot", "Tool Planner", "Tools", "End"],
                "edges": [("Start", "Chatbot"), ("Chatbot", "Tool Planner"), 
                         ("Tool Planner", "Tools"), ("Tools", "Chatbot"), ("Chatbot", "End")],
                "colors": {"Start": "green", "Chatbot": "blue", "Tool Planner": "orange", 
                          "Tools": "purple", "End": "red"}
            },
            "research_assistant": {
                "nodes": ["Start", "Research Node", "Search Tools", "Analysis", "End"],
                "edges": [("Start", "Research Node"), ("Research Node", "Search Tools"),
                         ("Search Tools", "Analysis"), ("Analysis", "End")],
                "colors": {"Start": "green", "Research Node": "blue", "Search Tools": "purple",
                          "Analysis": "orange", "End": "red"}
            },
            "code_assistant": {
                "nodes": ["Start", "Code Node", "Code Tools", "Validation", "End"],
                "edges": [("Start", "Code Node"), ("Code Node", "Code Tools"),
                         ("Code Tools", "Validation"), ("Validation", "End")],
                "colors": {"Start": "green", "Code Node": "blue", "Code Tools": "purple",
                          "Validation": "orange", "End": "red"}
            }
        }
        
        config = workflow_configs.get(workflow_type, workflow_configs["basic_chat"])
        
        # Create network diagram
        fig = go.Figure()
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for i, node in enumerate(config["nodes"]):
            node_x.append(i * 2)
            node_y.append(0)
            node_text.append(node)
            node_colors.append(config["colors"].get(node, "gray"))
        
        # Add edges
        edge_x = []
        edge_y = []
        
        for edge in config["edges"]:
            start_idx = config["nodes"].index(edge[0])
            end_idx = config["nodes"].index(edge[1])
            
            edge_x.extend([node_x[start_idx], node_x[end_idx], None])
            edge_y.extend([node_y[start_idx], node_y[end_idx], None])
        
        # Add edge traces
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add node traces
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=50, color=node_colors, line=dict(width=2, color='black')),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=12, color="white"),
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Current Workflow: {workflow_type.replace('_', ' ').title()}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Workflow Flow Diagram",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="black", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_workflow_steps(self):
        """Render current workflow execution steps"""
        st.subheader("📋 Workflow Execution Steps")
        
        workflow_steps = st.session_state.workflow_metrics.get("workflow_steps", [])
        
        if not workflow_steps:
            st.info("No workflow steps recorded yet")
            return
        
        # Create a timeline of workflow steps
        for i, step in enumerate(workflow_steps):
            with st.expander(f"Step {i+1}: {step.get('action', 'Unknown')} - {step.get('timestamp', 'Unknown time')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Action:** {step.get('action', 'Unknown')}")
                    st.write(f"**Duration:** {step.get('duration', 0):.2f}s")
                    st.write(f"**Status:** {step.get('status', 'Unknown')}")
                
                with col2:
                    if step.get('tool_calls'):
                        st.write("**Tool Calls:**")
                        for tool_call in step['tool_calls']:
                            st.write(f"• {tool_call.get('name', 'Unknown')}")
                    
                    if step.get('error'):
                        st.error(f"**Error:** {step['error']}")
    
    def render_conversation_analytics(self):
        """Render conversation analytics and performance metrics"""
        st.subheader("📊 Conversation Analytics")
        
        analytics = st.session_state.conversation_analytics
        metrics = st.session_state.workflow_metrics
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Messages",
                metrics.get("message_count", 0),
                help="Total messages in this session"
            )
        
        with col2:
            st.metric(
                "Tool Calls",
                metrics.get("tool_calls", 0),
                help="Total tool calls made"
            )
        
        with col3:
            session_duration = (datetime.now() - metrics.get("session_start", datetime.now())).total_seconds()
            st.metric(
                "Session Duration",
                f"{session_duration:.0f}s",
                help="Time since session started"
            )
        
        with col4:
            efficiency = analytics.get("workflow_efficiency", 0)
            st.metric(
                "Efficiency",
                f"{efficiency:.1f}%",
                help="Workflow efficiency score"
            )
        
        # Performance charts
        self.render_performance_charts()
        
        # Tool usage statistics
        self.render_tool_usage_stats()
    
    def render_performance_charts(self):
        """Render performance charts"""
        st.subheader("📈 Performance Trends")
        
        performance_data = st.session_state.workflow_metrics.get("performance_data", [])
        
        if len(performance_data) < 2:
            st.info("Collecting performance data...")
            return
        
        # Prepare data
        timestamps = [data["timestamp"] for data in performance_data]
        response_times = [data["response_time"] for data in performance_data]
        token_counts = [data.get("tokens", 0) for data in performance_data]
        
        # Response time chart
        fig_response = go.Figure()
        fig_response.add_trace(go.Scatter(
            x=timestamps,
            y=response_times,
            mode='lines+markers',
            name='Response Time (s)',
            line=dict(color='blue')
        ))
        
        fig_response.update_layout(
            title="Response Time Over Time",
            xaxis_title="Time",
            yaxis_title="Response Time (seconds)",
            height=300
        )
        
        st.plotly_chart(fig_response, use_container_width=True)
        
        # Token usage chart
        if any(token_counts):
            fig_tokens = go.Figure()
            fig_tokens.add_trace(go.Scatter(
                x=timestamps,
                y=token_counts,
                mode='lines+markers',
                name='Token Count',
                line=dict(color='green')
            ))
            
            fig_tokens.update_layout(
                title="Token Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Token Count",
                height=300
            )
            
            st.plotly_chart(fig_tokens, use_container_width=True)
    
    def render_tool_usage_stats(self):
        """Render tool usage statistics"""
        st.subheader("🔧 Tool Usage Statistics")
        
        tool_stats = st.session_state.conversation_analytics.get("tool_usage_stats", {})
        
        if not tool_stats:
            st.info("No tool usage data available")
            return
        
        # Create tool usage chart
        tools = list(tool_stats.keys())
        usage_counts = list(tool_stats.values())
        
        fig = go.Figure(data=[
            go.Bar(x=tools, y=usage_counts, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Tool Usage Frequency",
            xaxis_title="Tools",
            yaxis_title="Usage Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tool details table
        st.subheader("Tool Details")
        tool_data = []
        for tool_name, count in tool_stats.items():
            tool_info = self.tool_manager.get_tool_info(tool_name)
            tool_data.append({
                "Tool": tool_name,
                "Usage Count": count,
                "Description": tool_info.get("description", "No description"),
                "Category": tool_info.get("category", "Unknown")
            })
        
        if tool_data:
            st.dataframe(tool_data, use_container_width=True)
    
    def show_performance_stats(self):
        """Show detailed performance statistics"""
        st.subheader("📊 Detailed Performance Statistics")
        
        # Session statistics
        session_stats = self.langgraph_manager.get_session_statistics()
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        st.write("**Session Statistics:**")
        st.json(session_stats)
        
        # with col2:
            # Tool statistics
        tool_stats = self.tool_manager.get_tool_statistics()
        st.write("**Tool Statistics:**")
        st.json(tool_stats)
    
    def update_workflow_metrics(self, step_data: Dict[str, Any] = None):
        """Update workflow metrics with new step data"""
        if step_data is None:
            return
        
        metrics = st.session_state.workflow_metrics
        
        # Add workflow step
        step_data["timestamp"] = datetime.now()
        metrics["workflow_steps"].append(step_data)
        
        # Update performance data
        if "response_time" in step_data:
            metrics["performance_data"].append({
                "timestamp": step_data["timestamp"],
                "response_time": step_data["response_time"],
                "tokens": step_data.get("tokens", 0)
            })
        
        # Update counters
        if step_data.get("action") == "message_processed":
            metrics["message_count"] += 1
        
        if step_data.get("tool_calls"):
            metrics["tool_calls"] += len(step_data["tool_calls"])
            
            # Update tool usage stats
            for tool_call in step_data["tool_calls"]:
                tool_name = tool_call.get("name", "unknown")
                if tool_name not in st.session_state.conversation_analytics["tool_usage_stats"]:
                    st.session_state.conversation_analytics["tool_usage_stats"][tool_name] = 0
                st.session_state.conversation_analytics["tool_usage_stats"][tool_name] += 1
        
        # Keep only last 100 steps
        if len(metrics["workflow_steps"]) > 100:
            metrics["workflow_steps"] = metrics["workflow_steps"][-100:]
        
        if len(metrics["performance_data"]) > 100:
            metrics["performance_data"] = metrics["performance_data"][-100:]
