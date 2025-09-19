"""
Custom Streamlit components for enhanced chat functionality.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


def enhanced_chat_input(placeholder: str = "Type your message...", key: str = None) -> Optional[str]:
    """
    Enhanced chat input with additional features.
    """
    # Custom HTML for enhanced input
    html_code = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #dee2e6;
        z-index: 1000;
    ">
        <div style="max-width: 1200px; margin: 0 auto; display: flex; gap: 10px; align-items: center;">
            <input 
                type="text" 
                id="enhanced-chat-input"
                placeholder="{placeholder}"
                style="
                    flex: 1;
                    padding: 12px 16px;
                    border: 2px solid #dee2e6;
                    border-radius: 25px;
                    font-size: 16px;
                    outline: none;
                    transition: border-color 0.3s ease;
                "
                onkeydown="handleKeyDown(event)"
                onfocus="this.style.borderColor='#1f77b4'"
                onblur="this.style.borderColor='#dee2e6'"
            >
            <button 
                onclick="sendMessage()"
                style="
                    background: linear-gradient(135deg, #1f77b4, #1565c0);
                    color: white;
                    border: none;
                    border-radius: 50%;
                    width: 48px;
                    height: 48px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s ease;
                "
                onmouseover="this.style.transform='scale(1.1)'"
                onmouseout="this.style.transform='scale(1)'"
            >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
                </svg>
            </button>
        </div>
    </div>
    
    <script>
        function handleKeyDown(event) {{
            if (event.key === 'Enter' && !event.shiftKey) {{
                event.preventDefault();
                sendMessage();
            }}
        }}
        
        function sendMessage() {{
            const input = document.getElementById('enhanced-chat-input');
            const message = input.value.trim();
            if (message) {{
                // Send message to Streamlit
                window.parent.postMessage({{
                    type: 'chat_message',
                    message: message
                }}, '*');
                input.value = '';
            }}
        }}
        
        // Focus on input when loaded
        document.getElementById('enhanced-chat-input').focus();
    </script>
    """
    
    # Render the component
    components.html(html_code, height=80)
    
    return None


def typing_indicator(show: bool = False) -> None:
    """
    Display typing indicator animation.
    """
    if show:
        html_code = """
        <div style="
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 20px;
            margin: 8px 0;
            max-width: 200px;
        ">
            <div style="
                display: flex;
                gap: 4px;
                align-items: center;
            ">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <span style="color: #6c757d; font-size: 14px;">AI is typing...</span>
        </div>
        
        <style>
            .typing-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #6c757d;
                animation: typing 1.4s infinite ease-in-out;
            }
            
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            
            @keyframes typing {
                0%, 80%, 100% {
                    transform: scale(0);
                }
                40% {
                    transform: scale(1);
                }
            }
        </style>
        """
        components.html(html_code, height=60)


def message_bubble(message: str, role: str, timestamp: str = None, metadata: Dict = None) -> None:
    """
    Render a styled message bubble.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    is_user = role == "user"
    
    # Choose colors and alignment based on role
    if is_user:
        bg_color = "linear-gradient(135deg, #e3f2fd, #bbdefb)"
        text_color = "#1565c0"
        align = "flex-end"
        icon = "👤"
        margin = "margin-left: 20%;"
    else:
        bg_color = "linear-gradient(135deg, #f3e5f5, #e1bee7)"
        text_color = "#7b1fa2"
        align = "flex-start"
        icon = "🤖"
        margin = "margin-right: 20%;"
    
    html_code = f"""
    <div style="
        display: flex;
        justify-content: {align};
        margin: 12px 0;
    ">
        <div style="
            background: {bg_color};
            padding: 16px;
            border-radius: 18px;
            max-width: 80%;
            {margin}
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            animation: slideIn 0.3s ease-out;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 8px;
                font-weight: 600;
                color: {text_color};
                font-size: 14px;
            ">
                <span>{icon}</span>
                <span>{role.title()}</span>
            </div>
            
            <div style="
                color: #212529;
                line-height: 1.6;
                word-wrap: break-word;
            ">
                {message}
            </div>
            
            <div style="
                font-size: 12px;
                color: #6c757d;
                margin-top: 8px;
                text-align: right;
            ">
                🕒 {timestamp}
            </div>
        </div>
    </div>
    
    <style>
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
    </style>
    """
    
    components.html(html_code, height=120)


def model_status_card(model_name: str, status: str, details: Dict = None) -> None:
    """
    Display model status card.
    """
    status_colors = {
        "ready": "#28a745",
        "loading": "#ffc107", 
        "error": "#dc3545",
        "not_loaded": "#6c757d"
    }
    
    status_icons = {
        "ready": "✅",
        "loading": "⏳",
        "error": "❌", 
        "not_loaded": "⚪"
    }
    
    color = status_colors.get(status, "#6c757d")
    icon = status_icons.get(status, "⚪")
    
    html_code = f"""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border: 1px solid #dee2e6;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        ">
            <span style="font-size: 20px;">{icon}</span>
            <div>
                <h4 style="
                    margin: 0;
                    color: #1f77b4;
                    font-size: 16px;
                    font-weight: 600;
                ">{model_name}</h4>
                <p style="
                    margin: 4px 0 0 0;
                    color: {color};
                    font-size: 14px;
                    font-weight: 500;
                ">Status: {status.replace('_', ' ').title()}</p>
            </div>
        </div>
        
        {f'''
        <div style="
            background: rgba(0,0,0,0.05);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            color: #6c757d;
        ">
            {details.get('description', 'No additional details')}
        </div>
        ''' if details else ''}
    </div>
    """
    
    components.html(html_code, height=100 + (40 if details else 0))


def quick_action_buttons(actions: List[Dict[str, str]], key: str = None) -> Optional[str]:
    """
    Render quick action buttons.
    """
    buttons_html = ""
    for i, action in enumerate(actions):
        buttons_html += f"""
        <button 
            onclick="selectAction('{action['prompt']}')"
            style="
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
                border: 2px solid #dee2e6;
                border-radius: 12px;
                padding: 16px;
                margin: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                min-width: 200px;
                display: inline-block;
            "
            onmouseover="
                this.style.borderColor='#1f77b4';
                this.style.background='linear-gradient(135deg, #1f77b4, #1565c0)';
                this.style.color='white';
                this.style.transform='translateY(-2px)';
                this.style.boxShadow='0 4px 8px rgba(0,0,0,0.2)';
            "
            onmouseout="
                this.style.borderColor='#dee2e6';
                this.style.background='linear-gradient(135deg, #f8f9fa, #ffffff)';
                this.style.color='inherit';
                this.style.transform='translateY(0)';
                this.style.boxShadow='none';
            "
        >
            <div style="font-size: 24px; margin-bottom: 8px;">{action['icon']}</div>
            <div style="font-weight: 600; margin-bottom: 4px;">{action['title']}</div>
            <div style="font-size: 12px; color: #6c757d;">{action['description']}</div>
        </button>
        """
    
    html_code = f"""
    <div style="
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        padding: 16px;
    ">
        {buttons_html}
    </div>
    
    <script>
        function selectAction(prompt) {{
            window.parent.postMessage({{
                type: 'quick_action',
                prompt: prompt
            }}, '*');
        }}
    </script>
    """
    
    components.html(html_code, height=150)
    
    return None


def conversation_metrics(metrics: Dict[str, Any]) -> None:
    """
    Display conversation metrics in a dashboard style.
    """
    metrics_html = ""
    for key, value in metrics.items():
        metrics_html += f"""
        <div style="
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border: 1px solid #dee2e6;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='translateY(-4px)'" onmouseout="this.style.transform='translateY(0)'">
            <div style="
                font-size: 24px;
                font-weight: 700;
                color: #1f77b4;
                margin-bottom: 8px;
            ">{value}</div>
            <div style="
                font-size: 12px;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 500;
            ">{key.replace('_', ' ')}</div>
        </div>
        """
    
    html_code = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 16px;
        padding: 16px;
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-radius: 12px;
        margin: 16px 0;
    ">
        {metrics_html}
    </div>
    """
    
    components.html(html_code, height=120)


def export_chat_button(messages: List[Dict], filename: str = None) -> None:
    """
    Custom export chat button with enhanced functionality.
    """
    if filename is None:
        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Prepare data for export
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_messages": len(messages),
        "conversation": messages
    }
    
    json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    html_code = f"""
    <button 
        onclick="downloadChat()"
        style="
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            width: 100%;
            justify-content: center;
        "
        onmouseover="
            this.style.background='linear-gradient(135deg, #20c997, #28a745)';
            this.style.transform='translateY(-2px)';
            this.style.boxShadow='0 4px 8px rgba(0,0,0,0.2)';
        "
        onmouseout="
            this.style.background='linear-gradient(135deg, #28a745, #20c997)';
            this.style.transform='translateY(0)';
            this.style.boxShadow='none';
        "
    >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        </svg>
        📥 Export Chat History
    </button>
    
    <script>
        function downloadChat() {{
            const data = {json_data};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{
                type: 'application/json'
            }});
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{filename}';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            // Show success message
            showNotification('Chat history exported successfully!', 'success');
        }}
        
        function showNotification(message, type) {{
            const notification = document.createElement('div');
            notification.innerHTML = `
                <div style="
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #28a745;
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    z-index: 1000;
                    animation: slideIn 0.3s ease-out;
                ">
                    ✅ ${{message}}
                </div>
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {{
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => notification.remove(), 300);
            }}, 3000);
        }}
    </script>
    
    <style>
        @keyframes slideIn {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        
        @keyframes slideOut {{
            from {{ transform: translateX(0); opacity: 1; }}
            to {{ transform: translateX(100%); opacity: 0; }}
        }}
    </style>
    """
    
    components.html(html_code, height=60)
