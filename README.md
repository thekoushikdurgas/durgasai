# DurgasAI - Advanced AI Assistant with LangGraph Integration

DurgasAI is a comprehensive AI assistant platform that integrates LangGraph-powered chatbots with a rich ecosystem of tools, templates, and monitoring capabilities.

## Features

### 🤖 LangGraph-Powered Chatbot

- **Advanced Conversational AI**: Built on LangGraph for sophisticated conversation flows
- **Tool Integration**: Seamlessly integrates with DurgasAI's extensive tool library
- **Template System**: Multiple AI assistant personalities and behaviors
- **Multi-LLM Support**: OpenAI, Groq, and other LLM providers

### 🔧 Tool Management

- **Rich Tool Library**: Mathematical calculations, file operations, web search, data analysis
- **Dynamic Tool Loading**: Tools are loaded from JSON definitions with Python implementations
- **Tool Testing**: Built-in tool testing and validation interface
- **Security**: Sandboxed tool execution with permission controls

### 📋 Template System

- **Multiple Personalities**: Business Analyst, Code Assistant, Creative Writer, Teacher, Technical Expert
- **Customizable**: Create and modify AI assistant templates
- **Function Calling**: Configure which tools each template can use
- **Generation Parameters**: Fine-tune temperature, tokens, and other LLM parameters

### 📊 System Monitoring

- **Real-time Metrics**: CPU, memory, disk, and network monitoring
- **Performance Charts**: Visual performance trends and analytics
- **Process Management**: Monitor running processes and resource usage
- **Alerts**: Performance alerts and system health indicators

### 🎨 Modern UI

- **Streamlit Interface**: Clean, responsive web interface
- **Multi-page Navigation**: Chat, Tools, Templates, Monitoring, Settings
- **Real-time Updates**: Live chat interface with streaming responses
- **Export/Import**: Chat history and configuration management

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd DurgasAI
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   streamlit run app.py
   ```

   or

   ```bash
   python main.py
   ```

## Configuration

### API Keys

Configure your API keys in the Streamlit sidebar:

- **OpenAI API Key**: For GPT models
- **Groq API Key**: For Groq models
- **Tavily API Key**: For web search functionality

### Templates

Templates define AI assistant personalities and are stored in `output/templates/`. Each template includes:

- System instructions
- Function calling configuration
- Generation parameters
- Safety settings

### Tools

Tools are defined in `output/tools/` with JSON metadata and Python implementations in `output/tools/code/`.

## Usage

### Chat Interface

1. Select an AI assistant template from the sidebar
2. Choose your preferred LLM provider and model
3. Start chatting with your AI assistant
4. The assistant can use tools automatically based on your conversation

### Tool Management

1. Navigate to the "Tools" page
2. View available tools by category
3. Test tools with sample inputs
4. Enable/disable tools as needed

### Template Management

1. Go to the "Templates" page
2. Create new templates or modify existing ones
3. Configure function calling and generation parameters
4. Export/import templates for sharing

### System Monitoring

1. Visit the "System Monitor" page
2. View real-time system metrics
3. Monitor performance trends
4. Check for system alerts

## Architecture

### Core Components

- **LangGraph Integration**: Manages conversation flows and tool calling
- **Tool Manager**: Handles tool loading, validation, and execution
- **Template Manager**: Manages AI assistant personalities
- **Config Manager**: Handles application configuration
- **UI Components**: Streamlit-based user interface

### File Structure

```txt
DurgasAI/
├── app.py                 # Main application entry point
├── main.py               # Alternative entry point
├── requirements.txt      # Python dependencies
├── ui/                   # User interface components
│   ├── main.py          # Main UI application
│   └── components/      # UI components
│       ├── chatbot_interface.py
│       ├── tool_manager.py
│       ├── template_manager.py
│       └── system_monitor.py
├── utils/                # Utility modules
│   ├── config_manager.py
│   ├── tool_manager.py
│   ├── template_manager.py
│   └── langgraph_integration.py
├── output/               # Application data
│   ├── config/          # Configuration files
│   ├── tools/           # Tool definitions and code
│   ├── templates/       # AI assistant templates
│   └── sessions/        # Chat sessions
└── docs/                 # Documentation and examples
```

## Development

### Adding New Tools

1. Create a tool definition JSON file in `output/tools/`
2. Implement the tool function in `output/tools/code/`
3. The tool will be automatically loaded and available

### Creating Templates

1. Use the template editor in the UI
2. Or create JSON files in `output/templates/`
3. Follow the template schema for proper validation

### Extending Functionality

- **New UI Pages**: Add components to `ui/components/`
- **New Utilities**: Add modules to `utils/`
- **Configuration**: Update `output/config/config.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- Create an issue in the repository
- Check the documentation in the `docs/` folder
- Review the example implementations

## Acknowledgments

- **LangGraph**: For the powerful conversation framework
- **Streamlit**: For the excellent web interface framework
- **LangChain**: For the comprehensive LLM integration tools
- **OpenAI & Groq**: For providing excellent language models
