# 🤖 DurgasAI - Advanced AI Agent Platform

![DurgasAI Banner](https://img.shields.io/badge/DurgasAI-AI%20Agent%20Platform-blue?style=for-the-badge&logo=robot)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green?logo=chainlink)](https://langchain.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)

**DurgasAI** is a comprehensive AI agent platform that combines the power of **Streamlit**, **LangChain**, and **HuggingFace** to deliver an exceptional conversational AI experience. Built with modern web technologies and advanced language models, it offers a seamless interface for interacting with multiple AI models.

## 🌟 Key Features

### 🎯 **Multiple AI Models**
- **Mistral 7B Instruct**: Advanced instruction-following capabilities
- **Zephyr 7B Beta**: Excellent conversational AI model
- **Flan T5 Large**: Google's instruction-tuned model
- **DialoGPT Medium**: Microsoft's conversational model
- **BlenderBot 400M**: Facebook's open-domain chatbot

### 🧠 **LangChain Integration**
- Advanced conversation memory management
- Context-aware responses
- Chain-of-thought reasoning
- Custom prompt templates
- Session persistence

### 🎨 **Modern UI/UX**
- Responsive design with custom CSS
- Real-time chat interface
- Interactive components
- Dark/light theme support
- Mobile-friendly layout

### 🔧 **Advanced Configuration**
- Customizable system prompts
- Adjustable generation parameters
- Model switching on-the-fly
- Export/import conversations
- Analytics dashboard

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- HuggingFace account and API token
- 8GB+ RAM (for local models)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DurgasAI.git
   cd DurgasAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "HUGGINGFACE_API_TOKEN=your_token_here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage Guide

### 🔑 Getting Your API Token

1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token (starts with `hf_`)
4. Enter it in the application sidebar

### 🤖 Choosing Models

#### **For Beginners**
- **Zephyr 7B Beta**: Best overall conversational model
- **Mistral 7B Instruct**: Great for following instructions

#### **For Specific Tasks**
- **Code Help**: Mistral 7B Instruct
- **Creative Writing**: BlenderBot 400M
- **Educational Content**: Flan T5 Large
- **Casual Chat**: DialoGPT Medium

### ⚙️ Configuration Options

#### **Temperature** (0.1 - 2.0)
- **Low (0.1-0.3)**: Focused, consistent responses
- **Medium (0.4-0.8)**: Balanced creativity and consistency
- **High (0.9-2.0)**: Creative, varied responses

#### **Max Tokens** (50-1000)
- **Short (50-150)**: Quick responses
- **Medium (150-500)**: Detailed answers
- **Long (500-1000)**: Comprehensive explanations

#### **System Prompts**
- **Helpful Assistant**: General-purpose conversations
- **Code Expert**: Programming and technical help
- **Creative Writer**: Storytelling and creative content
- **Tutor**: Educational and learning support
- **Custom**: Define your own AI personality

## 🏗️ Architecture

```
DurgasAI/
├── app.py                 # Main Streamlit application
├── pages/
│   ├── aiagent.py        # AI Agent chat interface
│   ├── css/
│   │   └── styles.css    # Custom styling
│   ├── js/
│   │   └── app.js        # JavaScript enhancements
│   └── component/
│       └── chat_components.py  # Custom Streamlit components
├── utils/
│   ├── config.py         # Application configuration
│   ├── model_manager.py  # AI model management
│   ├── ui_helpers.py     # UI utility functions
│   └── error_handler.py  # Error handling utilities
├── logs/                 # Application logs
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Required
HUGGINGFACE_API_TOKEN=your_hf_token

# Optional
OPENAI_API_KEY=your_openai_key
LOG_LEVEL=INFO
MAX_MESSAGE_HISTORY=50
```

### Custom Models

To add custom models, edit `utils/config.py`:

```python
AVAILABLE_MODELS = {
    "custom_model": ModelConfig(
        name="Your Custom Model",
        model_id="your-org/your-model",
        provider=ModelProvider.HUGGINGFACE_API,
        description="Your model description",
        max_tokens=512,
        temperature=0.7
    )
}
```

## 🎯 Use Cases

### 💼 **Business Applications**
- Customer support automation
- Content generation
- Data analysis assistance
- Meeting summaries
- Email drafting

### 🎓 **Educational Use**
- Tutoring and homework help
- Research assistance
- Language learning
- Concept explanations
- Study guides

### 💻 **Development Support**
- Code review and debugging
- Documentation generation
- API design assistance
- Architecture planning
- Best practices guidance

### 🎨 **Creative Projects**
- Story and script writing
- Brainstorming sessions
- Character development
- Plot generation
- Creative problem solving

## 📊 Performance & Optimization

### **System Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 16GB+ |
| CPU | 2 cores | 4+ cores |
| Storage | 2GB | 10GB+ |
| Internet | Stable connection | High-speed broadband |

### **Performance Tips**

1. **Use API models** for faster responses
2. **Limit max tokens** for quicker generation
3. **Clear chat history** periodically
4. **Use appropriate temperature** settings
5. **Monitor system resources**

## 🛠️ Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking
mypy .
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

#### **"No model loaded" Error**
- Ensure you've clicked "Load Model"
- Check your API token is valid
- Verify internet connection

#### **API Rate Limit Exceeded**
- Wait a few minutes before retrying
- Consider using a different model
- Check HuggingFace usage limits

#### **Memory Issues**
- Use API models instead of local
- Reduce max tokens setting
- Clear chat history
- Restart the application

#### **Slow Responses**
- Try smaller models
- Check internet speed
- Reduce generation parameters
- Use API endpoints

#### **TensorFlow Warnings**
If you see TensorFlow warnings like:
```
oneDNN custom operations are on. You may see slightly different numerical results...
WARNING:tensorflow:From ... The name tf.reset_default_graph is deprecated...
```

These are informational messages and don't affect functionality. To suppress them:

1. **Automatic suppression**: The application automatically sets environment variables
2. **Manual setup**: Run `python env_setup.py` before starting
3. **Environment variables**:
   ```bash
   export TF_ENABLE_ONEDNN_OPTS=0
   export TF_CPP_MIN_LOG_LEVEL=2
   ```

### Getting Help

- 📖 Check the [documentation](docs/)
- 🐛 Report issues on [GitHub](https://github.com/yourusername/DurgasAI/issues)
- 💬 Join our [Discord community](https://discord.gg/durgasai)
- 📧 Email support: support@durgasai.com

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ Initial release
- ✅ Multiple AI model support
- ✅ LangChain integration
- ✅ Modern UI/UX
- ✅ Error handling
- ✅ Export functionality

### Upcoming Features
- 🔄 Voice input/output
- 🔄 File upload support
- 🔄 Multi-language support
- 🔄 Plugin system
- 🔄 API endpoints

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Streamlit](https://streamlit.io)** - Amazing web app framework
- **[LangChain](https://langchain.com)** - Powerful LLM framework
- **[HuggingFace](https://huggingface.co)** - Open-source AI models
- **[OpenAI](https://openai.com)** - Inspiration and research
- **Community contributors** - Thank you for your support!

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/DurgasAI&type=Date)](https://star-history.com/#yourusername/DurgasAI&Date)

---

<div align="center">
  <p>Made with ❤️ by the DurgasAI Team</p>
  <p>
    <a href="https://github.com/yourusername/DurgasAI">GitHub</a> •
    <a href="https://durgasai.com">Website</a> •
    <a href="https://twitter.com/durgasai">Twitter</a> •
    <a href="https://discord.gg/durgasai">Discord</a>
  </p>
</div>
