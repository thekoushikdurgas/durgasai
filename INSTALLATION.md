# 🚀 DurgasAI Installation Guide

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection
- HuggingFace account (free)

### 2. Installation

#### Option A: Automated Setup (Windows)

```bash
setup.bat
```

#### Option B: Manual Installation

```bash
# Install dependencies
pip install -r requirements-minimal.txt

# Or install the full requirements (optional)
pip install -r requirements.txt
```

### 3. Get Your API Token

1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token (starts with `hf_`)

### 4. Run the Application

#### Option A: Using Startup Script (Windows)

```bash
start_app.bat
```

#### Option B: Manual Start

```bash
streamlit run app.py
```

### 5. Open Your Browser

The application will automatically open at: `http://localhost:8501`

## 🔧 Configuration

### First Time Setup

1. **Enter API Token**: Paste your HuggingFace token in the sidebar
2. **Choose Model**: Select from available AI models
3. **Set System Prompt**: Choose or customize the AI's behavior
4. **Start Chatting**: Click "Load Model" and begin your conversation!

### Recommended Models for Beginners

- **Zephyr 7B Beta**: Best overall conversational model
- **Mistral 7B Instruct**: Great for following instructions
- **Flan T5 Large**: Excellent for educational content

## 🛠️ Troubleshooting

### Common Issues

#### "Module not found" Error

```bash
pip install -r requirements-minimal.txt
```

#### "No model loaded" Error

1. Make sure you entered your API token
2. Click the "Load Model" button
3. Wait for the success message

#### API Rate Limit

- Wait a few minutes before retrying
- Try a different model
- Check HuggingFace usage limits

#### Slow Performance

- Use API models instead of local models
- Reduce max tokens setting
- Check your internet connection

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU | 2 cores | 4+ cores |
| Storage | 2GB | 5GB+ |
| Internet | Stable connection | High-speed broadband |

## 📁 Project Structure

```
DurgasAI/
├── app.py                    # Main application
├── pages/
│   ├── aiagent.py           # AI chat interface
│   ├── css/styles.css       # Custom styling
│   ├── js/app.js           # JavaScript enhancements
│   └── component/
│       └── chat_components.py # UI components
├── utils/
│   ├── config.py           # Configuration
│   ├── model_manager.py    # AI model management
│   ├── ui_helpers.py       # UI utilities
│   └── error_handler.py    # Error handling
├── logs/                   # Application logs
├── requirements-minimal.txt # Core dependencies
├── requirements.txt        # Full dependencies
├── setup.bat              # Windows setup script
├── start_app.bat          # Windows startup script
└── start_app.sh           # Linux/Mac startup script
```

## 🎯 Usage Tips

### Getting Better Responses

1. **Be Specific**: Clear questions get better answers
2. **Use Context**: Reference previous messages
3. **Experiment**: Try different models and settings
4. **Adjust Temperature**: Lower for consistency, higher for creativity

### Model Selection Guide

- **Creative Writing**: BlenderBot 400M, higher temperature
- **Code Help**: Mistral 7B Instruct, lower temperature
- **General Chat**: Zephyr 7B Beta, medium temperature
- **Education**: Flan T5 Large, lower temperature

## 🆘 Getting Help

- 📖 Check the main [README.md](README.md)
- 🐛 Report issues on GitHub
- 💬 Join our Discord community
- 📧 Email: support@durgasai.com

## 🔄 Updates

To update the application:
```bash
git pull origin main
pip install -r requirements-minimal.txt --upgrade
```

---

**Happy Chatting! 🤖✨**
