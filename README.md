# Raspberry Pi AI Assistant

## Features

- Split-screen terminal interface with command history
- Secure API key storage with encryption
- Automatic command execution for system queries
- Support for multiple AI providers:
  - Google Gemini (1.5 Pro, 2.0)
  - Anthropic Claude
  - Local models via Ollama
- Smart handling of system commands and information
- Context-aware responses based on your Pi's configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rpi-ai-helper.git
cd rpi-ai-helper
```

2. Create and activate a virtual environment:
```bash
python3 -m venv pi_assistant_env
source pi_assistant_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the assistant:
```bash
python3 pi_assistant.py
```

## Configuration

On first run, you'll be prompted to choose between:
1. Cloud Models:
   - Google Gemini (requires Google AI API key)
   - Claude (requires Anthropic API key)
2. Local Models (requires Ollama installation)

### Using Cloud Models

#### Google Gemini
1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. When prompted, enter your API key
3. Choose from available Gemini models (1.5 Pro, 2.0)
4. Create a password to encrypt the API key

#### Claude
1. Get an API key from [Anthropic](https://www.anthropic.com/)
2. When prompted, enter your API key
3. Create a password to encrypt the API key

### Using Local Models (Ollama)
1. The assistant will offer to install Ollama if not present
2. Choose from available models optimized for Raspberry Pi
3. Models are downloaded and managed automatically

## Example Use Cases

1. System Information:
```
You: what's the CPU temperature?
Assistant: The current CPU temperature is 45.6°C
```

2. Hardware Details:
```
You: tell me about my pi
Assistant: You're running a Raspberry Pi 4 Model B with 4GB RAM
OS: Raspberry Pi OS (64-bit)
```

3. Storage Information:
```
You: how much disk space is left?
Assistant: Available storage on root partition: 25.3GB free of 32GB
```

4. Service Management:
```
You: is the SSH service running?
Assistant: SSH service status: active (running)
```

## Tips
- Use "next question" to start a fresh conversation
- Type "exit" to quit the assistant
- The last 3 prompts are shown in red at the bottom of the screen
- System information commands execute automatically
- Complex operations will ask for confirmation before proceeding
- Each model has different strengths:
  - Gemini: Great for technical tasks and code
  - Claude: Strong at analysis and explanation
  - Local: Works offline, good for basic tasks

## Requirements
- Python 3.7+
- Raspberry Pi running Raspberry Pi OS
- Internet connection (for cloud models)
- 4GB+ RAM recommended for local models

## License
MIT License - See LICENSE file for details 