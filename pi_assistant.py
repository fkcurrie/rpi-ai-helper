import subprocess
import os
import json
import requests
import time
from typing import Dict, Optional, Tuple, List
from config import SYSTEM_PROMPTS
from dotenv import load_dotenv
from anthropic import Anthropic
from llama_cpp import Llama  # For local LLaMA models
import os.path
import wget
from ollama import Client as OllamaClient  # For Ollama models
from cryptography.fernet import Fernet
import base64
import getpass
import shutil
import sys

class ConsoleManager:
    def __init__(self):
        # Get terminal size
        self.term_size = shutil.get_terminal_size()
        self.width = self.term_size.columns
        self.height = self.term_size.lines
        
        # Reserve bottom 3 lines for input
        self.input_height = 3
        self.output_height = self.height - self.input_height - 1  # -1 for separator
        
        # Clear screen and move to top
        print("\033[2J\033[H", end="")
        
        # Draw initial separator
        self.draw_separator()
        
        # Initialize output buffer
        self.output_buffer = []
        
        # Add prompt history
        self.prompt_history = []
        self.max_history = 3

    def draw_separator(self):
        """Draw line separator between output and input areas"""
        separator_pos = self.height - self.input_height - 1
        print(f"\033[{separator_pos};0H" + "-" * self.width)

    def print_output(self, text: str):
        """Print text in the output area with proper word wrapping"""
        # Clear output area
        print("\033[H", end="")
        for i in range(self.output_height):
            print(" " * self.width)
        print("\033[H", end="")
        
        # Add text to buffer and trim if needed
        lines = text.split('\n')
        for line in lines:
            # Word wrap
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= self.width:
                    current_line += word + " "
                else:
                    self.output_buffer.append(current_line)
                    current_line = word + " "
            if current_line:
                self.output_buffer.append(current_line)
        
        # Keep last line visible by trimming from start when buffer is full
        visible_lines = self.output_height - 2  # Leave 2 lines margin
        while len(self.output_buffer) > visible_lines:
            self.output_buffer.pop(0)
        
        # Print buffer with margin at bottom
        print("\033[H", end="")
        empty_lines = max(0, visible_lines - len(self.output_buffer))
        print("\n" * empty_lines, end="")
        for line in self.output_buffer:
            print(f"{line:<{self.width}}")
        
        # Redraw separator
        self.draw_separator()

    def print_input_history(self):
        """Display last 3 prompts in input area"""
        # Clear input area first
        self.clear_input_area()
        
        # Calculate starting position for history
        start_pos = self.height - self.input_height
        
        # Show last 3 prompts (or fewer if history is shorter)
        for i, prompt in enumerate(self.prompt_history[-self.max_history:]):
            # Remove any newlines and ensure the line fits within width
            cleaned_prompt = prompt.replace('\n', ' ')[:self.width]
            print(f"\033[{start_pos + i};0H\033[K\033[31m{cleaned_prompt}\033[0m")

    def get_input(self, prompt: str = "You: ") -> str:
        """Get input from the bottom section"""
        # Show history first
        self.print_input_history()
        
        # Move cursor to input position and get new input
        input_pos = self.height - 1  # Bottom line
        # Remove any newlines from prompt
        cleaned_prompt = prompt.replace('\n', ' ')
        print(f"\033[{input_pos};0H\033[K\033[31m{cleaned_prompt}", end="", flush=True)
        user_input = input()
        print("\033[0m", end="")  # Reset color
        
        # Add to history (full prompt + input) - ensure no newlines
        if user_input.strip():  # Only add non-empty inputs
            full_prompt = f"{cleaned_prompt}{user_input}"
            self.prompt_history.append(full_prompt)
            # Keep only last 3 prompts
            if len(self.prompt_history) > self.max_history:
                self.prompt_history.pop(0)
        
        return user_input

    def clear_input_area(self):
        """Clear the input area"""
        for i in range(self.input_height):
            print(f"\033[{self.height-i};0H\033[K", end="")

class RaspberryPiAssistant:
    def __init__(self):
        """Initialize the assistant"""
        self.os_info = self._get_os_info()
        self.pi_model = self._get_pi_model()
        self.api_key = None
        self.last_query = None
        
        print("\nWelcome to Raspberry Pi Assistant!")
        print(f"Detected System: {self.pi_model}")
        print(f"OS: {self.os_info.get('PRETTY_NAME', 'Unknown')}")
        
        # First ask user preference for model type
        self._handle_model_selection()
        
        # Get available models and select one
        self.model, self.is_local = self._get_available_model()
        print(f"\nUsing model: {self.model}")
        
        # Initialize RAG system
        self._initialize_rag_system()
        
        # Build system context
        self.system_context = self._build_system_context()
        self.conversation_history = []
        
        print(f"\nInitializing {self.model}...")
        self._warmup_model()

        # Only optimize CPU if using a local Ollama model
        if self.is_local and self.ollama_available:
            self._optimize_cpu_for_inference()

    def _get_os_info(self) -> Dict[str, str]:
        """Read and parse /etc/os-release file"""
        os_info = {}
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.rstrip().split('=', 1)
                        os_info[key] = value.strip('"')
            return os_info
        except FileNotFoundError:
            print("Warning: Could not find /etc/os-release")
            return {}

    def _get_pi_model(self) -> str:
        """Get Raspberry Pi model information"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('Model'):
                        return line.split(':')[1].strip()
        except FileNotFoundError:
            return "Unknown Raspberry Pi model"
        return "Unknown Raspberry Pi model"

    def _build_system_context(self) -> str:
        """Build system context for LLM prompts"""
        return f"""You are a Raspberry Pi assistant helping users with their {self.pi_model} running {self.os_info.get('PRETTY_NAME', 'Unknown')}.
Always provide specific advice for Raspberry Pi systems, considering:
- ARM architecture ({self.pi_model})
- Limited resources (RAM, CPU, storage)
- Raspberry Pi OS compatibility
- Common Raspberry Pi use cases

When discussing software or solutions:
1. Focus on ARM-compatible options
2. Consider resource constraints
3. Prefer official Raspberry Pi repositories
4. Format responses clearly with:
   - Short paragraphs
   - Bullet points for lists
   - Clear headings for sections
   - Code blocks for commands
"""

    def execute_command(self, command: str) -> Tuple[bool, str]:
        """Execute a shell command and show real-time output"""
        try:
            # Use Popen to get real-time output
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Store complete output for history
            full_output = []
            
            # Show output in real-time
            print(f"\n=== Executing: {command} ===")
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.rstrip())
                    full_output.append(output)
            
            return_code = process.poll()
            
            if return_code == 0:
                return True, ''.join(full_output)
            else:
                return False, f"Command failed with return code {return_code}\n{''.join(full_output)}"
                
        except Exception as e:
            return False, f"Error executing command: {e}"

    def _is_safe_command(self, command: str) -> bool:
        """Check if a command is safe to execute"""
        dangerous_commands = ['rm -rf', 'mkfs', 'dd', '> /dev', 'format']
        return not any(cmd in command.lower() for cmd in dangerous_commands)

    def _check_local_models(self) -> Dict[str, bool]:
        """Check which local models are available"""
        available_models = {}
        
        # Check Ollama installation and available models
        try:
            ollama_client = OllamaClient()
            models = ollama_client.list()
            available_models['ollama'] = len(models['models']) > 0
            if not available_models['ollama']:
                print("\nOllama is installed but no models found.")
                install = input("Would you like to pull some models? (y/n): ").lower()
                if install == 'y':
                    print("Pulling Ollama models...")
                    ollama_client.pull('codellama')
                    ollama_client.pull('deepseek-coder')
                    available_models['ollama'] = True
        except Exception:
            available_models['ollama'] = False
            print("\nOllama not found. You can install it with:")
            print("curl https://ollama.ai/install.sh | sh")
        
        # Check other local models
        for model_type, path in self.model_paths.items():
            if model_type != 'ollama':
                if not os.path.exists(path):
                    print(f"\n{model_type.capitalize()} model not found.")
                    download = input(f"Would you like to download it? (y/n): ").lower()
                    if download == 'y':
                        print(f"Downloading {model_type} model (this may take a while)...")
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        try:
                            if model_type == 'llama':
                                url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
                            elif model_type == 'deepseek':
                                url = "https://huggingface.co/TheBloke/deepseek-coder-6.7b-base-GGUF/resolve/main/deepseek-coder-6.7b-base.Q4_K_M.gguf"
                            wget.download(url, path)
                            available_models[model_type] = True
                        except Exception as e:
                            print(f"\nError downloading model: {e}")
                            available_models[model_type] = False
                    else:
                        available_models[model_type] = False
                else:
                    available_models[model_type] = True
        
        return available_models

    def _create_ascii_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Create an ASCII table with the given headers and rows"""
        # Define maximum widths for each column
        max_widths = {
            "Model": 30,        # Model name
            "Description": 60,  # Description text
            "Released": 12,     # Date
            "#": 3             # Number
        }
        
        # Calculate optimal column widths within limits
        widths = []
        for i, header in enumerate(headers):
            column = [str(row[i]) for row in rows]
            max_width = max_widths.get(header, 20)  # Default to 20 if not specified
            width = min(max_width, max(len(str(x)) for x in [header] + column))
            widths.append(width)
        
        # Create the table
        separator = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
        result = [separator]
        
        # Add headers
        header = '|' + '|'.join(f' {h:<{w}} ' for h, w in zip(headers, widths)) + '|'
        result.extend([header, separator])
        
        # Add rows with text wrapping for description
        for row in rows:
            # Handle description column wrapping
            desc_col = headers.index("Description") if "Description" in headers else -1
            if desc_col >= 0 and len(str(row[desc_col])) > widths[desc_col]:
                # Wrap description text
                import textwrap
                desc_lines = textwrap.wrap(str(row[desc_col]), widths[desc_col])
                first_line = True
                for desc in desc_lines:
                    if first_line:
                        line = [str(row[i]) if i != desc_col else desc for i in range(len(row))]
                        first_line = False
                    else:
                        line = [''] * len(row)
                        line[desc_col] = desc
                    result.append('|' + '|'.join(f' {str(c):<{w}} ' for c, w in zip(line, widths)) + '|')
            else:
                result.append('|' + '|'.join(f' {str(c):<{w}} ' for c, w in zip(row, widths)) + '|')
        
        result.append(separator)
        return '\n'.join(result)

    def _get_available_model(self) -> Tuple[str, bool]:
        """Get available models and let user choose"""
        models = []
        
        if self.model_provider == "google":
            try:
                import requests
                
                # Get available models from Gemini API
                url = "https://generativelanguage.googleapis.com/v1/models"
                headers = {
                    "x-goog-api-key": self.api_key
                }
                
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    model_list = response.json().get('models', [])
                    for model in model_list:
                        name = model['name'].split('/')[-1]
                        description = model.get('description', 'No description available')
                        version = model.get('version', 'Latest')
                        
                        # Include both pro and flash variants for 1.5 and 2.0
                        if (('gemini-1.5' in name.lower() or 'gemini-2' in name.lower()) and 
                            ('pro' in name.lower() or 'flash' in name.lower()) and
                            'deprecated' not in description.lower() and 
                            'discontinued' not in description.lower() and
                            'vision' not in name.lower()):
                            models.append((name, False, "google", description, version))
                
                # Sort models by version and type
                def sort_key(x):
                    name = x[0]
                    # Primary sort by version
                    if 'gemini-2' in name.lower():
                        version_score = 2
                    else:  # gemini-1.5
                        version_score = 1
                    
                    # Secondary sort by type (pro before flash)
                    type_score = 1 if 'pro' in name.lower() else 0
                    
                    return (version_score, type_score, x[4])
                
                models.sort(key=sort_key, reverse=True)
                
                if not models:
                    print("No Gemini models available. Please try a different model type.")
                    self.model_provider = None
                    self._handle_model_selection()
                    return None, None
                
                # Show model table
                print("\nAvailable models:")
                headers = ["#", "Model", "Description", "Version"]
                rows = []
                for i, (model, _, _, description, version) in enumerate(models, 1):
                    rows.append([str(i), model, description, version])
                print(self._create_ascii_table(headers, rows))
                
                # Get user choice
                while True:
                    choice = input("\nSelect a model number (or press Enter for default): ").strip()
                    if not choice:
                        return models[0][0], models[0][1]  # Default to first (newest) model
                    if choice.isdigit() and 1 <= int(choice) <= len(models):
                        idx = int(choice) - 1
                        return models[idx][0], models[idx][1]
                    print("Invalid selection. Please try again.")
                
            except Exception as e:
                print(f"Error getting Gemini models: {e}")
                print("Please try a different model type.")
                self.model_provider = None
                self._handle_model_selection()
                return None, None

        elif self.model_provider == "anthropic":
            try:
                temp_client = Anthropic(api_key=self.api_key)
                available_models = temp_client.models.list()
                
                # Track latest version of each variant
                latest_models = {
                    'opus': None,
                    'sonnet': None,
                    'haiku': None
                }
                
                # Find latest version of each variant
                for model in available_models:
                    if 'claude-3' in model.id.lower():
                        for variant in latest_models:
                            if variant in model.id.lower():
                                if not latest_models[variant] or model.id > latest_models[variant].id:
                                    latest_models[variant] = model
                
                # Add latest models with descriptions and dates
                descriptions = {
                    'opus': "Most capable model for complex tasks",
                    'sonnet': "Balanced model for general use",
                    'haiku': "Fast model for simple tasks"
                }
                
                # Extract and format dates from model IDs
                for variant, model in latest_models.items():
                    if model:
                        # Extract date from model ID (format: YYYYMMDD)
                        date_str = model.id.split('-')[-1]
                        # Convert to dd/mm/yyyy format
                        release_date = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[:4]}"
                        models.append((model.id, False, 'claude', descriptions[variant], release_date))
                    
            except Exception as e:
                print(f"Error getting Claude models: {e}")
            
        elif self.model_provider == "local":
            models = []
            # Add latest Claude models if API key exists
            if self.api_key:
                try:
                    temp_client = Anthropic(api_key=self.api_key)
                    available_models = temp_client.models.list()
                    
                    # Track latest version of each variant
                    latest_models = {
                        'opus': None,
                        'sonnet': None,
                        'haiku': None
                    }
                    
                    # Find latest version of each variant
                    for model in available_models:
                        if 'claude-3' in model.id.lower():
                            for variant in latest_models:
                                if variant in model.id.lower():
                                    if not latest_models[variant] or model.id > latest_models[variant].id:
                                        latest_models[variant] = model
                    
                    # Add latest models with descriptions and dates
                    descriptions = {
                        'opus': "Most capable model for complex tasks",
                        'sonnet': "Balanced model for general use",
                        'haiku': "Fast model for simple tasks"
                    }
                    
                    # Extract and format dates from model IDs
                    for variant, model in latest_models.items():
                        if model:
                            # Extract date from model ID (format: YYYYMMDD)
                            date_str = model.id.split('-')[-1]
                            # Convert to dd/mm/yyyy format
                            release_date = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[:4]}"
                            models.append((model.id, False, 'claude', descriptions[variant], release_date))
                        
                except Exception as e:
                    print(f"Error getting Claude models: {e}")
            
            if not models:
                print("No models available. Please install a local model or provide an API key.")
                exit(1)
        
        print("\nAvailable models:")
        
        # Prepare simplified table data
        headers = ["#", "Model", "Description", "Released"]
        rows = []
        
        for i, (model, _, _, description, release_date) in enumerate(models, 1):
            # Simplify model name by removing date
            model_name = model.split('-2024')[0]
            rows.append([str(i), model_name, description, release_date])
        
        print(self._create_ascii_table(headers, rows))
        
        # Get user choice
        while True:
            try:
                choice = input("\nSelect a model number (or press Enter for default): ").strip()
                if not choice:
                    model, is_local, model_type, _, _ = models[0]
                    # Initialize client before returning
                    self.client = Anthropic(api_key=self.api_key)
                    return model, is_local
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(models):
                        model, is_local, model_type, _, _ = models[choice_idx]
                        # Initialize client before returning
                        self.client = Anthropic(api_key=self.api_key)
                        return model, is_local
                    else:
                        print("Invalid selection. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid number.")

    def _make_api_request(self, prompt: str) -> str:
        """Make API request with retry logic"""
        try:
            if self.model_provider == "google":
                import requests
                import json
                
                # Add system context to the prompt
                enhanced_prompt = f"""As a Raspberry Pi assistant for a {self.pi_model} running {self.os_info.get('PRETTY_NAME', 'Unknown')}, 
please provide specific advice considering ARM architecture and resource constraints.

User Query: {prompt}

Format your response with:
- Clear sections with headings
- Bullet points for lists
- Short, focused paragraphs
- Code blocks for commands
- Resource-aware recommendations
"""
                
                # Gemini API endpoint
                url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent"
                
                # Request headers
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
                
                # Request body
                data = {
                    "contents": [{
                        "parts": [{
                            "text": enhanced_prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.8,
                    }
                }
                
                # Make request
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    raw_text = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Format the response
                    formatted_text = self._format_gemini_response(raw_text)
                    return formatted_text
                else:
                    return f"Error: {response.text}"
            
            elif self.model_provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.7,
                    system=self.system_context,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
            
            elif self.model_provider == "local":
                if self.is_local and hasattr(self.client, 'chat'):  # Ollama models
                    response = self.client.chat(
                        model=self.model,
                        messages=[{
                            'role': 'system',
                            'content': self.system_context
                        }, {
                            'role': 'user',
                            'content': prompt
                        }],
                        options={
                            'num_gpu': 0,           # CPU only
                            'num_thread': 3,        # Use 3 threads for better balance
                            'num_ctx': 512,         # Reduced context but not too small
                            'num_batch': 512,       # Increased batch size for better throughput
                            'num_keep': 5,          # Limit number of tokens to keep in context
                            'seed': 42,
                            'temperature': 0.7,
                            'top_k': 20,            # More focused token selection
                            'top_p': 0.95,          # Slightly higher nucleus sampling
                            'repeat_penalty': 1.1,
                            'mirostat': 2,          # Enable Mirostat 2.0 sampling
                            'mirostat_tau': 5.0,    # Target entropy
                            'mirostat_eta': 0.1,    # Learning rate
                            'f16_kv': True,         # Use float16 for key/value cache
                            'rope_frequency_base': 10000,  # Adjust attention mechanism
                            'rope_frequency_scale': 0.5
                        }
                    )
                    return response['message']['content']
                else:  # Claude models
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        temperature=0.7,
                        system=self.system_context,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return message.content[0].text
                
        except Exception as e:
            return f"Error communicating with API: {str(e)}"

    def _format_gemini_response(self, text: str) -> str:
        """Format Gemini response for better readability"""
        # Split into lines and remove empty lines at start/end
        lines = text.strip().split('\n')
        
        # Format bullet points and links consistently
        formatted_lines = []
        current_section = ""
        
        for line in lines:
            line = line.strip()
            
            # Handle headings
            if line.upper() == line and len(line) > 3:  # Section heading
                formatted_lines.extend(['', f"\n{line}", '-' * len(line), ''])
                current_section = line
            elif line.startswith('#'):  # Markdown heading
                clean_line = line.lstrip('#').strip()
                formatted_lines.extend(['', clean_line, '-' * len(clean_line), ''])
                current_section = clean_line
            # Handle bullet points
            elif line.startswith('*') or line.startswith('-'):
                formatted_lines.append(f"• {line[1:].strip()}")
            # Handle code blocks
            elif line.startswith('```'):
                if current_section:
                    formatted_lines.append(f"\nCommand for {current_section}:")
                formatted_lines.append('  ' + line.strip('`'))
            # Normal text
            elif line:
                formatted_lines.append(line)
        
        # Remove duplicate empty lines
        result = []
        prev_empty = False
        for line in formatted_lines:
            if line.strip() or not prev_empty:
                result.append(line)
            prev_empty = not line.strip()
        
        return '\n'.join(result)

    def _format_conversation_history(self) -> str:
        """Format recent conversation history for context"""
        if not self.conversation_history:
            return "No previous conversation"
            
        formatted = []
        # Get last 5 exchanges
        for entry in self.conversation_history[-5:]:
            role = entry['role']
            content = entry['content']
            # Include command results if available
            if 'command_result' in entry:
                content += f"\nCommand output: {entry['command_result']}"
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)

    def process_user_query(self, query: str, console: ConsoleManager, analysis_prompt: Optional[str] = None) -> str:
        """Process user query and return response"""
        try:
            # Store the query
            self.last_query = query
            
            # Check if this is a command execution request
            if query.strip().startswith('!') or query.strip().startswith('sudo'):
                return self._handle_command_execution(query)
            
            # For information requests, just get the response from the model
            prompt = analysis_prompt if analysis_prompt else query
            response = self._make_api_request(prompt)
            
            # Add to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': query
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': response
            })
            
            return response
        
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def _handle_command_execution(self, command: str) -> str:
        """Handle execution of system commands"""
        if not self._is_safe_command(command):
            return "Command not executed: This command has been blocked for safety reasons."
        
        success, output = self.execute_command(command)
        if success:
            return f"Command executed successfully:\n{output}"
        else:
            return f"Command failed:\n{output}"

    def execute_with_confirmation(self, command: str) -> Tuple[bool, str]:
        """Execute command with confirmation and real-time output"""
        print(f"\nAbout to execute: {command}")
        if input("Do you want to proceed? (y/n): ").lower() == 'y':
            return self.execute_command(command)
        return False, "Command execution cancelled by user"

    def _warmup_model(self) -> None:
        """Test the model with a simple query"""
        if self.model_provider == "google":
            try:
                response = self._make_api_request("Say 'Hello from Gemini!'")
                if not response.startswith("Error"):
                    print("\nModel test successful!")
                else:
                    print(f"\nWarning: {response}")
            except Exception as e:
                print(f"\nWarning: Model warmup failed: {e}")
                print("The assistant will still try to continue...")
        
        # ... existing warmup code for other models ...

    def format_response(self, response: str, command_results: List[str]) -> str:
        """Format the response for display"""
        formatted = ""
        
        if self.last_query:
            formatted += f"\nQuestion: {self.last_query}\n"
        
        if response:
            formatted += f"\nAnswer: {response}\n"
        
        if command_results:
            formatted += "\nSystem Information:\n"
            for result in command_results:
                formatted += f"{result}\n"
        
        return formatted

    def _check_sudo_needed(self, command: str) -> str:
        """Check if command needs sudo"""
        if not os.geteuid() == 0:  # Not running as root
            if not command.startswith('sudo ') and any(cmd in command for cmd in ['apt-get', 'systemctl', 'service']):
                return f"sudo {command}"
        return command

    def _check_latest_kernel(self) -> str:
        """Check latest available kernel for Raspberry Pi OS"""
        try:
            # Check Raspberry Pi OS GitHub releases or documentation
            response = requests.get("https://github.com/raspberrypi/rpi-firmware/releases")
            if response.status_code == 200:
                # Parse the response to find latest kernel version
                # This is a simplified example - would need proper parsing
                return "latest version information"
        except Exception as e:
            return f"Unable to check latest version: {e}"

    def _check_ollama_installed(self) -> Tuple[bool, bool]:
        """Check if Ollama is installed and if service exists
        Returns: (is_installed, has_service)"""
        # Check for Ollama installation directory
        binary_exists = os.path.exists('/usr/local/bin/ollama') and os.path.exists('/usr/share/ollama')
        
        # Check for service files
        service_files = [
            '/etc/systemd/system/ollama.service',  # System service
            os.path.expanduser('~/.local/share/systemd/user/ollama.service')  # User service
        ]
        service_exists = any(os.path.exists(f) for f in service_files)
        
        if binary_exists:
            print("Found Ollama installation at /usr/share/ollama")
            
        return binary_exists, service_exists

    def _uninstall_ollama(self) -> bool:
        """Uninstall Ollama and clean up files"""
        try:
            print("\nUninstalling Ollama...")
            cleanup_commands = [
                "sudo systemctl stop ollama",
                "sudo systemctl disable ollama",
                "sudo rm -f /usr/local/bin/ollama",
                "sudo rm -f /etc/systemd/system/ollama.service",
                "rm -f ~/.local/share/systemd/user/ollama.service",
                "sudo rm -rf /usr/local/lib/ollama",
                "sudo systemctl daemon-reload",
                "systemctl --user daemon-reload"
            ]
            
            for cmd in cleanup_commands:
                try:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd.split(), check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Warning: {e}")
                    continue
            
            print("Ollama uninstalled successfully!")
            return True
        except Exception as e:
            print(f"Error uninstalling Ollama: {e}")
            return False

    def _install_and_start_ollama(self) -> bool:
        """Install and start Ollama service"""
        try:
            print("\nInstalling Ollama...")
            install_cmd = ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
            process = subprocess.Popen(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Show output in real-time using select
            import select
            
            # Set up polling
            poller = select.poll()
            poller.register(process.stdout, select.POLLIN)
            poller.register(process.stderr, select.POLLIN)
            
            # Keep reading while process is running
            while process.poll() is None:
                events = poller.poll(100)  # 100ms timeout
                for fd, event in events:
                    if fd == process.stdout.fileno():
                        line = process.stdout.readline()
                        if line:
                            print(line.strip())
                    elif fd == process.stderr.fileno():
                        line = process.stderr.readline()
                        if line and "No NVIDIA/AMD GPU detected" not in line:
                            print(line.strip())
            
            # Get any remaining output
            stdout, stderr = process.communicate()
            if stdout:
                print(stdout.strip())
            if stderr and "No NVIDIA/AMD GPU detected" not in stderr:
                print(stderr.strip())
            
            # Check return code
            if process.returncode != 0:
                print(f"Installation failed with return code: {process.returncode}")
                return False
            
            # Give systemd time to start the service
            print("\nWaiting for Ollama service to initialize...")
            time.sleep(5)
            
            # Verify API is accessible with retries
            return self._verify_ollama_api()
            
        except Exception as e:
            print(f"\nError setting up Ollama: {e}")
            return False

    def _install_local_model(self, model_type: str, model_info: Dict) -> bool:
        """Install a local model with proper error handling"""
        try:
            print(f"\nPreparing to install {model_info['name']}...")
            print(f"Size: ~{model_info['size_mb']/1024:.1f}GB")
            print(f"Requirements: {model_info['requirements']}")
            
            install = input("Would you like to install this model? (y/n): ").lower()
            if install != 'y':
                print("Model installation cancelled.")
                return False

            # Check if this is an Ollama-based model
            if model_info.get('install_method') == 'ollama':
                print("\nSetting up Ollama...")
                if not self._install_and_start_ollama():
                    return False
                
                try:
                    # Pull the model
                    print("Pulling Gemma model...")
                    subprocess.run(['ollama', 'pull', 'gemma2:2b'], check=True)
                    
                    from ollama import Client as OllamaClient
                    self.client = OllamaClient()
                    
                    # Test the model
                    print("\nTesting model...")
                    response = self.client.chat(model='gemma2:2b', messages=[{
                        'role': 'user',
                        'content': 'What is the time in London?'
                    }])
                    if response:
                        print("Model installed and working!")
                        print(f"Response: {response['message']['content']}")
                    return True
                except Exception as e:
                    print(f"Error installing via Ollama: {e}")
                    print("Try running 'systemctl --user start ollama' manually")
                    return False

            # Regular GGUF model installation
            print(f"\nInstalling dependencies...")
            subprocess.run(['pip', 'install', 'llama-cpp-python'], check=True)
            
            print(f"\nDownloading {model_type} model...")
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{model_type}.gguf"
            
            if not self._try_download_from_sources(model_info, model_path):
                print("Failed to download from all sources")
                return False
            
            # Verify the downloaded file
            print("Verifying model file...")
            try:
                self.client = Llama(model_path=model_path)
                print("Model loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Cleaning up failed download...")
                os.remove(model_path)
                return False
                    
        except Exception as e:
            print(f"\nError during installation: {e}")
            if 'model_path' in locals() and os.path.exists(model_path):
                os.remove(model_path)
            return False

    def _try_download_from_sources(self, model_info: Dict, model_path: str) -> bool:
        """Try downloading from multiple sources"""
        url = model_info['url']
        try:
            print(f"\nTrying download from: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': '*/*'
            }
            
            response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                if total_size > 0:
                    with open(model_path, 'wb') as f:
                        downloaded = 0
                        for data in response.iter_content(chunk_size=1024*1024):
                            downloaded += len(data)
                            f.write(data)
                            done = int(50 * downloaded / total_size)
                            mb_done = downloaded / (1024*1024)
                            mb_total = total_size / (1024*1024)
                            print(f"\rDownloading: [{'=' * done}{' ' * (50-done)}] {mb_done:.1f}MB/{mb_total:.1f}MB", end='')
                    print("\nDownload complete!")
                    return True
            else:
                print(f"Error: Unable to download model (Status code: {response.status_code})")
        except Exception as e:
            print(f"Error with download: {e}")
        
        return False

    def _verify_ollama_api(self, max_retries=3, delay=2) -> bool:
        """Verify Ollama API is accessible with retries"""
        print("Verifying Ollama API connection...")
        for i in range(max_retries):
            try:
                from ollama import Client as OllamaClient
                client = OllamaClient()
                models = client.list()
                print("Successfully connected to Ollama API")
                return True
            except Exception as e:
                if i < max_retries - 1:
                    print(f"Waiting for Ollama service to be ready (attempt {i+1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    print(f"Error connecting to Ollama API: {e}")
                    print("Try running 'systemctl --user restart ollama' manually")
        return False

    def _get_installed_ollama_models(self) -> List[Dict]:
        """Get list of installed Ollama models"""
        try:
            process = subprocess.run(['ollama', 'list'], 
                                   stdout=subprocess.PIPE, 
                                   text=True)
            if process.returncode == 0:
                models = []
                for line in process.stdout.strip().split('\n')[1:]:  # Skip header
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            models.append({
                                'name': parts[0],
                                'size': parts[2]
                            })
                return models
        except Exception as e:
            print(f"Error getting installed models: {e}")
        return []

    def _initialize_rag_system(self) -> None:
        """Initialize RAG system components"""
        self.model_paths = {
            'llama': "models/llama-2-7b-chat.Q4_K_M.gguf",
            'deepseek': "models/deepseek-coder-6.7b-base.Q4_K_M.gguf",
            'ollama': None  # Ollama manages its own models
        }

    def _get_ollama_model_description(self, model_name: str) -> str:
        """Get description for Ollama model"""
        descriptions = {
            'gemma': "Google's lightweight model optimized for Raspberry Pi",
            'mistral': "Strong general purpose model with good performance",
            'codellama': "Specialized for code generation and analysis",
            'deepseek-coder': "Optimized for programming tasks",
            'neural-chat': "Balanced chat model with good reasoning",
            'llama2': "Meta's general purpose model",
            'phi': "Microsoft's efficient small model",
            'stable-code': "Focused on code generation"
        }
        
        for key, desc in descriptions.items():
            if key in model_name.lower():
                return desc
        return "Local language model"

    def _build_rag_context(self, query: str) -> str:
        """Build RAG context from system info and conversation history"""
        # System information context
        system_info = self._get_system_info()
        
        context = f"""Current system state:
{self._format_system_info(system_info)}

Recent conversation:
{self._format_conversation_history()}

User query: {query}

Instructions for responding:
1. FIRST RESPONSE SHOULD:
   - Understand and acknowledge the user's query
   - Ask clarifying questions about their needs and use case
   - Provide general information and comparisons
   - Explain concepts and trade-offs
   - Share relevant documentation links
   - DO NOT suggest installations or commands

2. IF USER REQUESTS INSTALLATION:
   - First confirm their specific requirements
   - Present different options with pros/cons
   - Ask which solution they'd prefer
   - Only then provide an installation plan

3. ALWAYS:
   - Be informative and educational
   - Consider Raspberry Pi ARM architecture
   - Focus on understanding before action
   - Break complex topics into digestible parts
   - Let the user drive the decision-making

4. NEVER:
   - Jump straight to installation commands
   - Make assumptions about user needs
   - Execute commands without discussion
   - Provide commands without context

5. Example flow:
   - Explain available options
   - Discuss trade-offs
   - Ask for preferences
   - Confirm understanding
   - THEN discuss implementation if requested
"""
        return context

    def _get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'device': self.pi_model,
            'os': self.os_info.get('PRETTY_NAME', 'Unknown'),
            'kernel': self._get_kernel_version(),
            'memory': self._get_memory_info(),
            'storage': self._get_storage_info(),
            'services': self._get_service_status()
        }

    def _format_system_info(self, system_info: Dict) -> str:
        """Format system information for display"""
        formatted = ""
        for key, value in system_info.items():
            formatted += f"{key}: {value}\n"
        return formatted

    def _get_kernel_version(self) -> str:
        """Get kernel version"""
        try:
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return "Unknown"
            
    def _get_memory_info(self) -> str:
        """Get memory information"""
        try:
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return "Unknown"
            
    def _get_storage_info(self) -> str:
        """Get storage information"""
        try:
            result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return "Unknown"
            
    def _get_service_status(self) -> str:
        """Get status of important services"""
        services = ['ssh', 'ollama']
        status = []
        for service in services:
            try:
                result = subprocess.run(['systemctl', 'is-active', service], 
                                      capture_output=True, text=True)
                status.append(f"{service}: {result.stdout.strip()}")
            except Exception:
                status.append(f"{service}: unknown")
        return ', '.join(status)

    def _check_ollama_service(self) -> bool:
        """Check if Ollama service is running"""
        try:
            # Check system service first
            result = subprocess.run(
                ['systemctl', 'status', 'ollama'],
                capture_output=True,
                text=True
            )
            if 'active (running)' in result.stdout:
                print("Ollama system service is running")
                return True
                
            # Check user service
            result = subprocess.run(
                ['systemctl', '--user', 'status', 'ollama'],
                capture_output=True,
                text=True
            )
            if 'active (running)' in result.stdout:
                print("Ollama user service is running")
                return True
                
            print("Ollama service is not running. You can start it with:")
            print("  systemctl --user start ollama")
            return False
            
        except Exception as e:
            print(f"Error checking Ollama service: {e}")
            return False

    def _optimize_cpu_for_inference(self):
        """Configure CPU settings for better inference performance on Raspberry Pi"""
        try:
            # Get Ollama process ID (only the first one if multiple)
            process = subprocess.run(['pgrep', 'ollama'], 
                                  capture_output=True, 
                                  text=True)
            if process.returncode == 0:
                ollama_pid = process.stdout.strip().split('\n')[0]
                
                try:
                    # Set CPU governor on RPi
                    for i in range(4):  # RPi 5 has 4 cores
                        governor_path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
                        if os.path.exists(governor_path):
                            subprocess.run(['sudo', 'sh', '-c', f'echo performance > {governor_path}'], check=True)
                    print("Set CPU governors to performance mode")
                except Exception as e:
                    print(f"Note: Could not set CPU governors: {e}")
                
                try:
                    # Set process priority
                    subprocess.run(['sudo', 'renice', '-n', '-10', ollama_pid], check=True)
                    print("Set Ollama process priority")
                    
                    # Set IO priority
                    subprocess.run(['sudo', 'ionice', '-c', '1', '-n', '0', '-p', ollama_pid], check=True)
                    print("Set IO priority")
                except Exception as e:
                    print(f"Note: Could not set process priorities: {e}")
                
                try:
                    # Use cores 2-3 for Ollama
                    subprocess.run(['sudo', 'taskset', '-p', '-c', '2-3', ollama_pid], check=True)
                    print("Set CPU affinity to cores 2-3")
                except Exception as e:
                    print(f"Note: Could not set CPU affinity: {e}")
                
                # Optimize memory
                try:
                    # Drop caches
                    subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=1'], check=True)
                    # Reduce swapping
                    subprocess.run(['sudo', 'sysctl', 'vm.swappiness=1'], check=True)
                    # Increase cache pressure
                    subprocess.run(['sudo', 'sysctl', 'vm.vfs_cache_pressure=200'], check=True)
                    print("Optimized memory settings")
                except Exception as e:
                    print(f"Note: Could not optimize memory: {e}")
                
        except Exception as e:
            print(f"Note: CPU optimization skipped: {e}")

    def _install_ollama_models(self) -> None:
        """Show available models from Ollama library and install selected ones"""
        # Models suitable for Raspberry Pi
        available_models = {
            'tinyllama': ("TinyLlama", "~1.1GB", "Extremely lightweight model, perfect for Raspberry Pi"),
            'gemma:2b': ("Gemma 2B", "~1.5GB", "Google's lightweight model, good for Raspberry Pi"),
            'phi:2.7b': ("Phi 2.7B", "~1.7GB", "Microsoft's efficient small model"),
            'stable-code:3b': ("StableCode 3B", "~2GB", "Optimized for code generation"),
            'neural-chat:7b-v3-q4': ("Neural Chat 7B Q4", "~4GB", "Optimized chat model"),
            'mistral:7b-q4': ("Mistral 7B Q4", "~4GB", "Strong general purpose model"),
            'codellama:7b-q4': ("CodeLlama 7B Q4", "~4GB", "Specialized for coding tasks"),
            'llama2:7b-q4': ("Llama 2 7B Q4", "~4GB", "Meta's general purpose model")
        }
        
        print("\nAvailable models from Ollama library:")
        headers = ["#", "Model", "Size", "Description"]
        rows = []
        for i, (model_id, (name, size, desc)) in enumerate(available_models.items(), 1):
            rows.append([str(i), name, size, desc])
        
        print(self._create_ascii_table(headers, rows))
        
        while True:
            choice = input("\nEnter model number to install (or 'done' to finish): ").strip()
            if choice.lower() == 'done':
                break
                
            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                model_id = list(available_models.keys())[int(choice)-1]
                name, size, _ = available_models[model_id]
                print(f"\nPulling {name} ({size})...")
                
                try:
                    process = subprocess.Popen(
                        ['ollama', 'pull', model_id],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Show progress in real-time
                    while True:
                        char = process.stdout.read(1)
                        if char == '' and process.poll() is not None:
                            break
                        print(char, end='', flush=True)
                    
                    if process.returncode == 0:
                        print(f"\n{name} successfully installed!")
                    else:
                        print(f"\nError installing {name}")
                        
                except Exception as e:
                    print(f"\nError pulling model: {e}")
            else:
                print("Invalid selection. Please try again.")

    def _fetch_relevant_online_info(self, query: str) -> str:
        """Let the model find relevant information based on the query"""
        return """
Please research and provide information about:
- Official documentation for suggested solutions
- ARM architecture compatibility
- Resource requirements
- Community experiences
- Best practices for Raspberry Pi
"""

    def _handle_api_credentials(self, env_var: str) -> None:
        """Handle API credentials with encryption"""
        from cryptography.fernet import Fernet
        import base64
        import getpass
        
        encrypted_file = '.env.encrypted'
        
        # Check for existing encrypted file
        if os.path.exists(encrypted_file):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Get password from user
                    password = getpass.getpass("Enter password to decrypt API keys: ")
                    key = base64.b64encode(password.encode().ljust(32)[:32])
                    f = Fernet(key)
                    
                    # Decrypt file
                    with open(encrypted_file, 'rb') as file:
                        encrypted_data = file.read()
                        decrypted_data = f.decrypt(encrypted_data)
                        
                    # Parse decrypted data and check for requested API key
                    api_key_found = False
                    for line in decrypted_data.decode().split('\n'):
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
                            if key == env_var:
                                self.api_key = value
                                api_key_found = True
                    
                    if api_key_found:
                        print(f"{env_var} loaded successfully!")
                        return
                    else:
                        print(f"\nNo {env_var} found in encrypted credentials.")
                        break
                    
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print("Incorrect password. Please try again.")
                    else:
                        print(f"Error decrypting credentials after {max_attempts} attempts.")
        
        # If we get here, either no encrypted file exists, decryption failed, or key not found
        self.api_key = os.getenv(env_var)
        
        if not self.api_key:
            print(f"\nNo {env_var} found.")
            if input("Would you like to enter an API key now? (y/n): ").lower() == 'y':
                api_key = getpass.getpass("Enter your API key: ").strip()
                self._save_encrypted_credentials({env_var: api_key})
                self.api_key = api_key
            else:
                print("\nNo API key provided. Please choose a different model type.")
                self._handle_model_selection()

    def _save_encrypted_credentials(self, credentials: Dict) -> None:
        """Save encrypted credentials to file"""
        try:
            # Get password for encryption
            while True:
                password = getpass.getpass("\nCreate a password to encrypt your API key: ")
                confirm = getpass.getpass("Confirm password: ")
                if password == confirm:
                    break
            
            # Create encryption key from password
            key = base64.b64encode(password.encode().ljust(32)[:32])
            f = Fernet(key)
            
            # Format credentials as env file content
            env_content = '\n'.join(f"{k}={v}" for k, v in credentials.items())
            
            # Encrypt the content
            encrypted_data = f.encrypt(env_content.encode())
            
            # Save to encrypted file
            with open('.env.encrypted', 'wb') as file:
                file.write(encrypted_data)
            
            print("\nAPI key encrypted and saved successfully!")
            print("Your key will be automatically loaded next time using your password.")
            
            # Update current environment
            for key, value in credentials.items():
                os.environ[key] = value
            
        except Exception as e:
            print(f"\nError saving credentials: {e}")
            print("Please try again or contact support.")

    def _handle_model_selection(self):
        """Handle model type selection"""
        print("\nAvailable Model Types:")
        print("1. Claude (Anthropic, requires API key)")
        print("2. Gemini (Google, requires API key)")
        print("3. Local Models (via Ollama, runs on your Pi)")
        
        while True:
            choice = input("\nSelect model type (1/2/3): ").strip()
            if choice == "1":
                self.model_provider = "anthropic"
                self._handle_api_credentials("ANTHROPIC_API_KEY")
                break
            elif choice == "2":
                self.model_provider = "google"
                self._handle_api_credentials("GOOGLE_API_KEY")
                break
            elif choice == "3":
                self.model_provider = "local"
                self.api_key = None
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    console = ConsoleManager()
    assistant = RaspberryPiAssistant()
    
    # Show welcome message in output area
    console.print_output("Welcome to Raspberry Pi Assistant!\n"
                        f"Detected System: {assistant.pi_model}\n"
                        f"OS: {assistant.os_info.get('PRETTY_NAME', 'Unknown')}\n"
                        f"Model: {assistant.model}\n\n"
                        "Type 'New Topic' to start a fresh conversation.")
    
    while True:
        try:
            user_input = console.get_input("\nYou ('New Topic' to reset context): ")
            
            if user_input.lower() == 'exit':
                console.print_output("\nGoodbye!")
                break
            
            if user_input.lower() == 'new topic':
                # Clear conversation history
                assistant.conversation_history = []
                # Clear output buffer
                console.output_buffer = []
                # Clear prompt history
                console.prompt_history = []
                # Clear screen and show fresh welcome
                console.print_output("Starting fresh conversation...")
                continue
            
            # Create follow-up prompt if this isn't the first question
            if assistant.conversation_history:
                context = assistant._build_rag_context(user_input)
                analysis_prompt = f"""This is a follow-up question in our conversation about {context}.
Based on the previous context and this new query, provide ONLY the command needed.
Use these safe commands: cat, ls, df, free, uname, vcgencmd, systemctl status

Query: {user_input}
"""
            else:
                analysis_prompt = None  # Let process_user_query create the initial prompt
            
            response = assistant.process_user_query(user_input, console, analysis_prompt)
            formatted_response = assistant.format_response(response, [])
            console.print_output(f"\nAssistant: {formatted_response}")
            
        except KeyboardInterrupt:
            console.print_output("\nGoodbye!")
            break

if __name__ == "__main__":
    main() 