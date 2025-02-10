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
        
        # Trim buffer to fit output area
        while len(self.output_buffer) > self.output_height:
            self.output_buffer.pop(0)
        
        # Print buffer
        print("\033[H", end="")
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
            print(f"\033[{start_pos + i};0H\033[K\033[31m{prompt}\033[0m")

    def get_input(self, prompt: str = "You: ") -> str:
        """Get input from the bottom section"""
        # Show history first
        self.print_input_history()
        
        # Move cursor to input position and get new input
        input_pos = self.height - 1  # Bottom line
        print(f"\033[{input_pos};0H\033[K\033[31m{prompt}", end="", flush=True)
        user_input = input()
        print("\033[0m", end="")  # Reset color
        
        # Add to history (full prompt + input)
        if user_input.strip():  # Only add non-empty inputs
            full_prompt = f"{prompt}{user_input}"
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
        self.api_key = None  # Initialize api_key attribute
        
        print("\nWelcome to Raspberry Pi Assistant!")
        print(f"Detected System: {self.pi_model}")
        print(f"OS: {self.os_info.get('PRETTY_NAME', 'Unknown')}")
        
        # First ask user preference for model type
        print("\nAvailable Model Types:")
        print("1. Cloud Models (Claude, requires API key)")
        print("2. Local Models (via Ollama, runs on your Pi)")
        
        while True:
            choice = input("\nSelect model type (1/2): ").strip()
            if choice == "1":  # Cloud Models
                self._handle_api_credentials()  # This will update self.api_key
                if not self.api_key:
                    print("\nNo API key found.")
                    if input("Would you like to enter an API key now? (y/n): ").lower() == 'y':
                        api_key = getpass.getpass("Enter your API key: ").strip()
                        self._save_encrypted_credentials({'ANTHROPIC_API_KEY': api_key})
                        self.api_key = api_key
                    else:
                        print("No API key provided. Switching to local models.")
                        choice = "2"
                break
            elif choice == "2":  # Local Models
                self.api_key = None
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        # Initialize based on choice
        self.ollama_available = False
        if choice == "2":  # Local Models
            print("\nChecking Ollama installation...")
            binary_exists, service_exists = self._check_ollama_installed()
            
            if binary_exists:
                print("\nOllama is already installed.")
                if self._check_ollama_service():
                    self.ollama_available = True
                    # Show installed models
                    installed_models = self._get_installed_ollama_models()
                    if installed_models:
                        print("\nInstalled Ollama models:")
                        for model in installed_models:
                            print(f"- {model['name']} ({model['size']})")
                    
                    # Ask about installing additional models
                    print("\nWould you like to install additional models from Ollama library? (y/n): ", end='')
                    if input().lower() == 'y':
                        self._install_ollama_models()
            else:
                print("\nOllama is not installed. Would you like to install it? (y/n): ", end='')
                if input().lower() == 'y':
                    if self._install_and_start_ollama():
                        self.ollama_available = True
                        self._install_ollama_models()
        
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
        is_root = os.geteuid() == 0
        return f"""You are a helpful assistant for Raspberry Pi users. Current system information:
- Device: {self.pi_model}
- OS: {self.os_info.get('PRETTY_NAME', 'Unknown')}
- Architecture: ARM
- Model: {self.model}
- Running as root: {is_root}

For informative queries:
1. Provide a single-line response using the command output
2. Format: "The [item] is [value from command]"
3. For comparison queries (latest version, updates available, etc.):
   - Check the Raspberry Pi OS documentation
   - Compare with current system values
   - Provide clear yes/no answers with version numbers
4. For multiple values, use bullet points

For action queries:
1. List each command on a single line with a short description:
   ```command``` - What this command does
2. Keep explanations concise

The system uses apt for package management and systemctl for service control."""

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
        # Calculate column widths
        widths = []
        for i in range(len(headers)):
            column = [str(row[i]) for row in rows]
            widths.append(max(len(str(x)) for x in [headers[i]] + column))
        
        # Create the table
        separator = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
        result = [separator]
        
        # Add headers
        header = '|' + '|'.join(f' {h:<{w}} ' for h, w in zip(headers, widths)) + '|'
        result.extend([header, separator])
        
        # Add rows
        for row in rows:
            result.append('|' + '|'.join(f' {str(c):<{w}} ' for c, w in zip(row, widths)) + '|')
        
        result.append(separator)
        return '\n'.join(result)

    def _get_available_model(self) -> Tuple[str, bool]:
        """Get available models and let user choose"""
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
        
        # Show model details on selection
        while True:
            try:
                choice = input("\nSelect a model number (or press Enter for default): ").strip()
                if not choice:
                    model, is_local, model_type, _, _ = models[0]
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(models):
                        model, is_local, model_type, _, _ = models[choice_idx]
                    else:
                        print("Invalid selection. Please try again.")
                        continue
                
                # Handle model initialization based on type
                if is_local and model_type == 'ollama':
                    try:
                        from ollama import Client as OllamaClient
                        self.client = OllamaClient()
                        return model, is_local
                    except Exception as e:
                        print(f"Error initializing Ollama model: {e}")
                        print("Please try another model.")
                        continue
                else:  # Cloud model (Claude)
                    self.client = Anthropic(api_key=self.api_key)
                    return model, is_local
                
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                print(f"Error initializing model: {e}")
                print("Please try another model.")

    def _make_api_request(self, context: str, retries: int = 0) -> str:
        """Make API request with retry logic"""
        try:
            if self.is_local and hasattr(self.client, 'chat'):  # Ollama models
                response = self.client.chat(
                    model=self.model,
                    messages=[{
                        'role': 'system',
                        'content': self.system_context
                    }, {
                        'role': 'user',
                        'content': context
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
                        {"role": "user", "content": context}
                    ]
                )
                return message.content[0].text
                
        except Exception as e:
            return f"Error communicating with API: {str(e)}"

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

    def process_user_query(self, query: str, console: ConsoleManager) -> str:
        """Process user query using RAG approach"""
        console.print_output("\n=== Building Context ===")
        context = self._build_rag_context(query)
        console.print_output("- Gathered system information")
        console.print_output("- Added conversation history")
        console.print_output("- Formatted query context")
        
        console.print_output("\n=== Analyzing Query ===")
        
        # Create analysis prompt
        analysis_prompt = f"""Based on this query and system context, provide a helpful response.
For system information queries:
1. If the query can be answered with a command, ONLY provide the command in a code block
2. DO NOT provide explanations unless specifically asked
3. Use only safe, read-only commands like: cat, ls, df, free, uname, vcgencmd

Context:
{context}

Query:
{query}
"""
        
        analysis = self._make_api_request(analysis_prompt)
        
        # Extract and execute safe informative commands
        command_results = []
        if '```' in analysis:
            commands = analysis.split('```')[1::2]  # Get all commands between backticks
            
            # Check if these are informative (read-only) commands
            safe_commands = ['cat', 'ls', 'df', 'free', 'uname', 'vcgencmd']
            is_informative = all(any(cmd.strip().startswith(safe) for safe in safe_commands) 
                               for cmd in commands)
            
            if is_informative:
                for command in commands:
                    command = command.strip()
                    success, result = self.execute_command(command)
                    if success:
                        # For temperature queries, format the result nicely
                        if 'temp' in command:
                            temp = int(result.strip()) / 1000
                            command_results.append(f"The current CPU temperature is {temp:.1f}°C")
                        else:
                            command_results.append(result.strip())
        
        # Return just the command output if available, otherwise the full response
        if command_results:
            return "\n".join(command_results)
        else:
            return analysis

    def execute_with_confirmation(self, command: str) -> Tuple[bool, str]:
        """Execute command with confirmation and real-time output"""
        print(f"\nAbout to execute: {command}")
        if input("Do you want to proceed? (y/n): ").lower() == 'y':
            return self.execute_command(command)
        return False, "Command execution cancelled by user"

    def _warmup_model(self) -> None:
        """Send a simple request to warm up the model"""
        warmup_text = "Hello! Please respond with 'Ready!'"
        try:
            if self.is_local:
                if hasattr(self.client, 'chat'):  # Ollama models
                    # Set model parameters for lower memory usage
                    response = self.client.chat(
                        model=self.model,
                        messages=[{
                            'role': 'user',
                            'content': warmup_text
                        }],
                        options={
                            'num_gpu': 0,  # CPU only
                            'num_thread': 4,  # Limit threads
                            'num_ctx': 512,  # Smaller context window
                            'seed': 42
                        }
                    )
                    if response:
                        print("Ollama model is ready!")
            else:  # Cloud models (Claude)
                response = self._make_api_request(warmup_text)
                if response:
                    print("Claude is ready!")
                else:
                    print("There might be an issue with the API connection...")
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Please try restarting the assistant.")

    def format_response(self, response: str, command_results: List[str] = None) -> str:
        """Format the response in a clear, structured way"""
        formatted = ""
        
        if command_results:
            # For informative queries, just show the command output
            if command_results[0].startswith('$'):
                result = command_results[0].split('\n')[1].strip()
                # Store the full kernel version in conversation history
                if 'uname' in command_results[0]:
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"kernel_version={result}"
                    })
                formatted = f"\nThe kernel version is {result}\n"
            else:
                formatted += "\n=== Command Results ===\n"
                for result in command_results:
                    formatted += f"{result}\n"
        else:
            formatted += "\n=== Response ===\n"
            formatted += response
        
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

    def _handle_api_credentials(self) -> None:
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
                        
                    # Parse decrypted data
                    for line in decrypted_data.decode().split('\n'):
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
                            if key == 'ANTHROPIC_API_KEY':
                                self.api_key = value
                    
                    print("API key loaded successfully!")
                    return
                    
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print("Incorrect password. Please try again.")
                    else:
                        print(f"Error decrypting credentials after {max_attempts} attempts.")
        
        # If we get here, either no encrypted file exists or decryption failed
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            print("\nNo API key found.")
            if input("Would you like to enter an API key now? (y/n): ").lower() == 'y':
                api_key = getpass.getpass("Enter your API key: ").strip()
                self._save_encrypted_credentials({'ANTHROPIC_API_KEY': api_key})
                self.api_key = api_key

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

def main():
    console = ConsoleManager()
    assistant = RaspberryPiAssistant()
    
    # Show welcome message in output area
    console.print_output("Welcome to Raspberry Pi Assistant!\n"
                        f"Detected System: {assistant.pi_model}\n"
                        f"OS: {assistant.os_info.get('PRETTY_NAME', 'Unknown')}\n"
                        f"Model: {assistant.model}")
    
    while True:
        try:
            user_input = console.get_input("\nYou: ")
            
            if user_input.lower() == 'exit':
                console.print_output("\nGoodbye!")
                break
            
            if user_input.lower() == 'next question':
                assistant.conversation_history = []
                console.print_output("\nStarting new chat...")
                continue
            
            response = assistant.process_user_query(user_input, console)
            formatted_response = assistant.format_response(response, [])
            console.print_output(f"\nAssistant: {formatted_response}")
            
        except KeyboardInterrupt:
            console.print_output("\nGoodbye!")
            break

if __name__ == "__main__":
    main() 