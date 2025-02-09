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
                
                # Add latest models with descriptions
                descriptions = {
                    'opus': "Most capable model for complex reasoning and analysis",
                    'sonnet': "Balanced model for general use and coding tasks",
                    'haiku': "Fast, efficient model for simpler tasks"
                }
                
                for variant, model in latest_models.items():
                    if model:
                        models.append((model.id, False, 'claude', descriptions[variant]))
                        
            except Exception as e:
                print(f"Error getting Claude models: {e}")
        
        # Add installed Ollama models
        if self.ollama_available:
            try:
                process = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, text=True)
                if process.returncode == 0:
                    for line in process.stdout.strip().split('\n')[1:]:  # Skip header
                        if line:
                            parts = line.split()
                            if len(parts) >= 3:
                                name = parts[0]
                                size = parts[2]
                                description = self._get_ollama_model_description(name)
                                models.append((name, True, 'ollama', description))
            except Exception as e:
                print(f"Error getting Ollama models: {e}")
        
        if not models:
            print("No models available. Please install a local model or provide an API key.")
            exit(1)
        
        print("\nAvailable models:")
        
        # Prepare table data
        headers = ["#", "Model", "Type", "Size", "Description"]
        rows = []
        
        for i, (model, is_local, model_type, description) in enumerate(models, 1):
            if is_local and model_type == 'ollama':
                rows.append([
                    str(i),
                    model,
                    "Local (Ollama)",
                    size,
                    description
                ])
            else:  # Cloud models (Claude)
                rows.append([
                    str(i),
                    model,
                    "Cloud",
                    "N/A",
                    description
                ])
        
        print(self._create_ascii_table(headers, rows))
        
        # Show model details on selection
        while True:
            try:
                choice = input("\nSelect a model number (or press Enter for default): ").strip()
                if not choice:
                    model, is_local, model_type, _ = models[0]
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(models):
                        model, is_local, model_type, _ = models[choice_idx]
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

    def process_user_query(self, query: str) -> str:
        """Process user query using RAG approach"""
        print("\n=== Building Context ===")
        context = self._build_rag_context(query)
        print("- Gathered system information")
        print("- Added conversation history")
        print("- Formatted query context")
        
        print("\n=== Analyzing Query ===")
        analysis_prompt = f"""Based on this context and query, determine:
1. Is this a simple factual question that can be answered directly?
2. Are there ambiguities or missing details that need clarification?
3. Could multiple valid approaches exist based on unstated requirements?

If the query is simple and clear:
- Provide a direct, concise answer
- Include relevant facts from system context
- No need for clarifying questions

If the query is ambiguous or complex:
- List 2-3 essential clarifying questions
- Explain why these details matter
- Wait for user responses before proceeding

Context:
{context}"""

        print("- Sending analysis prompt to model...")
        analysis = self._make_api_request(analysis_prompt)
        
        # Check if clarifying questions are needed
        if '?' in analysis and any(trigger in query.lower() for trigger in ['how', 'which', 'what should', 'recommend', 'suggest']):
            print("\n=== Clarifying Questions ===")
            questions = [q.strip() for q in analysis.split('\n') if '?' in q]
            answers = {}
            for q in questions:
                answer = input(f"\n{q}\n> ")
                answers[q] = answer
            
            # Get response with user's answers
            print("\n=== Generating Informed Response ===")
            response_prompt = f"""Based on user's responses:
{answers}

And the original context:
{context}

Provide a tailored response that:
1. Addresses their specific needs
2. Explains relevant options
3. Makes appropriate recommendations
4. Asks if they'd like to proceed with any specific solution

DO NOT suggest installations yet - wait for user to choose a direction."""
            
            response = self._make_api_request(response_prompt)
        else:
            # Direct response for simple queries
            response = analysis

        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "context": context
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response

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
        
        env_file = '.env'
        encrypted_file = '.env.encrypted'
        
        # Check for existing encrypted file
        if os.path.exists(encrypted_file):
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
                
                return
            except Exception as e:
                print(f"Error decrypting credentials: {e}")
        
        # If no encrypted file or decryption failed
        if os.path.exists(env_file):
            # Ask if user wants to encrypt existing .env
            if input("\nWould you like to encrypt your .env file? (y/n): ").lower() == 'y':
                try:
                    # Get new password
                    while True:
                        password = getpass.getpass("Enter new password for encryption: ")
                        confirm = getpass.getpass("Confirm password: ")
                        if password == confirm:
                            break
                        print("Passwords don't match, try again")
                    
                    key = base64.b64encode(password.encode().ljust(32)[:32])
                    f = Fernet(key)
                    
                    # Read and encrypt .env contents
                    with open(env_file, 'rb') as file:
                        data = file.read()
                        encrypted_data = f.encrypt(data)
                    
                    # Save encrypted file
                    with open(encrypted_file, 'wb') as file:
                        file.write(encrypted_data)
                    
                    # Optionally delete original .env
                    if input("Delete original .env file? (y/n): ").lower() == 'y':
                        os.remove(env_file)
                    
                    print(f"\nCredentials encrypted and saved to {encrypted_file}")
                    
                except Exception as e:
                    print(f"Error encrypting credentials: {e}")
        
        # Load API key from environment if available
        self.api_key = os.getenv('ANTHROPIC_API_KEY')

    def _save_encrypted_credentials(self, credentials: Dict) -> None:
        """Save encrypted credentials to file"""
        from cryptography.fernet import Fernet
        import base64
        
        env_file = '.env'
        encrypted_file = '.env.encrypted'
        
        # Check for existing encrypted file
        if os.path.exists(encrypted_file):
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
                
                return
            except Exception as e:
                print(f"Error decrypting credentials: {e}")
        
        # If no encrypted file or decryption failed
        if os.path.exists(env_file):
            # Ask if user wants to encrypt existing .env
            if input("\nWould you like to encrypt your .env file? (y/n): ").lower() == 'y':
                try:
                    # Get new password
                    while True:
                        password = getpass.getpass("Enter new password for encryption: ")
                        confirm = getpass.getpass("Confirm password: ")
                        if password == confirm:
                            break
                        print("Passwords don't match, try again")
                    
                    key = base64.b64encode(password.encode().ljust(32)[:32])
                    f = Fernet(key)
                    
                    # Read and encrypt .env contents
                    with open(env_file, 'rb') as file:
                        data = file.read()
                        encrypted_data = f.encrypt(data)
                    
                    # Save encrypted file
                    with open(encrypted_file, 'wb') as file:
                        file.write(encrypted_data)
                    
                    # Optionally delete original .env
                    if input("Delete original .env file? (y/n): ").lower() == 'y':
                        os.remove(env_file)
                    
                    print(f"\nCredentials encrypted and saved to {encrypted_file}")
                    
                except Exception as e:
                    print(f"Error encrypting credentials: {e}")
        
        # Save credentials to environment
        for key, value in credentials.items():
            os.environ[key] = value

def main():
    assistant = RaspberryPiAssistant()
    
    print("Welcome to Raspberry Pi Assistant!")
    print(f"Detected System: {assistant.pi_model}")
    print(f"OS: {assistant.os_info.get('PRETTY_NAME', 'Unknown')}")
    print(f"Model: {assistant.model}")
    
    while True:
        try:
            user_input = input("\nYou (ask a follow-up or say \"Next Question\" to start fresh): ").strip()
            
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'next question':
                assistant.conversation_history = []
                print("\nStarting new chat...")
                continue
            
            response = assistant.process_user_query(user_input)
            command_results = []
            
            # Handle command execution if response contains commands
            if '```' in response:
                commands = response.split('```')[1::2]  # Get all commands
                
                # Check if this is an informative query (read-only commands)
                is_informative = all(cmd.strip().startswith(('uname', 'cat', 'ls', 'df', 'free', 'dpkg', 'apt list')) 
                                   for cmd in commands)
                
                if is_informative:
                    # Automatically execute read-only commands
                    for command in commands:
                        command = command.strip()
                        success, result = assistant.execute_command(command)
                        if success:
                            command_results.append(f"$ {command}\n{result}")
                else:
                    # Interactive mode for system-modifying commands
                    for command in commands:
                        command = command.strip()
                        command = assistant._check_sudo_needed(command)
                        if command.startswith(('sudo', 'apt-get', 'systemctl')):
                            execute = input(f"\nWould you like to execute this command?\n{command}\n(y/n): ").lower()
                            if execute == 'y':
                                success, result = assistant.execute_with_confirmation(command)
                                command_results.append(f"Command: {command}\nResult: {result}")
                                if not success:
                                    print("Would you like help troubleshooting this error?")
            
            # Format and display response
            formatted_response = assistant.format_response(response, command_results)
            print("\nAssistant:", formatted_response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main() 