"""
Raspberry Pi AI Assistant
Author: Frank Currie (frank@sfle.ca)
Last Updated: February 9, 2025 22:41 EST
License: MIT

Copyright (c) 2025 Frank Currie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# System prompts
SYSTEM_PROMPTS = {
    "base_context": """
    You are a Raspberry Pi assistant helping users with their ARM-based Raspberry Pi system.
    You should provide accurate information about:
    - Package management (apt)
    - Hardware configuration
    - System services
    - Common troubleshooting
    Always consider the ARM architecture when suggesting packages or solutions.
    """
} 