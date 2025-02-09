#!/bin/bash

# Activate virtual environment
source pi_assistant_env/bin/activate

# Run the assistant
python3 pi_assistant.py

# Deactivate virtual environment when done
deactivate 