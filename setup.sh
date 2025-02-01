pip install torch torchvision ftfy regex tqdm opencv-python scikit-learn pillow
pip install git+https://github.com/openai/CLIP.git
#!/bin/bash

# # Exit script on error
# set -e

# # Print out each command
# set -x

# # Define Python version (adjust if needed)
# PYTHON_VERSION="3.9"

# # Check if virtualenv is already installed
# if ! command -v virtualenv &> /dev/null
# then
#     echo "virtualenv could not be found, installing..."
#     pip install --upgrade pip
#     pip install virtualenv
# fi

# # Create virtual environment if it doesn't exist
# if [ ! -d "venv" ]; then
#     echo "Creating virtual environment..."
#     virtualenv venv -p python${PYTHON_VERSION}
# fi

# # Activate virtual environment
# source venv/bin/activate

# # Install required dependencies
# echo "Installing dependencies..."
# pip install --upgrade pip
# pip install -r requirements.txt

# If you don't have a requirements.txt, here's the manual installation list:
pip install numpy
pip install opencv-python
# pip install tqdm
pip install torch
# pip install clip-by-openai
pip install scikit-learn
pip install hdbscan
pip install rawpy
pip install pillow
pip install pyyaml

echo "Setup completed successfully!"
