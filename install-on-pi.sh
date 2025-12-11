sudo apt update
sudo apt upgrade -y

#OpenCV dependencies
sudo apt install ffmpeg libsm6 libxext6 -y

#General python dependencies
sudo apt install gcc python3-dev -y

# Install python3-venv
sudo apt install python3-venv -y

# Parse arguments
INSTALL_CPU=true
for arg in "$@"
do
    if [ "$arg" == "--gpu" ]; then
        INSTALL_CPU=false
    fi
done

#Install virtual env in an user independent location
sudo mkdir -p /opt/dusty
sudo chown -R $(whoami):$(whoami) /opt/dusty/
cd /opt/dusty/
python3 -m venv env

#activate the virtual env
source /opt/dusty/env/bin/activate

# Install CPU-specific torch by default
# Run this BEFORE installing other deps so pip finds the cpu version first
if [ "$INSTALL_CPU" = true ]; then
    echo "Installing PyTorch CPU version (default)..."
    echo "Pass --gpu to install standard version with CUDA support."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Skipping CPU-specific torch install (GPU mode requested)..."
fi

# Clone repo and install dependencies
cd ~/
# Check if directory exists before cloning to avoid error if re-running
if [ ! -d "dustycam" ]; then
    git clone git@github.com:owlmoshpit/dustycam.git
fi
cd dustycam
# Install base + Pi deps
pip install -e ".[pi]"
