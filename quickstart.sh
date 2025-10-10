#!/bin/bash

# Quick Start Script for Tiny Recursive Models - AES
# Optimized for MacBook Pro M1 with 16GB RAM

set -e  # Exit on error

echo "=========================================="
echo "Tiny Recursive Models - AES Quick Start"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check if PyTorch MPS is available
echo ""
echo "Checking PyTorch MPS availability..."
python3 -c "import torch; print('MPS Available:', torch.backends.mps.is_available()); print('MPS Built:', torch.backends.mps.is_built())"

# Check for HuggingFace authentication
echo ""
echo "Checking HuggingFace authentication..."
if python3 -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); exit(0 if token else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ HuggingFace token found${NC}"
else
    echo -e "${YELLOW}⚠ HuggingFace token not found${NC}"
    echo "You need to login to HuggingFace to download datasets"
    echo "Run: huggingface-cli login"
    read -p "Do you want to login now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "Please login later with: huggingface-cli login"
    fi
fi

# Ask which prompt set to use
echo ""
echo "=========================================="
echo "Dataset Preparation"
echo "=========================================="
echo ""
echo "Available prompt sets:"
echo "  1) Prompts 1-2 (Score range: 2-12)"
echo "  2) Prompts 3-6 (Score range: 0-4)"
echo "  3) Prompt 7 (Score range: 2-24)"
echo "  4) All prompts"
echo ""
read -p "Select prompt set to prepare [1-4] (default: 1): " prompt_choice
prompt_choice=${prompt_choice:-1}

case $prompt_choice in
    1)
        PROMPT_SET="1-2"
        DATA_DIR="data/asappp_prompts_1-2"
        ;;
    2)
        PROMPT_SET="3-6"
        DATA_DIR="data/asappp_prompts_3-6"
        ;;
    3)
        PROMPT_SET="7"
        DATA_DIR="data/asappp_prompts_7"
        ;;
    4)
        PROMPT_SET="all"
        DATA_DIR="data/asappp"
        ;;
    *)
        echo -e "${RED}Invalid choice. Using Prompts 1-2${NC}"
        PROMPT_SET="1-2"
        DATA_DIR="data/asappp_prompts_1-2"
        ;;
esac

# Check if dataset already exists
if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/test" ]; then
    echo -e "${YELLOW}Dataset already exists at $DATA_DIR${NC}"
    read -p "Do you want to rebuild it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping dataset preparation..."
    else
        echo ""
        echo "Building dataset for prompt set: $PROMPT_SET"
        python3 dataset/build_asappp_dataset.py \
            --prompt-set "$PROMPT_SET" \
            --output-dir data/asappp \
            --num-aug 1
        echo -e "${GREEN}✓ Dataset built successfully${NC}"
    fi
else
    echo ""
    echo "Building dataset for prompt set: $PROMPT_SET"
    echo "This may take a few minutes..."
    python3 dataset/build_asappp_dataset.py \
        --prompt-set "$PROMPT_SET" \
        --output-dir data/asappp \
        --num-aug 1
    echo -e "${GREEN}✓ Dataset built successfully${NC}"
fi

# Use the correct data directory for training
if [ "$PROMPT_SET" == "all" ]; then
    echo ""
    echo -e "${YELLOW}Note: 'all' option builds multiple datasets.${NC}"
    echo "For training, please specify which one to use:"
    echo "  - data/asappp_prompts_1-2"
    echo "  - data/asappp_prompts_3-6"
    echo "  - data/asappp_prompts_7"
    read -p "Enter data directory for training [1-2]: " train_choice
    train_choice=${train_choice:-"1-2"}
    DATA_DIR="data/asappp_prompts_$train_choice"
fi

# Ask about training
echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo ""
read -p "Do you want to start training now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Setup complete! To train later, run:"
    echo ""
    echo "  source venv/bin/activate"
    echo "  python3 train_aes_m1.py --data-path $DATA_DIR --batch-size 16 --epochs 5000"
    echo ""
    exit 0
fi

# Training parameters
echo ""
echo "Configure training parameters (press Enter for defaults):"
echo ""
read -p "Batch size [16]: " batch_size
batch_size=${batch_size:-16}

read -p "Number of epochs [5000]: " epochs
epochs=${epochs:-5000}

read -p "Learning rate [3e-4]: " lr
lr=${lr:-3e-4}

read -p "Evaluation interval (epochs) [250]: " eval_interval
eval_interval=${eval_interval:-250}

read -p "Use Weights & Biases logging? (y/n) [n]: " use_wandb
use_wandb=${use_wandb:-n}

# Build command
CMD="python3 train_aes_m1.py \
    --data-path $DATA_DIR \
    --batch-size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --eval-interval $eval_interval \
    --d-model 128 \
    --d-hidden 256 \
    --n-heads 4 \
    --n-layers 2 \
    --h-cycles 2 \
    --l-cycles 3 \
    --checkpoint-path checkpoints/$(basename $DATA_DIR) \
    --seed 42"

if [[ $use_wandb =~ ^[Yy]$ ]]; then
    echo ""
    echo "Logging in to Weights & Biases..."
    wandb login || true
    CMD="$CMD --use-wandb --project-name TinyRecursiveModels-AES"
fi

# Start training
echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""
echo "Command: $CMD"
echo ""
echo "Training will start in 3 seconds..."
echo "Press Ctrl+C to cancel"
sleep 3

eval $CMD

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: checkpoints/$(basename $DATA_DIR)"
echo ""
echo "To resume training or evaluate, run:"
echo "  source venv/bin/activate"
echo "  python3 train_aes_m1.py --data-path $DATA_DIR [options]"
echo ""
