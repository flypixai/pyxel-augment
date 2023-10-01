#!/bin/bash

# Function to create a model directory
create_model_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Function to download a specific model
download_specific_model() {
    local model_name="$1"
    local download_url="$2"

    create_model_directory "models"
    model_path="$model_directory/$model_name"

    if [ -f "$model_path" ]; then
        echo "Model '$model_name' already exists in '$model_directory'. Skipping download."
    else
        echo "Downloading '$model_name'..."
        cd "models"
        wget "$download_url"
        echo "Download completed."
        cd ..
    fi
}

# Check if the number of arguments is less than 2
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

model_name="$1"

# Define model-specific download links (add more as needed)
case "$model_name" in
    "GSAM")
        download_url=("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        ;;
    "SEEM")
        download_url=("https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt")
        ;;
    *)  # Default case for unknown models
        echo "Unknown model name: $model_name"
        exit 1
        ;;
esac

# Directory name will be the same as the model name
model_directory="$model_name"

for item in "${download_url[@]}"; do
    download_specific_model "$model_name" "$download_url" "$model_directory"
done




