#!/bin/bash

create_models_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

download_specific_model() {
    local model_name="$1"
    local download_url="$2"

    model_path="$model_directory/$model_name"

    if [ -f "$model_path" ]; then
        echo "Model '$model_name' already exists in '$model_directory'. Skipping download."
    else
        echo "Downloading '$model_name'..."
        wget -P "$model_path" "$download_url"
        echo "Download completed."
    fi
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

model_name="$1"

case "$model_name" in
    "GSAM")
        download_url=("https://raw.githubusercontent.com/IDEA-Research/Grounded-Segment-Anything/main/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        ;;
    "SEEM")
        download_url=("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"
                    "https://raw.githubusercontent.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/main/demo_code/configs/seem/seem_focall_lang.yaml"
        )
        ;;
    *)  
        echo "Unknown model name: $model_name"
        exit 1
        ;;
esac

model_directory="models"
create_models_directory "models"

for item in "${download_url[@]}"; do
    download_specific_model "$model_name" "$item" "$model_directory"
done




