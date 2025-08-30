import os
from huggingface_hub import hf_hub_download

repo_id = "foduucom/stockmarket-pattern-detection-yolov8"
filename = "model.pt"

# Define the local path where the model should be saved
# This assumes the script is run from the trading-chart-patterns directory
local_dir = os.path.abspath(os.path.dirname(__file__))

print(f"Attempting to download {filename} from {repo_id} to {local_dir}...")

try:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print(f"Model downloaded successfully to: {model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
