# scripts/upload_to_hf.py

from huggingface_hub import HfApi, HfFolder, upload_folder
import os

# ------------------------------
# CONFIGURATION
# ------------------------------
HF_REPO_ID = "afsanehm/fake-news-detection-llm"
MODEL_PATH = os.path.join(os.path.dirname(__file__), \
             "..", "model", "fine_tuned_model")

# ------------------------------
# MAIN
# ------------------------------
def upload_model_to_hf():
    print(f"Uploading from: {os.path.abspath(MODEL_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("You must log in first: run `huggingface-cli login`")

    api = HfApi()
    upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=MODEL_PATH,
        path_in_repo=".",  # Upload all files to root of repo
        repo_type="model",
        token=token,
    )

    print(f" Model uploaded successfully to: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    upload_model_to_hf()

