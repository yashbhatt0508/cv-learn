
from pathlib import Path
from huggingface_hub import login, HfApi

# ==========================================
# AgeLens AI - Upload Model to Hugging Face
# ==========================================

REPO_ID = "Nekowow/AgeLens_AI"
MODEL_FILENAME = "resnet18_age_finetuned.pth"

# Find the model in the same folder as this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / MODEL_FILENAME


def main():
    # Check whether the model file exists
    if not MODEL_PATH.is_file():
        print("\nERROR: Model file not found!")
        print(f"Expected location: {MODEL_PATH}")
        print(
            "\nPlace resnet18_age_finetuned.pth "
            "in the same folder as upload_model.py."
        )
        return

    print(f"Model found: {MODEL_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")

    # Log in using your Hugging Face access token
    print("\nLog in to Hugging Face...")
    login()

    # Upload only the model checkpoint
    api = HfApi()

    print(f"\nUploading model to {REPO_ID}...")

    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo=MODEL_FILENAME,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload AgeLens AI model checkpoint",
    )

    print("\nSUCCESS! Model uploaded.")
    print(f"Repository: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
