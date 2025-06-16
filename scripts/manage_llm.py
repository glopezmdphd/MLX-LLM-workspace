from pathlib import Path
import sys

try:
    from mlx_lm import load
except ImportError as e:
    print("MLX is not installed. Please activate your venv and run 'pip install mlx-lm'.")
    print("ImportError:", e)
    sys.exit(1)

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIRS = {
    "original": BASE_DIR / "models" / "original",
    "quantized": BASE_DIR / "models" / "quantized",
    "exported": BASE_DIR / "models" / "exported"
}

def ensure_dirs():
    for path in MODEL_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)

def review_model():
    model_name = input("Enter the model name to review (e.g., mlx-community/Meta-Llama-3-8B): ")
    print(f"\nReviewing model: {model_name}")
    print("Model review functionality is limited to metadata display.")
    print("Tip: Check Hugging Face or MLX documentation for more details.\n")

def download_model():
    model_name = input("Enter the model name to download (e.g., mlx-community/Meta-Llama-3-8B): ")
    print(f"\nDownloading model: {model_name}")
    try:
        model, tokenizer = load(model_name)
    except Exception as e:
        print(f"Failed to download or load the model '{model_name}'.")
        print(f"Error: {e}")
        return
    save_path = MODEL_DIRS["original"] / model_name.replace("/", "_")
    save_path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(str(save_path))
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Model loaded but failed to save to disk: {e}")
    else:
        print("Model loaded in memory (saving to disk not supported by this MLX version).")

def quantize_model():
    print("Quantization functionality is not available in the current mlx_lm API.")
    print("Please refer to the MLX documentation for quantization support or update this script when the API is available.")

def export_model():
    print("Export functionality is not available in the current mlx_lm API.")
    print("Please refer to the MLX documentation for export support or update this script when the API is available.")

def display_menu():
    print("\nMLX LLM Interactive Menu")
    print("1. Review a model")
    print("2. Download a model")
    print("3. Quantize a model")
    print("4. Export a model")
    print("5. Exit")

def main():
    ensure_dirs()
    while True:
        display_menu()
        choice = input("Select an option (1-5): ").strip()
        if choice == '1':
            review_model()
        elif choice == '2':
            download_model()
        elif choice == '3':
            quantize_model()
        elif choice == '4':
            export_model()
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()

