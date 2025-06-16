import os
from pathlib import Path
import sys

try:
    from mlx_lm import load, quantize, export  # Adjust as needed for your MLX version
except ImportError:
    print("MLX is not installed. Please activate your venv and run 'pip install mlx-lm'.")
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
    model, tokenizer = load(model_name)
    save_path = MODEL_DIRS["original"] / model_name.replace("/", "_")
    save_path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(save_path))
        print(f"Model saved to {save_path}")
    else:
        print("Model loaded in memory (saving to disk not supported by this MLX version).")

def quantize_model():
    model_name = input("Enter the model name to quantize (e.g., mlx-community/Meta-Llama-3-8B): ")
    bits = input("Enter quantization bits (4 or 8): ")
    try:
        bits = int(bits)
        if bits not in [4, 8]:
            raise ValueError
    except ValueError:
        print("Invalid input. Please enter 4 or 8.")
        return
    print(f"\nQuantizing model: {model_name} to {bits}-bit")
    model_path = MODEL_DIRS["original"] / model_name.replace("/", "_")
    if not model_path.exists():
        print(f"Original model not found at {model_path}. Please download first.")
        return
    try:
        model, tokenizer = load(str(model_path))
    except Exception:
        model, tokenizer = load(model_name)
    quantized_model = quantize(model, bits)
    quant_path = MODEL_DIRS["quantized"] / f"{model_name.replace('/', '_')}_{bits}bit"
    quant_path.mkdir(parents=True, exist_ok=True)
    if hasattr(quantized_model, "save_pretrained"):
        quantized_model.save_pretrained(str(quant_path))
        print(f"Quantized model saved to {quant_path}")
    else:
        print("Quantized model in memory (saving to disk not supported by this MLX version).")

def export_model():
    model_name = input("Enter the model name to export (e.g., mlx-community/Meta-Llama-3-8B): ")
    export_format = input("Enter export format (onnx or gguf): ").strip().lower()
    bits = input("Enter quantization bits (4 or 8, or leave blank for original): ").strip()
    bits = int(bits) if bits in ['4', '8'] else None
    print(f"\nExporting model: {model_name} to format: {export_format}")
    if bits:
        quant_path = MODEL_DIRS["quantized"] / f"{model_name.replace('/', '_')}_{bits}bit"
        if not quant_path.exists():
            print(f"Quantized model not found at {quant_path}. Please quantize first.")
            return
        model, tokenizer = load(str(quant_path))
    else:
        model_path = MODEL_DIRS["original"] / model_name.replace("/", "_")
        if not model_path.exists():
            print(f"Original model not found at {model_path}. Please download first.")
            return
        model, tokenizer = load(str(model_path))
    export_path = MODEL_DIRS["exported"] / f"{model_name.replace('/', '_')}.{export_format}"
    if hasattr(model, "export"):
        model.export(str(export_path), format=export_format)
        print(f"Exported model saved to {export_path}")
    else:
        print("Export functionality not supported by this MLX version or model type.")

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

