# MLX-LLM-workspace

A unified, modular workflow for reviewing and downloading Large Language Models (LLMs) using [MLX](https://github.com/ml-explore/mlx) on Apple Silicon (macOS, M4/M3/M2/M1).  
Designed for edge AI development and deployment.

---

## Features

- **Review**: Inspect model metadata before download (metadata display only).
- **Download**: Fetch models from Hugging Face or other sources using MLX.
- **Quantize**: *Not available in the current Python script. Use the `mlx-lm` CLI for quantization.*
- **Export**: *Not available in the current Python script. Use the `mlx-lm` CLI for export/conversion.*
- **Single Script Workflow**: All steps handled by one Python script (`manage_llm.py`) via an interactive menu.

---

## Folder Structure

MLX-LLM-workspace/ ├── .gitignore ├── README.md ├── requirements.txt ├── data/ ├── models/ │ ├── exported/ │ ├── original/ │ └── quantized/ ├── notebooks/ └── scripts/ └── manage_llm.py

- **models/**: All model files are stored here, organized by type (`original`, `quantized`, `exported`).
- **scripts/**: Contains the main management script.
- **data/**: Place for any datasets (optional).
- **notebooks/**: For Jupyter notebooks.

---

## Usage

1. **Install dependencies:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Run the management script:**
    ```sh
    python scripts/manage_llm.py
    ```

3. **Follow the interactive menu to review or download models.**

---

## Quantization & Export

- **Quantization and export are not available in the current Python script.**
- To quantize or export models, use the `mlx-lm` command-line interface (CLI):

    ```sh
    mlx-lm quantize <model_path> <output_path> --bits 4
    mlx-lm export <model_path> <output_path> --format onnx
    ```

    See the [mlx-lm documentation](https://github.com/ml-explore/mlx-lm) for details.

---

## Notes

- The script will create the [models](http://_vscodecontentref_/4) directory structure automatically if it does not exist.
- Model saving depends on MLX support for [save_pretrained](http://_vscodecontentref_/5). If not supported, models are loaded in memory only.

---

## License

MIT