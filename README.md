# MLX-LLM-workspace

A unified, modular workflow for reviewing, downloading, quantizing, and exporting Large Language Models (LLMs) using [MLX](https://github.com/ml-explore/mlx) on Apple Silicon (macOS, M4/M3/M2/M1).  
Designed for edge AI development and deployment, including export to Jetson Orin NX.

---

## Features

- **Review**: Inspect model metadata before download.
- **Download**: Fetch models from Hugging Face or other sources using MLX.
- **Quantize**: Efficiently quantize models (4-bit, 8-bit) for local or edge use.
- **Export**: Convert models to formats (ONNX, GGUF) suitable for deployment on devices like Jetson Orin NX.
- **Single Script Workflow**: All steps handled by one Python script (`manage_llm.py`) using command-line arguments.

---

## Folder Structure

