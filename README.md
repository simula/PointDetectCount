
# 🩺 PointDetectCount: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models

This repository contains the code and data generation scripts used in the paper:

**[Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models](https://arxiv.org/html/2505.16647v1)**  
`Sushant Gautam, Michael A. Riegler, Pål Halvorsen`  
*arXiv preprint, May 2025*

---

## 📌 Overview

PointDetectCount is a unified multi-task framework for fine-tuning instruction-tuned vision-language models (VLMs) on three fundamental medical imaging tasks:

- **Pointing (Localization)**
- **Bounding Box Detection**
- **Counting (Object Enumeration)**

The model is trained and evaluated on the [MedMultiPoints](https://huggingface.co/datasets/SimulaMet/MedMultiPoints) dataset, a multimodal dataset comprising diverse clinical annotations.

---

## 📦 Dataset

Dataset is available via Hugging Face:
👉 [`SimulaMet/MedMultiPoints`](https://huggingface.co/datasets/SimulaMet/MedMultiPoints)

All raw images should be stored locally in the `MedMultiPoints-images/` directory.

### Download Images Locally

You can download the image files directly from the Hugging Face dataset using the
[`datasets`](https://github.com/huggingface/datasets) library:

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("SimulaMet/MedMultiPoints")

# Path to save images and a metadata file
output_dir = "MedMultiPoints-images"

import os
os.makedirs(output_dir, exist_ok=True)

# Save one image per unique hash
for sha, row in ds["train"].to_pandas().groupby("image_sha256").nth(0).iterrows():
    row["image_data"].save(os.path.join(output_dir, f"{sha}.jpg"))
```

This snippet creates the `MedMultiPoints-images/` folder (if it doesn't already
exist) and writes each image from the dataset to that directory using the image's
SHA-256 hash as the filename.

| Columns              | Type         | Description                                                       |
|-------------------|--------------|-------------------------------------------------------------------|
| `image`           | Image        | Raw medical image                                                 |
| `image_sha256`    | string       | SHA-256 checksum for integrity                                    |
| `img_size`        | `[int, int]` | Image dimensions: `[width, height]`                               |
| `points`          | `[[x, y]]`   | List of point annotations                                         |
| `bbox`            | `[[x1, y1, x2, y2]]` | List of bounding boxes                                   |
| `count`           | int          | Number of annotated objects                                       |
| `label`           | string       | Object class (e.g., polyp, sperm, cluster, etc.)                  |
| `collection_method` | string     | Task relevance (e.g., detection, counting)                        |
| `classification`  | string       | Free-form annotation description                                  |
| `organ`           | string       | Organ or modality type (e.g., GI tract, sperm)                    |

**Instruction-Fused JSONL Files**:

- [`multi-task-train.jsonl`](https://huggingface.co/datasets/SimulaMet/MedMultiPoints/resolve/main/instruction_dataset/multi-task-train.jsonl)
- [`multi-task-test.jsonl`](https://huggingface.co/datasets/SimulaMet/MedMultiPoints/resolve/main/instruction_dataset/multi-task-test.jsonl)

---

## 💾 Fine-Tuned Model

Model weights are available via Hugging Face:
👉 [`SimulaMet/PointDetectCount-Qwen2.5-VL-7B-LoRA`](https://huggingface.co/SimulaMet/PointDetectCount-Qwen2.5-VL-7B-LoRA)

---

## 🛠️ Repository Structure

| File/Folder           | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `create_datasetJSON.py` | Generates instruction-formatted JSONL files for multi-task fine-tuning |
| `evaluate_qwen.py`      | Evaluates VLM outputs against structured annotations (bbox, point, count) |
| `MedMultiPoints-images/` | Directory to store dataset images locally |

---

## 🚀 Usage

### Create Instruction Dataset

Run the conversion script to produce an instruction-formatted dataset. Adjust the image directory or output path if needed:

```bash
python create_datasetJSON.py --image-dir MedMultiPoints-images --output kvasir_valid.jsonl
```

### Evaluate Predictions

Compare your model's predictions with the provided ground truth using:

```bash
python evaluate_qwen.py --dataset kvasir_valid-qwen-6task-test.jsonl --results kvasir_valid-qwen-6task-test-result.jsonl
```

### Fine-Tune Qwen (LoRA)

Training uses the instruction-fused training file available at
[`multi-task-train.jsonl`](https://huggingface.co/datasets/SimulaMet/MedMultiPoints/resolve/main/instruction_dataset/multi-task-train.jsonl):

```bash
swift sft --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset /home/sushant/D1/MIUA/kvasir-format/multi-task-train.jsonl \
    --output_dir /home/sushant/D1/MIUA/kvasir-format/training2 \
    --num_train_epochs 5 \
    --eval_steps 200 \
    --save_total_limit 3 \
    --report_to wandb \
    --per_device_train_batch_size 4
```

### Inference

Infer using either the fine-tuned checkpoint or the original model:

```bash
# Finetuned model
swift infer --model SimulaMet/PointDetectCount-Qwen2.5-VL-7B-LoRA \
    --val_dataset https://huggingface.co/datasets/SimulaMet/MedMultiPoints/resolve/main/instruction_dataset/multi-task-test.jsonl \
    --result_path qwen_outputs/qwen-finetuned-6task-test500-result.jsonl \
    --use_hf true

# Public checkpoint
swift infer --model Qwen/Qwen2.5-VL-7B-Instruct \
    --val_dataset https://huggingface.co/datasets/SimulaMet/MedMultiPoints/resolve/main/instruction_dataset/multi-task-test.jsonl \
    --result_path qwen_outputs/qwen-public-6task-test500-result.jsonl \
    --use_hf true
```

---

## 🧠 Methodology Summary

We fine-tune [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) using [LoRA](https://arxiv.org/abs/2106.09685) for instruction-based multi-task image understanding.

- Each image is associated with 5 instruction-response pairs.
- Responses are expected to be JSON-formatted predictions.
- Tasks are trained jointly using commonly used language modeling loss.

For more details, see [Section IV of the paper](https://arxiv.org/html/2505.16647v1#S4).

---

## 📊 Evaluation Metrics

| Task             | Metrics (Key)                                  |
|------------------|------------------------------------------------|
| **Counting**     | MAE, MSE                                       |
| **Pointing**     | Point MAE, RMSE, Matching Accuracy, Zero-cases |
| **Bounding Box** | mAP, mAP@50, mAP@75, IoU                       |

Evaluation scripts are provided in `evaluate_qwen.py`.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{PointDetectCount,
  title = {{Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models}},
  author = {Sushant Gautam and Michael A. Riegler and Pål Halvorsen},
  journal = {arXiv},
  year = {2025},
  month = may,
  note = {[Online; accessed 17. Jun. 2025]},
  url = {https://arxiv.org/html/2505.16647v1}
}
```

---

## 📬 Contact

For questions or collaboration inquiries, reach out to:

📧 sushant@simula.no
