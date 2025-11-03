# 🧠 Lightweight Fine-Tuning with PEFT (LoRA & QLoRA)

This project demonstrates how to apply **Parameter-Efficient Fine-Tuning (PEFT)** techniques, specifically **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)**, to a pretrained foundation model using the **Hugging Face Transformers** and **PEFT** libraries.

---

## 📋 Project Overview

Modern large language models (LLMs) can be extremely expensive to fine-tune because of their large number of parameters.  
**PEFT** methods like **LoRA** allow you to adapt only a *small number* of additional parameters while keeping the base model frozen — making training efficient and lightweight.

This notebook fine-tunes a **DistilBERT** model on the **IMDB movie review dataset** for **binary sentiment classification**.  
It then compares the performance of:
- The original pretrained model
- Two LoRA configurations
- An optional QLoRA configuration (for GPUs)

Finally, the saved LoRA model is reloaded using `AutoPeftModelForSequenceClassification` to demonstrate **inference from saved weights**.

---

## 🧩 Key Components

### 1️⃣ PEFT Technique
- **LoRA (Low-Rank Adaptation):**
  - Injects trainable low-rank matrices into attention layers.
  - Reduces the number of trainable parameters by up to 99%.
  - Allows reusing pretrained models efficiently.

- **QLoRA (Quantized LoRA):**
  - Combines LoRA with 8-bit quantization using the `bitsandbytes` package.
  - Enables fine-tuning even on GPUs with limited VRAM.

---

### 2️⃣ Model Choice
- **Model:** `distilbert-base-uncased`
  - A lightweight version of BERT with excellent trade-off between performance and computational cost.
  - Suitable for resource-constrained environments like the provided workspace.

---

### 3️⃣ Dataset
- **Dataset:** `IMDB` (via Hugging Face Datasets)
  - Binary sentiment classification (positive or negative movie reviews).
  - Only small subsets of the training (1,000 samples) and test sets (500 samples) were used to fit workspace constraints.

---

### 4️⃣ Evaluation Metrics
- **Metrics:** `accuracy` and `F1` from the `evaluate` library.
- These metrics are computed using Hugging Face’s `Trainer` both before and after fine-tuning.
- The comparison highlights how LoRA adapters improve task-specific performance with minimal overhead.

---

### 5️⃣ Inference Workflow
After fine-tuning, the LoRA adapter is saved using:
```python
peft_model.save_pretrained("./lora_model_0")
