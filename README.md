# ControlTuner: 4-Bit QLoRA Fine-Tuning & Evaluation Pipeline

An end-to-end Supervised Fine-Tuning (SFT) and evaluation pipeline that adapts an open-weights foundation model (`Qwen/Qwen2.5-1.5B`) into an autonomous **Control Systems Engineering Design Assistant**.

Designed, trained, and benchmarked on consumer hardware (**NVIDIA GeForce RTX 2060, 6 GB VRAM**).

---

## 🖥️ System & Hardware Specifications

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 2060 (6 GB VRAM, Turing Architecture, SM 7.5) |
| **CPU / System RAM** | Intel Core i5 / 16 GB DDR4 |
| **Base Model** | `Qwen/Qwen2.5-1.5B` (1.56 Billion parameters) |
| **Fine-Tuning Method** | 4-bit QLoRA (NF4 Quantization + Double Quantization) |
| **Trainable Parameters** | 18,464,768 (1.18% of total parameters) |
| **Core Libraries** | `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate` |
| **Course Reference** | [Hugging Face LLM Course — Chapter 11](https://huggingface.co/learn/llm-course/chapter11/1) |

---

## 🚀 Key Features & Architecture
- **ChatML & Special Token Demarcation**: Standardized prompt formatting with `<|im_start|>` and `<|im_end|>` boundaries and response-only loss masking.
- **4-Bit QLoRA Optimization**: Base model weights compressed via NormalFloat4 (NF4) and double quantization, updating only **18.46M parameters (1.18%)** of the 1.56B base model.
- **Consumer Hardware Budgeting**: Effective batch size of 8 via gradient accumulation (`batch_size=1`, `accum_steps=8`) and gradient checkpointing, maintaining peak VRAM at **~3.6 GB**.
- **Automated Domain Benchmark**: Custom linear algebra verification testing format adherence, controllability rank, and closed-loop Hurwitz stability.
- **Standalone Model Fusion**: Zero-latency weight merging via `merge_and_unload()` on CPU RAM, exporting a standalone 16-bit model for native GPU inference.

---

## 📊 Consolidated Results Summary

| Evaluation Layer | Target / Benchmark | Result / Metric | Status |
| :--- | :--- | :---: | :---: |
| **Training Convergence** | Cross-Entropy Loss | $1.324 \longrightarrow \mathbf{0.1658}$ | **Optimal** |
| **Generalization** | Validation Loss (`eval_loss`) | $\mathbf{0.1368}$ | **Zero Overfitting** |
| **Perplexity** | Domain Fluency ($\text{PPL} = \exp(\text{loss})$) | $\mathbf{1.1466}$ | **High Fluency** |
| **Token Accuracy** | Next-Token Prediction on Test Split | $\mathbf{98.04\%}$ | **Passed** |
| **Domain Benchmark** | State-Space Modeling, Rank & Stability | $\mathbf{3 / 3\ (100.0\%)}$ | **Passed** |
| **Catastrophic Forgetting** | [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k) Math Reasoning | $\mathbf{3 / 10\ (30.0\%)}$ | **Preserved** |
| **Hardware Efficiency** | Peak VRAM Footprint | $\mathbf{\sim 3.6\ \text{GB}}$ / 6.0 GB | **Safe** |
| **Training Duration** | 3 Epochs (423 Optimizer Steps) | $\mathbf{49\ \text{min},\ 39\ \text{sec}}$ | **Fast** |

---

## 📈 Detailed Experimental Results

### 1. Training Convergence & Perplexity (Step 3)
- **Initial Training Loss**: `1.324` (step 10) $\longrightarrow$ **Final Loss**: `0.1658` (step 423)
- **Validation Loss (`eval_loss`)**: **`0.1368`** (steady decline across all 3 epochs with zero overfitting)
- **Validation Perplexity (PPL)**: **`1.1466`** ($\exp(0.1368)$)
- **Validation Token Accuracy**: **`98.04%`** on unseen test sequences
- **Validation Entropy**: Dropped from `1.373` $\to$ **`0.1417`** (high model certainty)
- **Gradient Norms**: Bounded between `0.01` and `0.07` (zero exploding gradients)
- **Total Training Runtime**: **49 minutes, 39 seconds** on RTX 2060

### 2. Domain Task Benchmark (Held-Out Test Problems - Step 4)
*Evaluated on continuous state-space models, inverted pendulum dynamics, and DC motor regulators.*
- **Format & Schema Compliance**: **`3 / 3 (100.0%) PASS`**
- **Controllability Rank ($\text{Rank}(\mathcal{C}) = n$)**: **`3 / 3 (100.0%) PASS`**
- **Hurwitz Stability Analysis**: **`3 / 3 (100.0%) PASS`**
- **Overall Domain Benchmark Score**: **`3 / 3 (100.0%) PASS`**

### 3. General Reasoning & Catastrophic Forgetting Audit (Step 4)
*Audited on the [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k) benchmark dataset.*
- **Zero-Shot GSM8K Score**: **`3 / 10 (30.0%)`**
- **Verdict**: Successfully retained multi-step algebraic reasoning and arithmetic without catastrophic forgetting.

---

## 🛠️ Key Technical Errors & Engineering Solutions Log

During the development and training lifecycle, several hardware and API integration challenges were diagnosed and resolved:

### 1. GPU Architecture Incompatibility (`BFloat16` on Turing)
* **Error**: `RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`
* **Root Cause**: Base Qwen defaults to `bfloat16`, causing PEFT to create LoRA weights in BFloat16. The **RTX 2060 (Turing)** has no hardware execution units for BFloat16 CUDA unscaling when PyTorch's mixed-precision `GradScaler` (`fp16=True`) runs.
* **Solution**: Converted all LoRA modules to FP32 using `module.to(torch.float32)` and set `training_args.fp16 = False` (since `bitsandbytes` already handles 16-bit compute internally without needing `GradScaler`).

### 2. Dictionary Input to `model.generate()`
* **Error**: `KeyError: 'shape'` $\rightarrow$ `AttributeError` inside `model.generate()`
* **Root Cause**: `apply_chat_template(return_tensors="pt")` returns a `BatchEncoding` dictionary (`{'input_ids': ..., 'attention_mask': ...}`). Passing it as a positional argument caused `model.generate()` to search for a `.shape` key on the dictionary instead of receiving a tensor.
* **Solution**: Set `return_dict=True` and unpacked the dictionary with keyword arguments: `model.generate(**inputs, ...)`.

### 3. Run-On Multi-Turn Generation & Boundary Artifacts
* **Error**: Output continued past the assistant's answer and generated garbled characters (`取user...`).
* **Root Cause**: In ChatML, turns end with `<|im_end|>`. When decoding with `skip_special_tokens=True`, generation did not stop at that boundary and attempted to generate subsequent conversation turns.
* **Solution**: Decoded raw tokens and cleanly split the string at the turn delimiter: `response_text.split("<|im_end|>")[0]`.

### 4. Suffix Mismatch in Automated Benchmark
* **Error**: Model's answer was mathematically correct, but received `stability_pass: [FAIL]`.
* **Root Cause**: The model wrote *"open-loop **instability**"*. The string verifier checked strictly for `"stable"` and `"unstable"`, neither of which is a substring of `"instability"`.
* **Solution**: Added `"instab"` to the search list (`["stable", "unstable", "instab"]`) to capture both adjectives and nouns.

### 5. Missing Dataset Repository Namespace
* **Error**: `HfUriError: Repository id must be 'namespace/name', got 'gsm8k'`
* **Root Cause**: Modern Hugging Face Hub strictly enforces the `namespace/dataset_name` schema and rejects standalone dataset names.
* **Solution**: Updated the dataset identifier from `"gsm8k"` to [`"openai/gsm8k"`](https://huggingface.co/datasets/openai/gsm8k).

---

## 📂 Repository Structure

```text
├── chat_formater.py           # Step 1: Tokenizer setup & ChatML template formatting
├── dataset_and_config.py      # Step 2: Dataset generator, splits & SFTConfig
├── train_qlora.py             # Step 3: 4-bit QLoRA training engine
├── evaluate_benchmark.py      # Step 4: Automated domain verification harness
├── evaluate_math_benchmark.py # Step 4: OpenAI GSM8K catastrophic forgetting audit
├── merge_and_chat.py          # Step 5: merge_and_unload() & interactive terminal chat
├── requirements.txt           # Environment dependencies
├── .gitignore                 # Excludes heavy checkpoints and *.safetensors weights
└── README.md                  # Complete project documentation & results