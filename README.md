
# Optimizing LLMs: Efficient Inference & Adaptation for LLaMA-2

This project investigates optimization techniques for Large Language Models (LLMs), focusing on LLaMA-2. It integrates efficient architectural improvements and inference methods to reduce memory usage and speed up text generation without compromising model quality. The work leverages techniques such as Rotary Positional Encoding (RoPE), Grouped Multi-Query Attention (GMQA), Key-Value Caching, SwiGLU activation, and LoRA (Low-Rank Adaptation).

## 📁 Project Structure

- `model.py`: Implements the Transformer model architecture inspired by LLaMA-2, including custom attention, feedforward, and normalization modules.
- `inference-3.py`: Script for standard text generation using the base LLaMA model with top-p (nucleus) sampling.
- `lora_qlora_inference.py`: Demonstrates inference with optional LoRA adaptation and quantization for more efficient computation.
- `Final_Report_Group30.pdf`: Detailed technical report explaining motivations, architecture choices, benchmarks, results, and future directions.

## 🚀 Features

- **Rotary Positional Encoding (RoPE)**: Efficient relative position encoding for attention layers.
- **Grouped Multi-Query Attention (GMQA)**: Reduces redundancy in attention computation.
- **Key-Value Caching**: Speeds up autoregressive generation by caching intermediate results.
- **SwiGLU Activation**: Improves model expressiveness with a gated linear unit.
- **LoRA Integration**: Enables efficient fine-tuning with low-rank matrices instead of full weight updates.
- **Top-p Sampling**: Controls randomness in output for better quality generations.

## 📊 Benchmark Highlights

- 20% improvement in inference speed vs baseline (1.8s vs 2.3s per prompt).
- BLEU score improvements in translation tasks.
- Enhanced output diversity and reduced hallucinations via sampling + prompt refinement.
- LoRA-enabled inference reduces training overhead and adapts well to new tasks.

## 🧪 How to Run

### 1. Install Dependencies

```bash
pip install torch tqdm sentencepiece
```

### 2. Prepare Checkpoints & Tokenizer

Ensure you have the LLaMA-2 checkpoint files and tokenizer model placed inside:

```
Llama-2-7b/
├── tokenizer.model
├── <checkpoint>.pth
├── params.json
```

### 3. Standard Inference

```bash
python inference-3.py
```

### 4. Inference with LoRA

```bash
python lora_qlora_inference.py
```

Optional flags inside `lora_qlora_inference.py` allow enabling/disabling LoRA or quantization.

## 📚 Report Summary

The [Final_Report_Group30.pdf](./Final_Report_Group30.pdf) includes:

- Technical breakdown of architectural enhancements
- Literature review of transformer optimizations
- Results from benchmarking text generation and translation
- Mathematical formulation of core model operations
- Discussion of a Retrieval-Augmented Generation (RAG) pipeline

## 👥 Contributors

- Priyanshu M Sharma
- Sarthak Mishra
- Sai Krishna Reddy Daka
- Kaumudi Patil
- Vibhu Dixit  
_(Arizona State University)_

## 🔮 Future Work

- Dynamic RAG integration
- Quantization + pruning for edge deployment
- Multilingual & adversarial robustness testing
- Continual learning with adaptive LoRA tuning

## 📄 License

MIT License
