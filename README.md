# LoRA from Scratch: Low-Rank Adaptation of GPT-2

A from-scratch PyTorch implementation of the paper **"LoRA: Low-Rank Adaptation of Large Language Models"** (Hu et al., 2021) — applied to GPT-2 for Shakespeare-style text generation.

> **Paper:** [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

## What This Project Does

This project implements LoRA (Low-Rank Adaptation) from scratch using only PyTorch — no external LoRA libraries. It fine-tunes GPT-2 (124M parameters) on Shakespeare's works, teaching the model to generate Shakespearean text by training only **~0.15% of the total parameters**.

**Before LoRA training:**
```
Prompt: "To be or not to be"
Output: "To be or not to be able to do that, but I think the most
important thing is that we need to have a conversation about how..."
```

**After LoRA training:**
```
Prompt: "To be or not to be"
Output: "To be or not to be, that is the question:
Whether thou dost seek the favour of the king,
Or bid thy servant speak in humble tongue..."
```

## Key Results

| Metric | Value |
|--------|-------|
| Base model parameters | 124,439,808 |
| LoRA trainable parameters | ~185,000 |
| Percentage trained | ~0.15% |
| Full model file size | ~500 MB |
| LoRA weights file size | ~1.5 MB |
| Storage reduction | ~300x smaller |
| Training time (Colab T4) | ~30 minutes |

## How LoRA Works

Large language models have millions of parameters, but fine-tuning research suggests that weight updates during adaptation live in a low-rank subspace. LoRA exploits this by freezing the pretrained weights **W₀** and injecting two small trainable matrices **B** and **A** such that:

```
W' = W₀ + BA × (α/r)
```

Where:
- **W₀** (768 × 2304) = original frozen weights (589,824 parameters)
- **B** (768 × 4) = trainable matrix initialized to zeros (3,072 parameters)
- **A** (4 × 2304) = trainable matrix initialized with random Gaussian (9,216 parameters)
- **r = 4** = rank (the bottleneck dimension)
- **α = 8** = scaling factor

At initialization, B is all zeros so BA = 0 and the model behaves identically to base GPT-2. During training, only A and B are updated while W₀ remains frozen.

## Project Structure

```
lora-from-scratch/
├── lora_from_scratch.ipynb    # Main implementation notebook
├── lora_ablations.ipynb       # Rank comparison & layer selection experiments
├── lora_weights.pt            # Saved LoRA weights (~1.5 MB)
├── training_losses.json       # Training loss history
├── figures/
│   ├── training_curve.png     # Loss over training steps
│   ├── rank_comparison.png    # Performance across different ranks
│   └── before_after.png       # Text generation comparison
└── README.md
```

## Implementation Details

### The LoRA Module (From Scratch)

The core implementation is a single PyTorch module that maps directly to Section 4.1 of the paper:

```python
class LoRALayer(nn.Module):
    def __init__(self, fan_in, fan_out, rank=4, alpha=8):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, fan_out))   # Gaussian init
        self.lora_B = nn.Parameter(torch.zeros(fan_in, rank))    # Zero init
        self.scale = alpha / rank                                 # α/r scaling

    def forward(self, original_weights):
        return original_weights + (self.lora_B @ self.lora_A) * self.scale
```

LoRA is injected into GPT-2's attention layers using PyTorch's `parametrize` utility, which intercepts weight access so the model reads `W₀ + BA` instead of just `W₀` — without modifying GPT-2's source code.

### Layers Adapted

Following Section 4.2 and 7.1 of the paper, LoRA is applied to:
- **`c_attn`** — the combined Query/Key/Value projection (Wq, Wk, Wv)
- **`c_proj`** — the output projection (Wo)

across all 12 transformer blocks in GPT-2 small.

### Training Configuration

| Hyperparameter | Value | Paper Reference |
|---------------|-------|-----------------|
| Rank (r) | 4 | Section 7.2 |
| Alpha (α) | 8 | Section 4.1 |
| Learning rate | 1e-4 | Section 5 |
| Optimizer | AdamW | Section 4.1 |
| Weight decay | 0.01 | — |
| Batch size | 2 | — |
| Max sequence length | 256 | — |
| Epochs | 3 | — |
| Dataset | Tiny Shakespeare (~1MB) | — |

### Weight Merging for Deployment

After training, the LoRA matrices are merged back into the base weights for zero-overhead inference:

```
W_deployed = W₀ + BA × (α/r)
```

The merged model produces identical output to the LoRA-adapted model but without computing BA at every forward pass — matching the paper's claim of no additional inference latency (Section 4.1).

## Ablation Studies

### Rank Comparison (Section 7.2)

Trained with ranks r = 1, 2, 4, 8, 16 on the same dataset and compared final loss and generation quality. Consistent with the paper's findings, even very low ranks (r = 1, 2) produce meaningful adaptation, supporting the hypothesis that fine-tuning updates have low intrinsic dimensionality.

### Layer Selection (Section 7.1)

Compared applying LoRA to:
1. Only `c_attn` (QKV projection)
2. Only `c_proj` (output projection)
3. Both `c_attn` and `c_proj`

Adapting both layers with a smaller rank outperformed adapting a single layer with a higher rank at the same total parameter budget — matching the paper's recommendation.

## How to Run

### Requirements
- Google Colab (free tier with T4 GPU) or any machine with a CUDA GPU
- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers

### Quick Start

1. Open `lora_from_scratch.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run all cells sequentially

```bash
# If running locally
pip install torch transformers matplotlib

# Clone and run
git clone https://github.com/YOUR_USERNAME/lora-from-scratch.git
cd lora-from-scratch
jupyter notebook lora_from_scratch.ipynb
```

## What I Learned

- **The math is simpler than it looks.** LoRA's core idea — decomposing weight updates into low-rank matrices — requires only basic linear algebra (matrix multiplication and dimensionality).
- **Initialization matters.** Setting B to zeros ensures the model starts from the pretrained checkpoint exactly. Random initialization for both A and B would corrupt the pretrained knowledge immediately.
- **PyTorch's `parametrize` is powerful.** It lets you modify how weights are accessed without changing model source code — essential for applying LoRA to any pretrained model.
- **Rank is surprisingly forgiving.** Even rank 1 produces noticeable adaptation, confirming the paper's central hypothesis about low intrinsic dimensionality of fine-tuning updates.
- **The storage savings are dramatic.** Going from ~500MB (full model) to ~1.5MB (LoRA weights) with minimal quality loss demonstrates why LoRA became the standard for model adaptation.

## Paper Sections → Code Mapping

| Paper Section | What It Covers | Where in Code |
|--------------|----------------|---------------|
| Section 4.1 | LoRA method & math | `LoRALayer` class |
| Section 4.2 | Applying to transformers | `inject_lora()` function |
| Section 5 | Training setup & hyperparameters | Training loop |
| Section 7.1 | Which layers to adapt | Layer selection ablation |
| Section 7.2 | Rank analysis | Rank comparison ablation |

## References

- **Paper:** Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Official Implementation:** [microsoft/LoRA](https://github.com/microsoft/LoRA)
- **GPT-2:** Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
- **Umar Jamil's Tutorial:** [YouTube — LoRA Explained Visually + PyTorch Code](https://www.youtube.com/@umarjamilai)

## License

MIT
