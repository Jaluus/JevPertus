# Jevtests


We will start by implementing a simple local Jevtzpe model.
The code is kept lean and simple, desgined not for production use, but for understanding the inner workings of Qwen and Jev.

FlappyJev is a small subproejct of the Jevtests project. It is a simple implementation of the Flappy Bird game, designed to test the capabilities of Jev.

## Loading Qwen3.5 text weights

Install `torch`, `huggingface_hub`, and `safetensors`. Run this from the project directory:

```python
import torch
from qwen3_5_4b import Qwen3_5Model

model = Qwen3_5Model.from_pretrained(
    "Qwen/Qwen3.5-4B",
    device="cpu",  # Use "cuda" to load onto a GPU.
    dtype=torch.bfloat16,
)
```

You can also pass a local checkpoint directory containing `config.json` and the
safetensors file(s). Downloads are cached by Hugging Face; `cache_dir` and
`revision` can be passed to `from_pretrained`.

The loader checks tensor names and shapes, loads only the text parameters, and
ties the output projection to the token embeddings. Downloaded shards also
contain vision and auxiliary prediction weights, which are skipped. The text
weights occupy approximately 8.4 GB in bfloat16, plus memory for activations.

Use the matching `Qwen/Qwen3.5-4B` tokenizer (for example, via Transformers'
`AutoTokenizer`) to produce input IDs; chat prompts need its chat template.
The model returns logits and still recomputes the full sequence on each call.
