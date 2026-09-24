# JevPertus

![JevPertus banner](assets/jevpertus-banner.png)

[![Code license: MIT](https://img.shields.io/badge/Code-MIT-green)](#license)
[![Data license: Apache 2.0](https://img.shields.io/badge/Data-Apache%202.0-blue)](https://github.com/jaredpalmer/kev/blob/main/LICENSE)

A simple, lightweight implementation of the Jev model on top of the Apertus LLM.
JevPertus combines an Apertus text backbone, LoRA adapters, and a small pointer head to assign probabilities to a question's answer options.

It supports multiple-choice questions, ordered rating scales, and true/false statements. Predictions come from scoring the supplied options in a single backbone pass.

## How it works

1. Encode the context (`state`), question (`instructions`), and options using Apertus special tokens `<SPECIAL_100>` through `<SPECIAL_104>`.
2. Run the encoded sequence through the backbone to obtain token hidden states.
3. Use a pointer head to compare the final decision state with each option's end state.
4. Apply softmax to the option logits to obtain probabilities.

Training updates the LoRA adapters and pointer head while keeping the base weights frozen. Batches support questions with different sequence lengths and option counts.

## Setup

Use Python 3.11 or newer.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU use, install a PyTorch build compatible with your CUDA environment.

The default backbone is `swiss-ai/Apertus-v1.5-8B`. The loader downloads model files from Hugging Face when they are not cached; it also accepts a local checkpoint directory. Configure Hugging Face credentials if your chosen checkpoint requires authentication.

The full backbone must fit on the selected device alongside activations and training state.

## Data format

The training script reads `data/train.jsonl` and evaluates on `data/test.jsonl`. Each line is a JSON object containing a shared `state` and a mapping of question IDs to labeled questions:

```json
{"state":"You live in Zurich.","questions":{"q1":{"type":"choice","instructions":"Which country do you live in?","criteria":{"A":"Switzerland","B":"France"},"label":"A"},"q2":{"type":"noul","instructions":"You live in Switzerland.","label":true},"q3":{"type":"score","instructions":"How certain are you?","criteria":["Uncertain","Certain"],"label":1}}}
```

| Type     | `criteria`                                                | Label in JSONL                                         |
| -------- | --------------------------------------------------------- | ------------------------------------------------------ |
| `choice` | Mapping of answer keys to descriptions, in option order   | Answer key such as `"A"`, or a zero-based option index |
| `score`  | Ordered list of rating descriptions                       | Zero-based option index                                |
| `noul`   | Optional mapping with `"false"` and `"true"` descriptions | Boolean, or `0` for false and `1` for true             |

Every training and evaluation question needs a label. The loader converts choice keys and boolean labels into zero-based indices. For inference, supply an individual question with its own `state` and omit the label.

## Training

An example train script can be found in `train_jevpertus.py`.
To configure training, set the following environment variables:

| Setting         | Default                    |
| --------------- | -------------------------- |
| `BASE_MODEL`    | `swiss-ai/Apertus-v1.5-8B` |
| `DATA_DIR`      | `data`                     |
| `DEVICE`        | `cuda:0`                   |
| `EPOCHS`        | `2`                        |
| `BATCH_SIZE`    | `1`                        |
| `LORA_RANK`     | `16`                       |
| `LEARNING_RATE` | `5e-5`                     |
| `OUTPUT_DIR`    | `runs/jevpertus`           |

Training writes per-step loss and per-epoch evaluation metrics to `loss_history.jsonl`, and saves a checkpoint after each epoch:

```text
runs/jevpertus/
├── loss_history.jsonl
├── epoch_1/
│   ├── adapter_config.json
│   ├── jev_config.json
│   └── jev.pt
└── epoch_2/
    ├── adapter_config.json
    ├── jev_config.json
    └── jev.pt
```

Checkpoints contain the LoRA adapter weights, pointer-head weights, and configuration. Base-model weights and optimizer state are not saved. Loading a checkpoint requires access to its base model. Choose a new `OUTPUT_DIR` for each run to preserve previous metrics and checkpoints.

## Inference

To see how JevPertus performs on a few example questions, run `inference_jevpertus.py`. The script loads a checkpoint and prints predictions for three sample questions.

The examples cover all three question types. Choice and score questions print a probability for each option; `noul` questions print the probability of true.

## Project layout

| File                          | Purpose                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `dataloader.py`               | JSONL loading, question encoding, padding, and batching        |
| `modeling/apertus/apertus.py` | Apertus text architecture                                      |
| `modeling/apertus/loading.py` | Loading compatible original Apertus and V1.5 text checkpoints  |
| `modeling/pointerhead.py`     | Option-scoring head                                            |
| `modeling/jev.py`             | Backbone/head composition, LoRA setup, and checkpoint handling |
| `train_jevpertus.py`          | Training and evaluation entry point                            |
| `inference_jevpertus.py`      | Checkpoint loading and example predictions                     |

## Acknowledgments

JevPertus builds on the Apertus backbone. The banner is inspired by the Apertus wordmark, with custom JevPertus lettering.

## Authors

- [Jan-Lucas Uslu](https://github.com/Jaluus) - JevPertus.
- [Jared Palmer and the Kev contributors](https://github.com/jaredpalmer/kev) - source of the data included in this repository.

## License

- **Code:** licensed under the [MIT License](https://opensource.org/license/mit).
- **Data:** the datasets in `data/` come from [Kev](https://github.com/jaredpalmer/kev) and are licensed under [Apache License 2.0](https://github.com/jaredpalmer/kev/blob/main/LICENSE). Upstream copyright: 2026 Jared Palmer.
