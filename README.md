# Hotel Guest Intent: LoRA Fine-Tune of Qwen2.5-0.5B

Fine-tunes **Qwen2.5-0.5B-Instruct** with **LoRA (PEFT)** to classify hotel guest messages into
**25 intents** (check_in, check_out, book_parking_space, get_refund, store_luggage, shuttle_service, …).
A hotel voice agent can use it to route each request to the right tool quickly and cheaply.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `bitext/Bitext-hospitality-llm-chatbot-training-dataset` (25k rows, CDLA-Sharing-1.0) |
| Method | LoRA r=16, alpha=32, dropout 0.05, on q/k/v/o + MLP projections; loss masked to the answer only |
| Training | 4,000 balanced examples (160 per intent), 2 epochs, lr 2e-4 cosine, Colab T4 |
| Evaluation | Exact-match accuracy on 500 held-out messages, **before vs after** fine-tuning |

## Results
Run on a free Google Colab **Tesla T4**, 12-10-2026 (full numbers in `results.json`):

| Model | Accuracy on 500 held-out messages |
|---|---|
| Base Qwen2.5-0.5B-Instruct (zero-shot, same prompt) | **44.0%** (220/500) |
| + LoRA fine-tune | **99.0%** (495/500) |

- **Trainable parameters:** 8.8M of 503M (**1.75%**). The adapter is a few MB; the base model is unchanged.
- **Training:** 250 steps (2 epochs) in **30.7 minutes**; loss fell from 0.31 to ~0.003.
- **No leakage:** the test messages come from a separate split, never seen in training.
- **Honest caveat:** Bitext is a synthetic dataset with many paraphrases of each request, so real guest messages (typos, mixed languages, several requests at once) would score lower. The next step would be testing on real or Arabic/English mixed messages.

## Run
**Colab (recommended):** open `lora_hotel_intent.ipynb` → T4 GPU → Run all.

**Locally:**
```bash
pip install "transformers>=4.56" peft datasets accelerate torch
python lora_hotel_intent.py          # needs a GPU for the full run
SMOKE=1 python lora_hotel_intent.py  # CPU check: 3 steps, 8 test examples
```

## What I'd explain in an interview
- **Why LoRA:** it freezes the base weights and trains small low-rank matrices (rank r) added to chosen layers, so only ~1% of parameters train. It's cheap, fast, and the adapter is a few MB.
- **r and alpha:** r sets the capacity of the update; alpha/r scales it (here 2).
- **Loss masking:** prompt tokens get label -100, so the model learns only to produce the intent.
- **Balanced sampling:** an equal number of examples per intent, so frequent intents don't dominate.
- **Fair evaluation:** the same held-out messages and the same prompt before and after training.
