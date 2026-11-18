# Code and Data Supplement — Medical LLM Faithfulness Audit (AAAI-27)

Anonymous companion archive for:
*On Auditing Reasoning Faithfulness and Confidence Calibration in Medical LLMs*.

## Contents
- `scripts/` — faithfulness, metacognition, frontier API, RQ2/RQ3 training & analysis
- `data/manifests/` — frozen eval manifests (`seed=42`), smoke sets, rebuild instructions
- `results/` — reported JSONL/summaries
- `configs/run_configs/` — recorded training hyperparameters (adapter weights omitted)
- `requirements.txt`, `.env.example`, `LICENSE` (MIT)

Trained LoRA adapters are omitted (multi-GB). Regenerate with the provided training scripts.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # API keys only for frontier/paraphrase
```

**Infrastructure:** 1× GPU ≥16GB VRAM (bf16) or ≥6GB (4-bit QLoRA); Python 3.10+; PyTorch ≥2.1; transformers ≥4.45; peft; bitsandbytes; accelerate; datasets. Pilot runs used RunPod RTX PRO 6000-class GPUs.

## Reproduce
```bash
python3 scripts/eval_faithfulness.py \
  --model FreedomIntelligence/HuatuoGPT-o1-8B \
  --manifest data/manifests/medxpertqa_reasoning_n200_seed42.json --seed 42

python3 scripts/rq2/eval_metacognition.py \
  --model FreedomIntelligence/HuatuoGPT-o1-8B \
  --manifest data/manifests/metamedqa_eval_seed42.json --seed 42

# rebuild full training manifests first (see data/manifests/README_MANIFESTS.md)
bash scripts/rq1/run_dense_lora.sh
bash scripts/rq1/run_moe_lora.sh
bash scripts/rq1/run_abstention_recovery.sh   # RQ3: lr=1e-4, max_seq_len=1536

python3 scripts/analysis/compute_uncertainty.py
```

HF datasets: `TsinghuaC3I/MedXpertQA`, `maximegmd/MetaMedQA`, `GBaker/MedQA-USMLE-4-options`, `openlifescienceai/medmcqa`.
