# Reproducing the COPS Pipeline

This file documents end-to-end reproduction of the experiments reported in
*"Simulating Difficult Clients in Counseling Roleplay via Activation Steering"*.
For a "tables and figures only" rebuild from the included JSON/JSONL artifacts,
see [README.md](README.md) — that path takes minutes and needs no GPU or API
key. The stages below are only required to regenerate the model outputs and
judge scores themselves.

## 0. Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # full env, ~12 GB of wheels (torch, unsloth, transformers)
# OR: pip install -r requirements-steering.txt for inference-only (no unsloth/trl)
```

API keys required:
```bash
export OPENAI_API_KEY=...           # for context generation (GPT-5.4-mini) and one judge
export MISTRAL_API_KEY=...          # for Mistral-Magistral-Small-2509 judge
export IONOS_API_KEY=...            # for GPT-OSS-120B judge (hosted via IONOS)
export CLUSTER_HOME=/path/on/cluster # SLURM working dir (used in scripts/cops_submit_*.sh)
```

The held-out evaluation indices (200–399 of `cops_contexts.jsonl`) are
seed-fixed; re-running stages 3–6 on the same contexts is deterministic up to
provider-side non-determinism in the LLM judges.

## 1. Build contexts

```bash
python scripts/build_cops_contexts.py \
    --config configs/persona_axes/client_persona_axes_v1.json \
    --output data/processed_anonymized/cops_contexts.jsonl
```

This produces ~5400 (persona, context, target_label) triplets. The first 200
indices are used to fit per-persona steering vectors; 200–399 are held out for
evaluation.

> The included `cops_contexts.jsonl` is already PII-filtered (using
> `openai/privacy-filter`, revision `7ffa9a04`).

## 2. Hidden-state extraction & steering vectors (GPU, SLURM)

```bash
# Layer sweep — fits steering vectors for every transformer layer.
bash scripts/cops_submit_layer_sweep.sh
# Writes hidden-state caches and per-layer vectors under
# outputs/steering_eval/layer_sweep/
```

GPU budget: ~1.5 h per model on a single A40 (4 models × 4 axes).

## 3. Steering inference

```bash
# Headline matrix (canonical α grid, all 4 models × 4 axes, both poles).
bash scripts/cops_submit_paper_matrix_v2.sh

# Negative-alpha sweep for Audit 1.
bash scripts/cops_submit_paper_matrix_v2_negalpha.sh

# Prompt baseline.
bash scripts/cops_submit_prompt_baseline.sh

# Persona-aware variants (Audit 2).
bash scripts/cops_submit_persona_variants.sh
```

GPU budget: ~4–6 h per `cops_submit_*` script. Outputs land under
`outputs/steering_eval/paper_v2*/` and `outputs/steering_eval/persona_variants/`.

## 4. Three-provider LLM judge

```bash
python scripts/cops_judge_steering_eval.py \
    --inputs outputs/steering_eval/paper_v2_negalpha \
    --providers cluster_mistral openai ionos \
    --out outputs/steering_eval/judged_v2_negalpha
python scripts/cops_judge_ensemble.py \
    --judged outputs/steering_eval/judged_v2_negalpha \
    --out outputs/steering_eval/judged_v2_negalpha_ensemble
```

Judge calls use SHA-256 caching keyed on `(prompt, model, temperature)`; the
cache lives in `outputs/steering_eval/judge_cache/` (not shipped — regenerate
on first run). API budget: ~$80 USD per full headline pass.

## 5. OnCoCo content classification

```bash
# On the cluster:
sbatch slurm/generated/classify_oncoco_kiz0.sh
# Locally (single A40, ~12 min):
python scripts/classify_responses_oncoco.py \
    --inputs outputs/steering_eval/paper_v2_negalpha \
    --out outputs/steering_eval/oncoco_labels.jsonl
python scripts/analyze_oncoco_label_shift.py
```

## 6. Inter-rater reliability

```bash
python scripts/compute_irr.py \
    --studio outputs/human_eval/v3/studio_export.json \
    --summary outputs/human_eval/v3/irr_summary.json \
    --out paper/tables
```

Human studio export contains 360 ratings across 4 raters (anonymized
`RaterA`–`RaterD`), corresponding to 124 items with ≥3 raters.

## 7. Build paper tables and figures

```bash
python scripts/build_paper_tables.py
python scripts/build_paper_figures.py
python scripts/build_qualitative_figure_pdf.py   # Figure 1 (Luisa example)
```

This consumes the JSON/JSONL files under `outputs/steering_eval/judged*` and
emits LaTeX files under `paper/tables/` and PDFs under `paper/figures/`.

## 8. Compile the PDF

```bash
cd paper
TEXINPUTS=./vendor/acl-style-files: latexmk -pdf -interaction=nonstopmode main.tex
```

## Compute budget summary

| Stage | Hardware | Time | API cost |
|---|---|---|---|
| 1. Context generation | CPU + GPT-5.4-mini | ~30 min | ~$15 |
| 2. Hidden-state extraction | 1× A40 | ~6 h | – |
| 3. Steering inference | 1× A40 per submit script | ~16–24 h total | – |
| 4. 3-provider judging | CPU | ~3 h | ~$80 |
| 5. OnCoCo classification | 1× A40 | ~12 min | – |
| 6. IRR | CPU | < 1 min | – |
| 7. Tables/Figures | CPU | < 5 min | – |
| 8. LaTeX | CPU | < 1 min | – |

## Troubleshooting

- **`latexmk` missing fonts:** install TeX Live with `texlive-fonts-recommended`
  or use Overleaf with the `paper/` folder as the project root.
- **Hugging Face gated model:** ensure `huggingface-cli login` for Gemma; set
  `HF_TOKEN` for non-interactive use.
- **SLURM scripts assume KIZ0 partitions** (`p6`, `p7`); adapt `--partition`
  and `--account` flags to your cluster.
