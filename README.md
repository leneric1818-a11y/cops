# COPS Pipeline — Code & Data Release (Anonymous Review)

Companion repository for the anonymous submission
**"Simulating Difficult Clients in Counseling Roleplay via Activation Steering"**.

This repository contains the source code, configuration, anonymized data,
generated outputs (judge-ensemble labels + OnCoCo classifier outputs), and
the LaTeX source of the paper. Together these allow (1) rebuilding the paper
PDF from source, (2) regenerating all tables and figures from the included
JSON/JSONL artifacts, and (3) re-running the full pipeline from raw contexts.

## Repository layout

```
.
├── README.md                Quick-start (this file)
├── REPRODUCE.md             Step-by-step pipeline reproduction
├── LICENSE                  MIT (code in src/ and scripts/)
├── LICENSE-DATA             CC BY-NC 4.0 (data/, outputs/, paper/figures+tables)
├── requirements.txt         Full env (training + steering)
├── requirements-steering.txt  Lightweight env (steering inference only)
├── pyproject.toml           Package metadata
├── src/llm_ft_comparison/   Python package: pipelines, evaluation, models
├── scripts/                 Pipeline scripts (data prep, steering, judging, ablations, figure/table builders)
├── configs/                 Persona axes, benchmarks, model matrices, taxonomies, cluster templates
├── data/processed_anonymized/cops_contexts.jsonl  400 roleplay contexts (indices 0–399; 200–399 are held-out)
├── outputs/
│   ├── steering_eval/       Inference outputs, per-provider judge scores, ensemble summaries, OnCoCo labels
│   └── human_eval/v3/       Human studio export + IRR summary (raters anonymized as RaterA/B/C/D)
├── docs/                    Design notes (COPS_DESIGN, COPS_METHODS_V2, COPS_RESULTS_PAPER_V2, paper_section_oncoco_DRAFT)
└── paper/
    ├── main.tex             ACL/EMNLP submission source
    ├── main.pdf             Compiled PDF
    ├── acl.sty, acl_natbib.bst, vendor/acl-style-files/
    ├── bib/persona_vectors.bib
    ├── figures/, tables/    Auto-generated assets
    └── README.md            Paper-only build notes
```

## Quick start: rebuild the paper

```bash
cd paper
TEXINPUTS=./vendor/acl-style-files: latexmk -pdf main.tex
# Output: paper/main.pdf
```

The repository ships with the precompiled `paper/main.pdf` for reference.

## Quick start: rebuild tables & figures from included outputs

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-steering.txt
pip install pandas numpy scipy matplotlib seaborn statsmodels

python scripts/build_paper_tables.py
python scripts/build_paper_figures.py
```

Outputs are written into `paper/tables/*.tex` and `paper/figures/*.pdf`.

## Full pipeline reproduction

See [REPRODUCE.md](REPRODUCE.md) — covers environment setup, the eight pipeline
stages (context generation → steering vectors → inference → judging → OnCoCo
content classification → tables/figures → LaTeX), required compute budget, and
the SLURM-template scripts under `scripts/cops_submit_*.sh`.

## Models

All open-weight models referenced in the paper are loadable from the Hugging
Face Hub:

- `Qwen/Qwen3-4B`
- `google/gemma-4-E4B-it`
- `Qwen/Qwen3.5-9B`
- `mistralai/Mistral-7B-Instruct-v0.3`

OnCoCo content classifier: `xlm-roberta-large-OnCoCo-DE-EN` (68-label client/
counselor head). PII filter: `openai/privacy-filter` (rev. `7ffa9a04`).

## Anonymization notes

- Author/affiliation information is removed from all code and configuration.
- SLURM submission scripts use `${CLUSTER_HOME}` / `${USER}` placeholders.
- The four human raters are anonymized as `RaterA`, `RaterB`, `RaterC`, `RaterD`
  consistently across `outputs/human_eval/v3/`, `scripts/compute_irr.py`, and
  any LaTeX table referencing per-rater statistics.
- Counseling contexts in `data/processed_anonymized/cops_contexts.jsonl` were
  PII-filtered with `openai/privacy-filter` prior to release (see
  `LICENSE-DATA`).
- Bibliography keys in `paper/bib/persona_vectors.bib` reference prior work in
  third person, consistent with ACL anonymous-submission policy.

## License

- **Code** (`src/`, `scripts/`, `configs/`): MIT — see [LICENSE](LICENSE).
- **Data & outputs** (`data/`, `outputs/`, `paper/figures/`, `paper/tables/`):
  CC BY-NC 4.0 — see [LICENSE-DATA](LICENSE-DATA).
