# Paper Source

Anonymous-review submission of *"Simulating Difficult Clients in Counseling
Roleplay via Activation Steering"*.

## Build

```bash
TEXINPUTS=./vendor/acl-style-files: latexmk -pdf -interaction=nonstopmode main.tex
```

Output: `main.pdf`. The compiled PDF is committed for reference.

## Layout

- `main.tex` — entry point; uses `\usepackage[review]{acl}`.
- `bib/persona_vectors.bib` — bibliography.
- `figures/` — auto-generated figures (`.pdf`, `.png`, TikZ).
- `tables/` — auto-generated LaTeX tables.
- `acl.sty`, `acl_natbib.bst`, `vendor/acl-style-files/` — vendored ACL/EMNLP
  style files.

## Regenerating figures and tables

From the repository root:

```bash
python scripts/build_paper_tables.py        # → paper/tables/*.tex
python scripts/build_paper_figures.py       # → paper/figures/*.pdf
python scripts/build_qualitative_figure_pdf.py
```

These scripts consume the JSON/JSONL artifacts under
`../outputs/steering_eval/` and `../outputs/human_eval/v3/`.
