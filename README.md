# nlp_quantum_hub

> Reproducible pipeline for the paper **“Quantum Sentiment
> Classification with DisCoCat: A Controlled Empirical Study on a
> Short Phrase SST-2 Subset.”**
>
> A 60-run controlled study (2 readers × 2 ansatzes × 3 depths × 5
> seeds) of variational quantum circuits applied to binary sentiment
> classification on short phrases from the Stanford Sentiment Treebank,
> built on `lambeq` and the PennyLane `lightning.qubit` simulator and
> compared against three classical baselines (LR + TF-IDF, LR + GloVe,
> BiLSTM).

---

## Headline result

| Model                 | Test accuracy   | # Trainable parameters |
| --------------------- | --------------- | ---------------------- |
| LR + TF-IDF (best classical) | **94.25 %** | 2,829                  |
| **Q-spiders (any ansatz, any depth)** | **92.05 %** | **2,940** |
| BiLSTM (1 layer, 16-d) | 91.83 %        | 82,369                 |
| LR + GloVe (300-d)    | 86.50 %         | 90,301                 |
| Q-cups-Sim14-L₃        | 91.85 %        | 8,920                  |
| Q-cups-IQP-L₂ (barren-plateau collapse) | 62.80 % | 5,200 |

McNemar paired tests on the 400-phrase test set confirm that
**Q-spiders is statistically indistinguishable from BiLSTM**
(*p* = 0.60) while using **28× fewer parameters**, and it trails the
sparse linear baseline LR + TF-IDF by 2.2 points (*p* = 0.045).

Two structural phenomena are documented as part of the same grid:

1. **Representational saturation of the Spiders reader** under
   single-qubit atomic types — all six ansatz/depth combinations
   collapse to the same accuracy.
2. **Reproducible barren-plateau collapse** of `Cups + IQP` at depth 2
   (62.8 % across all 5 seeds), surviving the
   `ReduceLROnPlateau` scheduler and unit-norm gradient clipping.

See `papers/ajdsai/main.pdf` (or the LNCS variant in `papers/lncs/`)
for the full write-up.

---

## Repository layout

```
nlp_quantum_hub/
├── README.md            ← you are here
├── LICENSE              ← MIT
├── CITATION.cff         ← machine-readable citation metadata
├── requirements.txt     ← Python dependencies
├── Makefile             ← one-command reproduction (`make all`)
│
├── scripts/             ← pipeline drivers, numbered 00–09
│   ├── 00_download_glove.py        – fetch GloVe vectors (baselines/lr_glove)
│   ├── 00_fix_lambeq_url.py        – patch dead lambeq endpoint (see §5.5)
│   ├── 01_download_sst.py          – fetch Stanford SST corpus
│   ├── 02_prepare_data.py          – parse trees, extract phrases
│   ├── 02b_subsample_qnlp.py       – build 5000/400/400 balanced split (seed 42)
│   ├── 03_eda.py                   – corpus statistics + plots
│   ├── 04_train_baselines.py       – LR+TF-IDF, LR+GloVe, BiLSTM
│   ├── 05_parse_diagrams.py        – DisCoCat diagrams via SpidersReader / CupsReader
│   ├── 06_train_quantum.py         – 60-run quantum grid
│   ├── 07_evaluate_test.py         – score every config on the held-out 400 phrases
│   ├── 08_final_analysis.py        – aggregate plots + base McNemar pairs
│   ├── 08b_extra_mcnemar.py        – extra McNemar pairs cited in the paper
│   ├── 09_extra_figures.py         – appendix figures
│   └── 99_demo.py                  – minimal end-to-end demo
│
├── src/                 ← reusable modules
│   ├── baselines/       – LR / GloVe / BiLSTM implementations
│   ├── discocat/        – diagram parsing + visualisation
│   ├── circuits/        – ansatz templates (IQP, Sim14)
│   ├── quantum/         – training loop, ansatz factory, data layer
│   ├── preprocessing/   – SST tokeniser + vocabulary
│   ├── training/        – shared training utilities
│   └── evaluation/      – metric helpers
│
├── data/                ← raw and processed corpora (gitignored)
│   ├── raw/             – Stanford SST + GloVe (downloaded by scripts/01,00)
│   └── processed/       – qnlp 5000/400/400 split (built by scripts/02b)
│
├── results/             ← experiment artefacts
│   └── final/
│       ├── summary_test.json           – aggregate metrics (paper Tables 1–3)
│       ├── summary_test.txt            – human-readable summary
│       ├── test_metrics_classical.json – per-seed classical metrics
│       ├── test_metrics_quantum.json   – per-seed quantum metrics (60 runs)
│       ├── mcnemar_results.txt         – pairwise McNemar tests (paper Table 4)
│       ├── predictions/                – per-config per-seed CSV predictions on test set
│       └── plots/                      – every figure that appears in the paper
│
└── papers/              ← LaTeX sources for both submission variants
    ├── lncs/            – Springer LNCS format (24 p)
    └── ajdsai/          – Asian Journal of Data Science and AI (Emerald, 20 p)
        └── cover_letter.txt
```

---

## Quick-start

```bash
git clone https://github.com/hublinhdn/nlp_quantum_hub.git
cd nlp_quantum_hub

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Reproduce everything end-to-end (≈ 6–10 h on a single CPU)
make all

# Or, run individual stages
make data        # download + split          (≈ 10 min)
make baselines   # train classical models    (≈ 20 min)
make quantum     # 60-run quantum grid       (≈ 5–9 h)
make evaluate    # score the held-out test   (≈ 5 min)
make analysis    # plots + McNemar tests     (≈ 2 min)
make papers      # build both LaTeX papers   (≈ 1 min)
```

If you only want to inspect the precomputed results: every artefact
needed to regenerate the paper figures and tables is already committed
under `results/final/`. Run `make analysis` and `make papers` to rebuild
plots and PDFs without re-training.

---

## Reproducing individual figures and tables

| Paper element                  | Generated by                                |
| ------------------------------ | ------------------------------------------- |
| Table 1 (corpus statistics)    | `scripts/03_eda.py`                         |
| Table 2 (classical baselines)  | `scripts/04_train_baselines.py` + `07`      |
| Table 3 (60-run quantum grid)  | `scripts/06_train_quantum.py` + `07`        |
| **Table 4 (McNemar)**          | `scripts/08_final_analysis.py` + `08b`      |
| Figure 1 (DisCoCat pipeline)   | `scripts/05_parse_diagrams.py` (example)    |
| Figure 2 (sentence diagrams)   | `scripts/05_parse_diagrams.py`              |
| Figure 3 (training curves)     | `scripts/08_final_analysis.py`              |
| Figure 4 (per-length accuracy) | `scripts/08_final_analysis.py`              |
| Figure 5 (confusion matrices)  | `scripts/08_final_analysis.py`              |
| Figure 6 (depth-2 collapse)    | `scripts/08_final_analysis.py`              |
| Figures 7–12 (appendix)        | `scripts/09_extra_figures.py`               |

Pre-built artefacts in `results/final/plots/` mirror what each script
produces; running the script overwrites the corresponding PNG in place.

---

## Environment notes

- **Python**: developed and tested on **3.10**. The PennyLane 0.34 +
  `lightning.qubit` combination is stable on macOS arm64 (Apple Silicon)
  and Linux x86_64.
- **`lambeq` version**: pinned to `>=0.4.3, <0.6`. The BobcatParser
  endpoint shipped with `lambeq` 0.5 has been deprecated; this project
  works around the issue by relying on the self-contained
  `SpidersReader` and `CupsReader` only. See
  `scripts/00_fix_lambeq_url.py` and **§5.5** of the paper for the full
  story.
- **Hardware**: every experiment runs on a single CPU. No GPU or
  quantum device is required. The 60-run grid is ≈ 5–9 hours of
  wall-clock on a 2024-era laptop.

---

## Citation

If you use this code or build on the experiments, please cite the
accompanying paper:

```bibtex
@article{doletran2026qnlp,
  title  = {Quantum Sentiment Classification with {DisCoCat}:
            A Controlled Empirical Study on a Short Phrase
            {SST-2} Subset},
  author = {Do, Nhat-Linh and Le, Quang-Thai and Tran-Huynh, Minh-Tan},
  journal = {Asian Journal of Data Science and Artificial Intelligence},
  year   = {2026},
  note   = {Under review}
}
```

A `CITATION.cff` file is also provided for tools that consume the
Citation File Format (GitHub auto-recognises it on the repository
landing page).

---

## Acknowledgements

The authors thank Dr. Le Quang Minh of Ho Chi Minh City Open
University, instructor of the Natural Language Processing course, for
guidance on the direction of this work.

---

## Licence

This project is released under the **MIT licence** — see
[`LICENSE`](./LICENSE) for the full text. The Stanford SST corpus and
GloVe vectors retain their original licences (Stanford NLP terms).
