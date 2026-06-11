# =====================================================================
# nlp_quantum_hub — top-level Makefile
#
# Common targets:
#   make setup        Install Python deps into the active venv
#   make data         Download SST + GloVe, build qnlp 5000/400/400 subset
#   make baselines    Train LR+TF-IDF, LR+GloVe, BiLSTM (5 seeds each)
#   make quantum      Train all 60 quantum runs (reader x ansatz x depth x seed)
#   make evaluate     Score test set, write summary_test.json
#   make analysis     Plots + McNemar tests + final figures
#   make papers       Build both LaTeX papers (LNCS + AJDSAI)
#   make all          Full pipeline end-to-end (≈ 6-10 h on CPU)
#   make clean        Remove derived artefacts (keeps data/raw)
#
# Hyperparameters and seeds match the values reported in the paper
# (5 seeds: 0-4; 60-run grid; balanced 5000/400/400 split with seed 42).
# =====================================================================

PY ?= python

.PHONY: setup data baselines quantum evaluate analysis papers all clean help

help:
	@grep -E '^# +make ' Makefile | sed 's/^# *//'

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/01_download_sst.py
	$(PY) scripts/00_download_glove.py
	$(PY) scripts/02_prepare_data.py
	$(PY) scripts/02b_subsample_qnlp.py --seed 42 --train 5000 --dev 400 --test 400
	$(PY) scripts/03_eda.py

baselines:
	$(PY) scripts/04_train_baselines.py --seeds 0 1 2 3 4

quantum:
	$(PY) scripts/05_parse_diagrams.py
	$(PY) scripts/06_train_quantum.py --readers spiders cups --ansatzes iqp sim14 --depths 1 2 3 --seeds 0 1 2 3 4

evaluate:
	$(PY) scripts/07_evaluate_test.py

analysis:
	$(PY) scripts/08_final_analysis.py
	$(PY) scripts/08b_extra_mcnemar.py
	$(PY) scripts/09_extra_figures.py

papers:
	$(MAKE) -C papers/lncs pdf-only
	$(MAKE) -C papers/ajdsai pdf-only

all: setup data baselines quantum evaluate analysis papers

clean:
	rm -rf results/baseline results/quantum results/diagrams results/eda
	find . -name '__pycache__' -type d -exec rm -rf {} +
	$(MAKE) -C papers/lncs clean 2>/dev/null || true
	$(MAKE) -C papers/ajdsai clean 2>/dev/null || true
