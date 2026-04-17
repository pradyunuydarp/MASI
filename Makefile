PYTHON := python3
VENV_PYTHON := .venv/bin/python
PYTHONPATH_SRC := PYTHONPATH=src

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  data-demo                Run the synthetic Phase 1 preprocessing demo"
	@echo "  data-manifest            Build the synthetic preprocessing manifest"
	@echo "  amazon-download          Download the bounded Amazon CSJ review prefix"
	@echo "  masi-tokens              Build Phase 1 + Phase 2 MASI fused tokens"
	@echo "  train-smoke              Run the one-click MASI smoke pipeline"
	@echo "  train-full               Run the one-click MASI full CSJ pipeline"
	@echo "  report                   Build the LaTeX implementation note PDF with tectonic"
	@echo "  recommender-demo         Run the recommender demo on imported Amazon data"
	@echo "  recommender-demo-synth   Run the recommender demo on the old synthetic config"
	@echo "  compile                  Compile-check src/ and scripts/"

.PHONY: data-demo
data-demo:
	$(PYTHONPATH_SRC) $(PYTHON) scripts/demo_phase1_prep.py --config configs/data_prep_demo.json

.PHONY: data-manifest
data-manifest:
	$(PYTHONPATH_SRC) $(PYTHON) scripts/build_dataset_manifest.py --config configs/data_prep_demo.json

.PHONY: amazon-download
amazon-download:
	$(PYTHONPATH_SRC) $(PYTHON) scripts/download_amazon_csj_dataset.py

.PHONY: masi-tokens
masi-tokens:
	$(PYTHONPATH_SRC) $(VENV_PYTHON) scripts/build_masi_tokens.py --config configs/masi_tokens_amazon_csj_demo.json

.PHONY: train-smoke
train-smoke:
	$(PYTHONPATH_SRC) $(VENV_PYTHON) scripts/train_masi.py --config configs/masi_train_csj_smoke.json

.PHONY: train-full
train-full:
	$(PYTHONPATH_SRC) $(VENV_PYTHON) scripts/train_masi.py --config configs/masi_train_csj_full.json

.PHONY: report
report:
	tectonic --outdir docs docs/masi_implementation_note.tex

.PHONY: recommender-demo
recommender-demo:
	$(PYTHONPATH_SRC) $(VENV_PYTHON) scripts/demo_recommender_foundation.py --config configs/recommender_amazon_csj_demo.json

.PHONY: recommender-demo-synth
recommender-demo-synth:
	$(PYTHONPATH_SRC) $(VENV_PYTHON) scripts/demo_recommender_foundation.py --config configs/recommender_demo.json

.PHONY: compile
compile:
	$(PYTHONPATH_SRC) $(VENV_PYTHON) -m compileall src scripts
