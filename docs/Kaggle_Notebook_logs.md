# Kaggle Notebook Logs

This log captures operational lessons from getting `notebooks/08_full_dataset_pipeline.ipynb` running on Kaggle with the prepared `masi-amazon-csj-full-dataset` input. It is intended as a handoff reference for future agents and as source material for documenting implementation challenges.

## Current Kaggle Run Contract

- Entry notebook: `notebooks/08_full_dataset_pipeline.ipynb`
- Source config: `configs/Full_dataset.json`
- Kaggle runtime config: written at `/kaggle/working/masi_artifacts/configs/Full_dataset.kaggle_safe_runtime.json`
- Kaggle storage root: `/kaggle/working/masi_artifacts`
- Default Kaggle run root: `/kaggle/working/masi_artifacts/outputs/amazon_csj_full_dataset_kaggle_long_safe_train`
- Attached dataset expected at: `/kaggle/input/datasets/dheerajrajanala/masi-amazon-csj-full-dataset`
- Required attached dataset entries:
  - `Clothing_Shoes_and_Jewelry.jsonl`
  - `meta_Clothing_Shoes_and_Jewelry.jsonl`
  - `images/`
  - `image_download_manifest.json`
  - `subset_manifest.json`

The notebook intentionally runs a bounded slice from the prepared full dataset on Kaggle. The current implementation keeps CLIP embeddings in memory, so the proposal-scale `Full_dataset.json` caps are too large for one Kaggle session.

Current Kaggle-safe profile support:

- `smoke_safe`: preserves the previously validated bounded run used by the saved Kaggle output notebook.
- `scaled_safe`: the previously completed metric-improvement profile, still bounded because CLIP embeddings are kept in memory.
- `long_safe`: the recommended continuation metric-improvement profile; it now restores the previous bundle and runs a larger bounded slice.

Previously observed `smoke_safe` caps:

- `max_users = 512`
- `max_items = 1024`
- `max_review_records = 2_000_000`
- `clip.batch_size = 8`
- `alignment.batch_size = 128`
- `tokenization.batch_size = 128`
- `tokenization.epochs = 4`
- `experiment.batch_size = 16`
- `experiment.history_max_tokens = 96`
- `experiment.mlm_epochs = 1`
- `experiment.autoregressive_epochs = 2`

Completed `scaled_safe` caps:

- `max_users = 4096`
- `max_items = 8192`
- `max_review_records = 20_000_000`
- `clip.batch_size = 16`
- `alignment.batch_size = 256`
- `alignment.epochs = 5`
- `tokenization.batch_size = 256`
- `tokenization.epochs = 15`
- `experiment.batch_size = 32`
- `experiment.history_max_tokens = 128`
- `experiment.mlm_epochs = 5`
- `experiment.autoregressive_epochs = 15`
- `experiment.hidden_dim = 256`
- `experiment.num_heads = 8`
- `experiment.num_layers = 4`

Current recommended `long_safe` continuation caps:

- `max_users = 12288`
- `max_items = 24576`
- `max_review_records = 50_000_000`
- `clip.batch_size = 16`
- `alignment.batch_size = 256`
- `alignment.epochs = 10`
- `alignment.learning_rate = 0.0004`
- `alignment.hard_negative_count = 24`
- `alignment.window_size = 4`
- `tokenization.batch_size = 256`
- `tokenization.epochs = 30`
- `tokenization.learning_rate = 0.0004`
- `experiment.batch_size = 32`
- `experiment.history_max_tokens = 160`
- `experiment.mlm_epochs = 10`
- `experiment.autoregressive_epochs = 30`
- `experiment.learning_rate = 0.00025`
- `experiment.hidden_dim = 256`
- `experiment.num_heads = 8`
- `experiment.num_layers = 4`
- `experiment.max_eval_candidates = 1024`
- `experiment.eval_candidate_batch_size = 128`

Set `USE_KAGGLE_SAFE_LIMITS = False` only on a larger machine or after adding sharded CLIP embedding persistence.

Resume automation added after the first scale-up pass:

- `AUTO_RESTORE_RESUME_BUNDLE = True` scans attached Kaggle inputs for a previous exported zip or unzipped bundle.
- The bundle is restored into `/kaggle/working/masi_artifacts/outputs/amazon_csj_full_dataset_kaggle_long_safe_train`.
- The runtime config sets `checkpointing.restore_from_checkpoints = true`, so Phase 1, Phase 2, MLM, and autoregressive checkpoints are loaded before another continuation run.
- Restored runs advance `data_chunk_index` and set `dataset.user_rank_offset = data_chunk_index * max_users`, so each continuation trains on a different bounded user-rank chunk while preserving the same run root for checkpoint compatibility.
- The restore helper ranks final and periodic checkpoint candidates by modification time, then `global_step` when available, so an interrupted continuation can resume from the latest retained step checkpoint instead of falling back to an older final checkpoint.
- The export cell writes `resume_bundle_manifest.json` and packages the full run root, including final checkpoints, retained periodic checkpoints, fused IDs, resolved configs, summaries, manifests, and `data_chunk` state.

The saved output notebook `Kaggle_interactions/masi-full-dataset.ipynb` produced warm `HR@10 = 0.08262` and cold `HR@10 = 0.0` under the `scaled_safe` profile. The older smoke-scale result was warm `HR@10 = 0.04248` and cold `HR@10 = 0.0`. See `docs/kaggle_full_dataset_scaleup_journey.md` for the interpretation and the next scale-up plan.

## Issues Encountered And Fixes

### Rerunning The First Cell Deleted The Active Working Directory

Observed error:

```text
fatal: Unable to read current working directory: No such file or directory
```

Cause:

- The kernel current working directory could be `/kaggle/working/MASI`.
- The setup cell removed `/kaggle/working/MASI` with `shutil.rmtree(REPO_DIR)`.
- `git clone` then started from a deleted current directory.

Fix:

- Before deleting the repo checkout, the notebook now runs `os.chdir(KAGGLE_WORKING_ROOT)`.
- `subprocess.run(..., cwd=KAGGLE_WORKING_ROOT)` is used for `git clone`.

### Missing `FORCE_PREPARE`

Cause:

- The prepare cell referenced `FORCE_PREPARE`, but the first setup cell did not define it.

Fix:

- Added `FORCE_PREPARE = False` beside the other runtime toggles.

### NumPy Conflict On Kaggle

Observed warning:

```text
tensorflow requires numpy<2.2.0,>=1.26.0
numba requires numpy<2.1,>=1.22
```

Cause:

- `pyproject.toml` originally allowed/required a too-new NumPy version for Kaggle's preinstalled stack.

Fix:

- The `recommender` extra now uses `numpy>=1.26,<2.1`.
- The notebook install cell also patches older cloned repos before installing, so Kaggle runs are protected even before all notebook-side changes are pushed and pulled.

### Hugging Face Unauthenticated Warning And Token Handling

Observed warning:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

Fix:

- Add a Kaggle secret named exactly `HF_TOKEN`.
- The notebook reads it with `UserSecretsClient().get_secret("HF_TOKEN")`.
- The token is passed to both the notebook process and child subprocesses via:
  - `HF_TOKEN`
  - `HUGGING_FACE_HUB_TOKEN`

Security note:

- Never paste Hugging Face tokens into notebook code, docs, commits, or chat.
- If a token is exposed, revoke it in Hugging Face settings and create a new read token.

### CLIP Download Or Loading Appeared Stuck

Observed output:

```text
Loading weights: ... Materializing param=...
```

Learnings:

- The Hugging Face warning itself is not an error.
- The progress line means `transformers` is loading CLIP checkpoint weights, not yet running MASI training.
- Browser sleep/lock does not directly stop the Kaggle VM, but it can freeze visible output or lead to later session reclamation.

Fixes added:

- The notebook sets a writable Hugging Face cache:
  - `HF_HOME=/kaggle/working/masi_artifacts/hf_cache`
  - `HF_HUB_CACHE=/kaggle/working/masi_artifacts/hf_cache/hub`
  - `TRANSFORMERS_CACHE=/kaggle/working/masi_artifacts/hf_cache/transformers`
- The notebook disables Xet transfers with `HF_HUB_DISABLE_XET=1`.
- The notebook preloads CLIP before the long training command so download/load failures happen early.
- The notebook writes a standalone CLIP directory under `/kaggle/working/masi_artifacts/hf_models/openai_clip-vit-base-patch32` and creates a zip beside it when running on Kaggle.
- Publish that zip as a private Kaggle Dataset/Model and attach it to later sessions; the notebook auto-detects attached CLIP directories under `/kaggle/input` by looking for `config.json`, `preprocessor_config.json`, and model weights.
- The CLIP loader is patched to use:

```python
low_cpu_mem_usage=False
use_safetensors=True
```

This avoids the higher-risk meta-tensor materialization path in the training subprocess.

### Long Training Output Is Hard To Read

Current behavior:

- The full-dataset notebooks clear stale volatile outputs and run pip installs with quiet flags.
- CLIP encoding, behavior alignment, text/vision RQ-VAE training, cross-modal MLM, and autoregressive fine-tuning emit bounded tqdm percentage bars rather than a line per step.
- Kaggle-safe profiles set periodic checkpoint saves to every 25 optimizer steps. Each periodic checkpoint directory has a `latest.json` manifest, and the notebook checkpoint-inspection cell prints the latest retained path for each stage.

### CLIP Loaded But `build_masi_tokens.py` Was Killed

Observed error:

```text
subprocess.CalledProcessError: ... build_masi_tokens.py ... died with <Signals.SIGKILL: 9>
```

Cause:

- `SIGKILL: 9` from Kaggle is consistent with an out-of-memory or resource kill.
- The original `Full_dataset.json` asks for up to `102400` users and `204800` items.
- `build_masi_tokens.py` currently stores full text and image embedding dictionaries in memory before downstream alignment and quantization.

Fix:

- The notebook now derives a Kaggle-safe runtime config from `configs/Full_dataset.json` by default.
- The source proposal-scale config remains unchanged for local or larger-machine work.

Future improvement:

- Implement sharded CLIP embedding extraction and persisted embedding shards.
- Stream or memory-map embeddings into Phase 1 and Phase 2.
- Add a manifest-based resume path for token building so a Kaggle session can continue from partial embeddings.

## Validation And Outputs

Validation is present in Phase 3, after token building and recommender training finish.

The pipeline writes:

- run manifest: `RUN_ROOT / "run_manifest.json"`
- token summary: `RUN_ROOT / "phase12_tokens" / "masi_token_summary.json"`
- experiment summary: `RUN_ROOT / "phase3_experiment" / "experiment_summary.json"`
- export bundle: `/kaggle/working/masi_artifacts/outputs/amazon_csj_full_dataset_train_artifacts.zip` or the run-name equivalent

The evaluation summary contains:

- `warm_metrics`
- `cold_metrics`
- `HR@10`
- `NDCG@10`
- `Coverage@10`
- average inference latency
- number of evaluated examples

The warm/cold split is deterministic leave-one-out with `cold_start_ratio` from the experiment config.

## Recommended Future Build Rules

- Keep Kaggle notebooks bounded unless sharded preprocessing exists.
- Treat `/kaggle/working` as ephemeral and export bundles before ending a session.
- Use Kaggle Secrets for `HF_TOKEN`; never hardcode credentials.
- Keep the clone step rerun-safe by changing out of the repo before deleting it.
- Preserve both source config and runtime config in logs so experiment scale is explicit.
- Any change to setup flow should update this log, `TODO_TASKS.md`, and `README.md` if user-facing.
