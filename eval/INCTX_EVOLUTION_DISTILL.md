# In-Context Evolution Distillation Experiments

This guide sets up training-free experiments that use RL evolution traces as in-context demonstrations.

## 1) Expected rollout events format

Create a JSONL file where each line is one rollout event from a checkpoint.

Required fields:
- `problem_id` (or `id`/`unique_id`)
- `problem`
- `model_id` (checkpoint id, e.g. `ckpt_000`, `ckpt_001`)
- `generation`
- `reward`

Optional:
- `answer` (gold answer, useful for debugging)

Example:

```json
{"problem_id":"aime24_1","problem":"...","answer":"314","model_id":"ckpt_000","generation":"...","reward":0.0}
{"problem_id":"aime24_1","problem":"...","answer":"314","model_id":"ckpt_001","generation":"...","reward":0.0}
{"problem_id":"aime24_1","problem":"...","answer":"314","model_id":"ckpt_002","generation":"...","reward":1.0}
```

## 2) Run evolution-distilled eval

From repo root:

```bash
python eval/run_inctx_evolution_distill.py \
  --model_path ./models/Qwen_Qwen3-Coder-Next \
  --eval_data ./eval/aime24_test.jsonl \
  --rollout_events_path ./eval/rollout_events.jsonl \
  --trace_path ./eval/inctx_traces.jsonl \
  --output_path ./eval/inctx_evolution_results.jsonl \
  --num_shots 3 \
  --max_steps 5 \
  --temperature 0.0
```

If `--trace_path` already exists, the script loads it directly.

## 3) Baselines and ablations

### A. Zero-shot baseline
- Run your normal evaluator without evolution traces (existing script).

### B. Evolution-distill (main)
- Use `--num_shots 3` and `--max_steps 5`.

### C. Trace-count ablation
- Sweep `--num_shots` in `{1, 2, 3, 4}`.

### D. Chain-length ablation
- Sweep `--max_steps` in `{3, 5, 7}`.

### E. Negative-step ablation
- Compare default vs `--keep_negative_steps`.

## 4) Suggested report table

- Dataset
- pass@1
- #shots (`num_shots`)
- max steps (`max_steps`)
- include negative steps (yes/no)

This gives a direct demonstration of whether in-context evolution traces improve reasoning quality.