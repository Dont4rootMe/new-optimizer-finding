# Cluster jobs

This directory is the canonical non-interactive path for the single-process
EvolutionLoop on Cloud.ru ML Space. The scheduler job is `type="binary"` with
one 8×H100 worker: one coordinator owns a tensor-parallel SGLang server and all
task evaluators. This ML Space allocation still invokes a binary command once
per GPU, so the shell entrypoint exits nonzero ranks and assigns all GPUs to
rank 0. Do not remove that guard or use an unguarded `pytorch2` job.

The production profile serves the exact
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`
snapshot with SGLang 0.5.16, TP=8, checkpoint-bundled DSpark, BF16 compressed
attention state, chunked prefill, and at most eight concurrent requests. On
H100 the stock FP4 checkpoint must use SGLang's Hopper W4A16/Marlin path; never
force a Blackwell-only MXFP4 backend.

## Submit

Run through the environment bridge documented in
`agents/remote_cluster_access.md`:

```bash
bash "$MLS_ENV" "$MLS_PY" -m scripts.cluster.submit \
  --project-root /absolute/nfs/path/to/this-branch-clone \
  --run-id deepseek-v4-flash-0731-circle-300
```

Use `--dry-run` first when changing any scheduler property. The submitter writes
`submission_request.json`, then `submission.json` with the stable job name.

## Observe and resume

```bash
bash "$MLS_ENV" "$MLS_PY" -m scripts.cluster.monitor \
  --job-name lm-mpi-job-... \
  --run-dir /absolute/nfs/path/optimizer_cluster_runs/deepseek-v4-flash-0731-circle-300
```

The monitor writes `monitor_status.json`, append-only `monitor_history.jsonl`,
and finally `completion_event.json`. Population state and LLM telemetry remain
under `population/`; rerunning the same job/run directory resumes an inflight
seed or generation through EvolutionLoop's canonical state protocol.

Each model call is logged to `population/llm_usage.jsonl`. A completed job also
writes `token_usage_summary.json`, grouped by route, stage, and generation.
