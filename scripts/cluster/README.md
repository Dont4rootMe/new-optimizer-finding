# Cluster jobs

This directory is the canonical non-interactive path for the single-process
EvolutionLoop on Cloud.ru ML Space. The scheduler job is `type="binary"` with
one 8×H100 worker: one coordinator owns a tensor-parallel SGLang server and all
task evaluators. The request sets `processes_per_worker=1`; omitting it makes
ML Space default to one MPI process per GPU. The shell entrypoint keeps a
nonzero-rank guard as defense in depth and assigns all eight GPUs to rank 0.
Do not remove either contract or use a `pytorch2` job for EvolutionLoop.

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

SR008 does **not** mount the Jupyter server's NFS tree into a job. By default
the submitter embeds a stdlib-only bootstrap that clones the public repository
over HTTPS into `/home/jovyan/evolutionloop-deepseek-v4/source/<full-commit>`,
fetches the exact commit, verifies `HEAD`, and only then executes the canonical
entrypoint. The regional checkout, model cache, environment, and run directory
survive later allocations and make retry/resume cheap. `--direct-shared-path`
is only for a separately proven cluster whose NFS really is shared.

Use `--dry-run` first when changing any scheduler property. The submitter writes
`submission_request.json`, then `submission.json` with the exact source commit,
regional paths, and stable job name.

## Observe and resume

```bash
bash "$MLS_ENV" "$MLS_PY" -m scripts.cluster.monitor \
  --job-name lm-mpi-job-... \
  --run-dir /absolute/nfs/path/optimizer_cluster_runs/deepseek-v4-flash-0731-circle-300
```

The monitor writes `monitor_status.json`, append-only `monitor_history.jsonl`,
and finally `completion_event.json`. While the job is running, parseable
`EVOLUTIONLOOP_EVENT` records expose generation/inflight/token progress through
scheduler logs. Population state and LLM telemetry remain on regional NFS;
rerunning the same run ID resumes an inflight seed or generation through the
canonical EvolutionLoop state protocol.

After a terminal job, export the regional run back to workspace NFS:

```bash
bash "$MLS_ENV" "$MLS_PY" -m scripts.cluster.transfer \
  --regional-run-dir /home/jovyan/evolutionloop-deepseek-v4/runs/<run-id> \
  --destination /home/jovyan/<workspace-path>/cluster-results/<run-id>
```

Each model call is logged to `population/llm_usage.jsonl`. A completed job also
writes `token_usage_summary.json`, grouped by route, stage, and generation.
