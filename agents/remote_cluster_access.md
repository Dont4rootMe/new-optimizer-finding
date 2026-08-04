# Remote Cloud.ru H100 job access

Last verified: 2026-08-04 (Europe/Moscow). Never copy private-key contents,
gateway variables, API keys, or model tokens into the repository or logs.

## SSH entry point

```bash
ssh echimbulatov-text-diff.ai0001071-02601@ssh-a100-jupyter.ai.cloud.ru \
  -p 2222 -i ~/.ssh/sber_cloud_keys/mlspace_bayes_private_key.txt
```

Observed environment:

- remote user/home: `jovyan`, `/home/jovyan`;
- bundled Python: `/home/user/conda/bin/python`;
- scheduler library: `client_lib` 0.6.3 (ML Space, not Slurm);
- the interactive Jupyter node's A100 is not the requested job worker.

## SSH environment bridge

A plain SSH shell does not inherit ML Space gateway variables. Use the existing
bridge; it extracts only the required variables from the live Jupyter process
without printing their values:

```bash
MLS_ENV=/home/jovyan/echimbulatov/fork_afedorov/constant_repos/LatentDiffusion/with_mls_env.sh
MLS_PY=/home/user/conda/bin/python
bash "$MLS_ENV" "$MLS_PY" -c 'import client_lib; client_lib.jobs(region="SR008")'
```

If that helper moves, preserve its narrow behavior: import only `GWAPI_ADDR`,
`GWAPI_KEY`, `GWAPI_V2_URL`, `CLUSTER_KEY`, `NB_PREFIX`, `NAMESPACE`, and
`WORKSPACE_ID` from the Jupyter process; do not persist or echo their values.

## 8×H100 worker

Cloud.ru exposes physical H100s under the historical `a100plus.*` prefix:

```text
region:        SR008
instance_type: a100plus.8gpu.80vG.96C.1456G
hardware:      8 x NVIDIA H100 80 GB, 96 CPU, 1456 GiB RAM
alternate:     a100plus.8gpu.80vG.96C.1952G
```

Availability is dynamic. Check immediately before submitting:

```bash
bash "$MLS_ENV" "$MLS_PY" - <<'PY'
import client_lib
from rich.console import Console
Console().print(client_lib.get_available_resources_count(region="SR008"))
PY
```

At the 2026-08-04 check both 8-GPU SKUs had zero immediately free workers;
smaller H100 allocations were free. A queued job may therefore stay Pending.

## Canonical submit path

Use the separate finalization checkout on the Jupyter control plane; do not
switch or edit the author's checkout. A 2026-08-04 probe proved that an SR008
job sees its own `/home/jovyan` but **none** of the Jupyter paths below
`/home/jovyan/echimbulatov`. The submitter therefore sends a one-line,
base64-wrapped stdlib bootstrap which clones and verifies the exact source
commit in persistent regional NFS:

```bash
bash "$MLS_ENV" "$MLS_PY" -m scripts.cluster.submit \
  --project-root /home/jovyan/echimbulatov/fork_afedorov/constant_repos/new-optimizer-finding/runs/finalize-evolutionloop-deepseek-v4 \
  --run-id deepseek-v4-flash-0731-circle-300
```

Runtime layout:

```text
/home/jovyan/evolutionloop-deepseek-v4/
  source/<full-git-commit>/
  runtime/sglang-0.5.16-cu126/
  toolchains/cuda-nvcc-12.9.86/
  kernel_cache/deep_gemm-sm90-cuda-nvcc-12.9.86/
  kernel_cache/tvm-ffi-sm90-cuda-nvcc-12.9.86-tvmffi-0.1.11/
  model_cache/huggingface/
  runs/<run-id>/
```

The runtime name is a compatibility boundary, not decoration. Never reuse the
legacy `runtime/sglang-0.5.16/` CUDA-13 environment on the heterogeneous SR008
driver pool. The cu126 runtime must pass its ready-marker/live-CUDA gate with
CUDA `compat` paths removed as documented in `scripts/cluster/README.md`.
DeepGEMM's JIT is compiled separately with pinned `nvcc 12.9.86`; do not add
that compiler prefix to `LD_LIBRARY_PATH` or replace the portable cu126 serving
runtime with its libraries. SGLang TVM-FFI compilation uses the same prefix as
`CUDA_HOME` and its `targets/x86_64-linux/lib` only through compile-time
`LIBRARY_PATH`. The job precompiles and persists the TP=8 IPC, communicator,
and BF16 custom-all-reduce modules before server startup. Never add the 12.9
prefix to runtime `LD_LIBRARY_PATH` merely to make `-lcudart` link.

The submitter uses the verified job-compatible image
`cr.ai.cloud.ru/2754eb6e-ae19-4123-87ce-06ec3cc96500/job-latentdiffusion:flash-clear`,
region SR008, one 8-GPU worker, `type="binary"`, detached mode, Internet access,
large shared memory, and `processes_per_worker=1`. Omitting that last field was
experimentally shown to launch eight MPI ranks. The canonical entrypoint also
exits any unexpected nonzero rank and restores devices 0–7 for the sole TP=8
server. EvolutionLoop itself remains one coordinator.

Old team notebooks used `queue_name="diff"` and
`priority_class="high"`. They are not defaulted because that policy was for an
older allocation; pass them explicitly only after confirming SR008 accepts it.

## Status, logs, and monitor

```python
import client_lib

JOB = "lm-mpi-job-..."
print(client_lib.get_job_status(JOB, region="SR008"))
client_lib.logs(JOB, region="SR008", tail=200, verbose=False)

# Destructive; only when cancellation is intended:
# client_lib.kill(JOB, region="SR008")
```

The durable monitor is preferred to manual log polling:

```bash
bash "$MLS_ENV" "$MLS_PY" -m scripts.cluster.monitor \
  --job-name "$JOB" --run-dir /absolute/nfs/path/to/run
```

It writes scheduler progress and a terminal `completion_event.json`; live
generation/token progress is emitted as `EVOLUTIONLOOP_EVENT` JSON in job logs.
Typical states are Pending, Inqueue, Starting, Running, Completed, Failed,
Cancelled/Deleted. Save `submission.json`; its job name is the stable handle.

After completion, use `python -m scripts.cluster.transfer` as documented in
`scripts/cluster/README.md` to copy the regional run into workspace NFS.

Official references:

- <https://cloud.ru/docs/aicloud/mlspace/concepts/client-lib__job>
- <https://cloud.ru/docs/aicloud/mlspace/concepts/client-lib__common-config>
