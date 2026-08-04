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

The persistent serving environment is `sglang-0.5.16-cu126`. This is
intentional: SR008 allocations have exposed both R560/CUDA-12.6 and R580
drivers. SGLang's PyPI metadata defaults to CUDA 13, which fails on the R560
nodes. `bootstrap_deepseek_env.sh` follows SGLang 0.5.16's upstream CUDA-12
Docker recipe (cu126 PyTorch, CUDA-12 FlashInfer/CUTLASS dependencies, Hopper
kernels; release 0.4.5's actually published SM90 wheel is cu129), verifies that
CUDA initializes, and only then publishes the runtime ready marker. Never reuse
the incompatible legacy `sglang-0.5.16` directory.

DeepGEMM is different from the serving runtime: it JIT-compiles Hopper cubins
when SGLang encounters a new kernel shape. The base image's `nvcc 12.6.85`
cannot compile the 128-bit shared-memory operand used by DeepSeek-V4's MHC
prenorm kernel. `bootstrap_cuda_toolchain.sh` therefore creates a separate
NVIDIA conda prefix, `cuda-nvcc-12.9.86`, and validates both an SM90a cubin and
the real `tf32_hc_prenorm_gemm` path before model loading. Only the compiler
path is exported through `DG_JIT_NVCC_COMPILER`; its libraries are never added
to `LD_LIBRARY_PATH`, so PyTorch remains on the portable cu126 runtime. The
persistent `SGLANG_DG_CACHE_DIR` reuses compiled kernels across job retries.

The immutable compiler prefix also pins NVIDIA `libcurand-dev 10.3.10.19`.
FlashInfer compiles its sampling/renormalization kernels during CUDA-graph
capture and requires `curand.h`; a compiler-only `cuda-nvcc` prefix reaches
model load but fails before the HTTP endpoint becomes ready. The bootstrap now
compiles a real SM90a `curand_kernel.h` cubin before publishing its ready
marker. cuRAND is compile-time-only here and the prefix still never enters
runtime `LD_LIBRARY_PATH`. DeepGEMM's persistent cubin cache remains keyed only
by NVCC version and SM architecture, so adding the headers does not invalidate
already compiled kernels.

SGLang's TP=8 NVLink custom-all-reduce path also JIT-compiles three TVM-FFI
modules (`cuda_ipc`, `communicator`, and BF16 world-size-8 custom all-reduce).
The same isolated 12.9 prefix is exposed as `CUDA_HOME`; its
`targets/x86_64-linux/lib` directory is added only to compile-time
`LIBRARY_PATH`, because TVM-FFI links `-lcudart`. It is deliberately absent
from runtime `LD_LIBRARY_PATH`. Before the eight server ranks start,
`smoke_sglang_jit_toolchain.py` compiles all three modules once and validates
their content-addressed artifacts in the persistent `TVM_FFI_CACHE_DIR`. This
avoids both the CUDA-12.6 template error and an eight-rank first-start JIT race.

Before any GPU Python process starts, `cuda_driver_env.sh` removes CUDA
forward-compatibility directories from `LD_LIBRARY_PATH`. ML Space mounts the
node's real driver under its native paths; preferring an older image-bundled
`compat/libcuda.so` causes CUDA error 803 on newer-driver allocations. Other
CUDA, NCCL, HPC-X, and scheduler-mounted NVIDIA paths are preserved.

Because the SGLang wheel itself retains its PyPI-default
`Requires-Dist: cuda-python>=13`, a plain `pip check` reports exactly that one
metadata mismatch against the intentionally installed `cuda-python 12.x`.
This is expected for the audited CUDA-12 dependency view; installing CUDA 13
to silence it breaks portability to R560 nodes. Runtime acceptance is the
version/native-import/live-CUDA gate plus the persisted package freeze.

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

The complete run remains durable under regional NFS. The production job writes
`result_summary.json` and emits it in the terminal `run_completed` event, so
generation/inflight integrity, survivor scores, organism outcomes, and token
totals can be audited from scheduler logs even though the Jupyter and SR008 NFS
namespaces differ. Completion is rejected unless the requested generation is
reached, both inflight transactions are absent, and real token telemetry parses
without errors.

Do not use the repository's legacy `scripts.cluster.transfer` command on the
current control plane: Cloud.ru has disabled the `client_lib.copy_from_nfs` and
`copy_to_nfs` functions retained by client_lib 0.6.3, and its legacy logs route
returns 404. For a full workspace-NFS export, configure the supported
`cloudru-ml-cli` (`mls` >= 0.7.1) with user credentials and create an NFS→NFS
transfer rule, for example:

```bash
mls transfer create \
  --name evolutionloop-export-<run-id> \
  --connector-id <sr008-nfs-connector> --connector-type nfs \
  --dst-connector-id <workspace-nfs-connector> --dst-connector-type nfs \
  --source evolutionloop-deepseek-v4/runs/<run-id> \
  --destination <workspace-path>/cluster-results/<run-id> \
  --strategy write_all
```

Never synthesize that profile from scheduler gateway variables or pass storage
credentials into a job environment. Official Cloud.ru documentation lists the
old client-lib copy functions as disabled and the `mls transfer` command as the
supported interface.

Each model call is logged to `population/llm_usage.jsonl`. A completed job also
writes `token_usage_summary.json`, grouped by route, stage, and generation.
Startup acceptance artifacts include `deepgemm_toolchain_smoke.json` and
`sglang_jit_toolchain_smoke.json`; the latter records the exact compiler,
TVM-FFI version, GPU/compute capability, cache namespace, and compiled modules.
`result_summary.json` is the compact, log-readable terminal results capsule.
