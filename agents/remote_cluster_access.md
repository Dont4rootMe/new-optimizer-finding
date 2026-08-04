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

Synchronize this finalization branch into a separate NFS clone. Do not switch
or edit the author's existing checkout. SR008 jobs have verified visibility
under `/home/jovyan/echimbulatov/...`; deeper
`fork_afedorov/constant_repos/...` paths are visible from Jupyter but were not
mounted into the H100 job container. From the job-visible clone:

```bash
bash "$MLS_ENV" "$MLS_PY" -m scripts.cluster.submit \
  --project-root /home/jovyan/echimbulatov/new-optimizer-finding-finalize \
  --run-id deepseek-v4-flash-0731-circle-300
```

The submitter uses the verified job-compatible image
`cr.ai.cloud.ru/2754eb6e-ae19-4123-87ce-06ec3cc96500/job-latentdiffusion:flash-clear`,
region SR008, one 8-GPU worker, `type="binary"`, detached mode, Internet access,
and large shared memory. On the observed allocation even `binary` invokes the
shell once per GPU; the canonical entrypoint exits every nonzero MPI rank and
restores all eight visible devices on rank 0. This guard is mandatory because
EvolutionLoop itself is one coordinator.

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

It writes scheduler/run progress and a terminal `completion_event.json`.
Typical states are Pending, Inqueue, Starting, Running, Completed, Failed,
Cancelled/Deleted. Save `submission.json`; its job name is the stable handle.

Official references:

- <https://cloud.ru/docs/aicloud/mlspace/concepts/client-lib__job>
- <https://cloud.ru/docs/aicloud/mlspace/concepts/client-lib__common-config>
