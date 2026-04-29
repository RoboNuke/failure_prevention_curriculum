# Isaac Lab on HPC (Apptainer/Singularity)

Setup for running Isaac Lab on a Rocky Linux HPC cluster with Apptainer 1.4.5+, no Docker, no admin rights.

## Environment

- **Cluster OS**: Rocky Linux 8/9, Slurm
- **Container runtime**: Apptainer 1.4.5 (Singularity-compatible)
- **GPU**: NVIDIA L40S (sm_89), driver 590.48.01, CUDA 13.1
- **Storage**: home dir on small NFS quota, `~/hpc-share` on Lustre (1.5T)

## 1. Configure build environment

The default temp dir is too small and Lustre causes squashfs xattr issues during build. Use local node `/tmp` for build scratch.

```bash
export APPTAINER_TMPDIR=/tmp/apptainer_$USER
export APPTAINER_CACHEDIR=/tmp/apptainer_$USER/cache
mkdir -p $APPTAINER_TMPDIR
```

## 2. NGC authentication

Isaac Lab images on `nvcr.io` require an NGC account.

1. Register at https://ngc.nvidia.com (free)
2. Profile → Setup → Generate API Key
3. Login on the HPC:

```bash
apptainer remote login --username '$oauthtoken' docker://nvcr.io
# Paste API key as password
```

## 3. Definition file

Save as `~/hpc-share/isaaclab/isaaclab.def`:

```
Bootstrap: docker
From: nvcr.io/nvidia/isaac-lab:2.1.0

%environment
    export ISAACSIM_PATH=/isaac-sim
    export ISAACSIM_PYTHON_EXE=/isaac-sim/python.sh
    export ISAACLAB_PATH=/workspace/isaaclab
    export OMNI_KIT_ALLOW_ROOT=1
    export ACCEPT_EULA=Y
    export PRIVACY_CONSENT=Y
    export PYTHONNOUSERSITE=1

%post
    ls -la /isaac-sim || (echo "Isaac Sim not found!" && exit 1)
    ls -la /workspace/isaaclab || (echo "Isaac Lab not found!" && exit 1)

%runscript
    exec /workspace/isaaclab/isaaclab.sh "$@"
```

## 4. Build the image

Build on a compute node (login nodes typically have RAM/process limits that crash mksquashfs):

```bash
srun --partition=<cpu-partition> --time=2:00:00 --mem=32G --pty bash

export APPTAINER_TMPDIR=/tmp/apptainer_$USER
export APPTAINER_CACHEDIR=/tmp/apptainer_$USER/cache
mkdir -p $APPTAINER_TMPDIR

cd ~/hpc-share/isaaclab
apptainer build --fakeroot \
    --mksquashfs-args "-mem 8G -processors 4" \
    isaaclab.sif isaaclab.def 2>&1 | tee build.log
```

Notes:
- `--fakeroot` needed since you don't have root on the cluster
- `--mksquashfs-args "-mem 8G -processors 4"` prevents OOM during squashfs creation
- Build takes 30–60 min, final image ~20 GB

## 5. Create writable cache directories

The `.sif` is read-only, but Isaac Sim writes runtime caches at several paths. Bind-mount writable dirs from the host:

```bash
mkdir -p ~/hpc-share/isaac_cache/{kit_data,kit_cache,ov_data,ov_cache,nvidia_omniverse}
```

## 6. Run Isaac Lab

Allocate a GPU node and run with `--containall` (prevents host Python contamination) plus the cache binds:

```bash
srun --partition=<gpu-partition> --gres=gpu:1 --time=1:00:00 --pty bash

apptainer run --nv --containall \
    --bind ~/hpc-share:/work \
    --bind /tmp:/tmp \
    --bind ~/hpc-share/isaac_cache/kit_data:/isaac-sim/kit/data \
    --bind ~/hpc-share/isaac_cache/kit_cache:/isaac-sim/kit/cache \
    --bind ~/hpc-share/isaac_cache/ov_data:/root/.local/share/ov \
    --bind ~/hpc-share/isaac_cache/ov_cache:/root/.cache/ov \
    --bind ~/hpc-share/isaac_cache/nvidia_omniverse:/root/.nvidia-omniverse \
    ~/hpc-share/isaaclab/isaaclab.sif \
    -p /workspace/isaaclab/scripts/environments/random_agent.py \
    --task Isaac-Stack-Cube-Franka-v0 \
    --num_envs 16 \
    --headless
```

### Why each flag matters

| Flag | Purpose |
|------|---------|
| `--nv` | Mounts host NVIDIA driver libraries into container |
| `--containall` | Isolates host filesystem; prevents host `~/.local` Python packages from leaking in |
| `--bind /tmp:/tmp` | Gives Isaac Sim a writable scratch space |
| `--bind .../kit_data` and `kit_cache` | Isaac Sim writes config + shader caches here |
| `--bind .../ov_data`, `ov_cache`, `nvidia_omniverse` | Omniverse runtime data dirs (under `/root/...` since `--containall` makes `$HOME=/root`) |
| `--headless` | No display required |

Bind destinations must already exist inside the container. `/root/.nvidia-omniverse/logs` does not exist by default, so bind the parent `nvidia_omniverse` dir instead — logs are created inside it at runtime.

## 7. Convenience wrapper (optional)

```bash
cat > ~/hpc-share/isaaclab/run.sh << 'EOF'
#!/bin/bash
CACHE=~/hpc-share/isaac_cache
mkdir -p $CACHE/{kit_data,kit_cache,ov_data,ov_cache,nvidia_omniverse}
apptainer run --nv --containall \
    --bind ~/hpc-share:/work \
    --bind /tmp:/tmp \
    --bind $CACHE/kit_data:/isaac-sim/kit/data \
    --bind $CACHE/kit_cache:/isaac-sim/kit/cache \
    --bind $CACHE/ov_data:/root/.local/share/ov \
    --bind $CACHE/ov_cache:/root/.cache/ov \
    --bind $CACHE/nvidia_omniverse:/root/.nvidia-omniverse \
    ~/hpc-share/isaaclab/isaaclab.sif "$@"
EOF
chmod +x ~/hpc-share/isaaclab/run.sh
```

Usage:
```bash
~/hpc-share/isaaclab/run.sh -p /workspace/isaaclab/scripts/environments/random_agent.py --task Isaac-Stack-Cube-Franka-v0 --num_envs 16 --headless
```

## 8. Slurm batch template

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=isaaclab
#SBATCH --output=logs/%j.out

~/hpc-share/isaaclab/run.sh -p /work/your_training_script.py
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Out of memory (cache_alloc)` during build | mksquashfs unbounded RAM | Add `--mksquashfs-args "-mem 8G -processors 4"` |
| `Unrecognised xattr prefix lustre.lov` | `APPTAINER_TMPDIR` on Lustre | Set to `/tmp/apptainer_$USER` (local disk) |
| `Read-only file system: /isaac-sim/kit/data` | Container fs is read-only | Bind-mount writable host dirs (Step 5) |
| `Using python from: /nfs/.../miniforge/...` | Host Python leaking in | Use `--containall` |
| `destination ... doesn't exist in container` | Bind target path missing in image | Bind a parent dir that does exist |
| `ModuleNotFoundError: No module named 'isaacsim.core'` | Trying to import before `AppLauncher` runs | Use `AppLauncher(headless=True)` first, or run scripts that handle this |