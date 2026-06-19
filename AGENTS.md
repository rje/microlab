# Microlab Agent Instructions

## Python Environment

This project uses a dedicated Anaconda environment named `microlab`.

Do not install packages into `base`.
Do not run project Python commands with the system Python or base Python.

For interactive shell work:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate microlab
```

For non-interactive commands, prefer the absolute conda executable:

```bash
/home/rje/anaconda3/bin/conda run -n microlab python ...
/home/rje/anaconda3/bin/conda run -n microlab python -m pip ...
/home/rje/anaconda3/bin/conda run -n microlab pytest
/home/rje/anaconda3/bin/conda run -n microlab ruff check .
```

When adding or changing dependencies:

1. Update `environment.yml`.
2. Apply changes with:

   ```bash
   /home/rje/anaconda3/bin/conda env update -n microlab -f environment.yml --prune
   ```

3. Verify imports from the `microlab` environment.

## Project Layout

- `plans/` contains project plans and setup notes.
- `papers/` contains the organized paper library and manifest.
- `scripts/` contains utility scripts.
- `site/` contains the Microlab Console React frontend.
- `site/content/` contains structured dashboard data.
- `ops/systemd/` contains user-service templates.
- `environment.yml` is the source of truth for the Python environment.

## Microlab Console

The local website is served by the Python script in the `microlab` conda environment:

```bash
/home/rje/anaconda3/bin/conda run -n microlab python scripts/serve_site.py --host 127.0.0.1 --port 8765
```

Node commands are only run inside `site/`:

```bash
cd site
npm install
npm test
npm run build
```

Do not install frontend dependencies at the project root.

## Jupyter

Use the registered Jupyter kernel named `Python (microlab)`.

If the kernel is missing, recreate it with:

```bash
/home/rje/anaconda3/bin/conda run -n microlab python -m ipykernel install --user --name microlab --display-name "Python (microlab)"
```

## GPU Checks

Before training runs, verify CUDA from inside the environment:

```bash
/home/rje/anaconda3/bin/conda run -n microlab python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`nvidia-smi` may require a server reboot after NVIDIA driver updates if the loaded kernel module and user-space NVML library versions differ.
