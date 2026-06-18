# Microlab Python Environment

Microlab uses a dedicated Anaconda environment named `microlab`. Do not install project packages into `base`.

## Create or Update

From the project root:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
```

If the environment already exists:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda env update -n microlab -f environment.yml --prune
```

## Activate

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate microlab
```

If your shell already initializes conda, the `source` line is not necessary.

## Verify

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The Python executable should live under an Anaconda env path such as:

```text
/home/rje/anaconda3/envs/microlab/bin/python
```

## Note

If `torch.cuda.is_available()` is false, first check `nvidia-smi`. CUDA training requires the NVIDIA driver and runtime libraries to agree.
