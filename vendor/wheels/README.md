# Bundled Kaggle wheels

These wheels let the Kaggle runner install packages that are absent or too old
in the standard Kaggle image without contacting PyPI or replacing Kaggle's
CUDA-enabled PyTorch stack.

- `pyts==0.13.0`
- `psutil==6.1.1` (Linux `abi3` wheel for Kaggle)
- `tsai==0.4.1`

Regenerate them from PyPI with:

```powershell
python -m pip download --only-binary=:all: --no-deps `
  --dest vendor/wheels pyts==0.13.0 tsai==0.4.1

python -m pip download --only-binary=:all: --no-deps `
  --platform manylinux2014_x86_64 --python-version 312 `
  --implementation cp --abi cp312 --dest vendor/wheels psutil==6.1.1
```

The packages remain subject to their respective upstream licenses, which are
included in the wheel metadata.
