# Bundled Kaggle wheels

These pure-Python wheels let the Kaggle runner install the two packages that
are absent from the standard Kaggle image without contacting PyPI or replacing
Kaggle's CUDA-enabled PyTorch stack.

- `pyts==0.13.0`
- `tsai==0.4.1`

Regenerate them from PyPI with:

```powershell
python -m pip download --only-binary=:all: --no-deps `
  --dest vendor/wheels pyts==0.13.0 tsai==0.4.1
```

The packages remain subject to their respective upstream licenses, which are
included in the wheel metadata.
