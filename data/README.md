# Data directory

This public release intentionally does not include large raw benchmark data,
OpenML caches, or generated data archives.

Large data files and cached benchmark artifacts will be hosted separately:

[Google Drive data and cached results](https://drive.google.com/drive/folders/1degj8NKU1FJTib9hocOrUMixivfM_5vt?usp=sharing)

Place downloaded files under `data/raw/` or pass a dataset directory to the
relevant experiment script, for example:

```bash
python experiments/exp8_real_data.py --data-dir data/raw
```
