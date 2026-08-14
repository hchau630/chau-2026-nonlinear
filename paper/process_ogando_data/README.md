1. Run
```
python process_ogando_data/compress_data.py ogando_data -o
```
2. Run
```
python process_ogando_data/normalize_data.py ogando_data -o
```
3. (optional) Run
```
python process_ogando_data/validate_normalized.py ogando_data
```
4. Run
```
python process_ogando_data/preprocess_data.py ogando_data --2d -o
```