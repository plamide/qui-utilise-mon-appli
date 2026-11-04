import pandas as pd

def read_ds(ds_name: str):
        path = ds_name if ds_name.endswith('.csv') else f'data/{ds_name}.csv'
        with open(path) as f:
            lines = f.readlines()
            max_actions = max(len(str(c).split(",")) for c in lines)
            _names = ["util", "navigateur"] if "train" in path else ["navigateur"]
            _names.extend(range(max_actions - len(_names)))
            from io import StringIO
            return pd.read_csv(StringIO("".join(lines)), names=_names, dtype=str)
