# ml_engine/datasets/utils.py
import pandas as pd
import numpy as np
from typing import Dict, Any

def parse_csv(file_path: str) -> Dict[str, Any]:
    df = pd.read_csv(file_path)
    columns = list(df.columns)
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    common_targets = ['target', 'label', 'y', 'class', 'outcome']
    target_candidates = [col for col in columns if col.lower() in common_targets]
    if not target_candidates and len(columns) > 0:
        target_candidates = [columns[-1]]
    preview_df = df.head(5).replace({np.nan: None})
    preview = preview_df.to_dict(orient="records")
    return {
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "target_candidates": target_candidates,
        "preview": preview,
    }