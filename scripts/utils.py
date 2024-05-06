import pandas as pd
import os 
from typing import Union

def load_dataset(x_path: Union[str, os.PathLike], y_path: Union[str, os.PathLike]) -> pd.DataFrame:
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    return X, y.iloc[:, 0]
    
    