from dataclasses import dataclass
import pandas as pd
from .data import ordered

@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def leave_last_out(ratings, min_interactions=5):
    data = ordered(ratings)
    groups = data.groupby('user_id', sort=False)
    eligible = groups.user_id.transform('size') >= min_interactions
    reverse = groups.cumcount(ascending=False)
    return Split(data[~(eligible & (reverse < 2))].copy(),
                 data[eligible & (reverse == 1)].copy(), data[eligible & (reverse == 0)].copy())
