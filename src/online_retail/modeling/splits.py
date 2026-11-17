"""Date-grouped expanding folds with complete training outcomes."""

import pandas as pd

FOLD_WINDOWS = [
    ("2010-07-05", "2010-07-26"),
    ("2010-08-02", "2010-08-30"),
    ("2010-09-06", "2010-10-11"),
]


def temporal_folds(data: pd.DataFrame):
    """Yield positional indices; outcome ends cannot cross validation start."""
    for start, end in FOLD_WINDOWS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        fit = (data.cutoff + pd.Timedelta(days=50) <= start).to_numpy().nonzero()[0]
        valid = data.cutoff.between(start, end).to_numpy().nonzero()[0]
        if not len(fit) or not len(valid):
            raise ValueError("Insufficient data for configured chronological folds")
        for rows in (fit, valid):
            if set(data.iloc[rows].RepurchaseWithin50Days) != {0, 1}:
                raise ValueError("Each fold must contain both binary classes")
        yield fit, valid
