import numpy as np
import numpy.linalg


def cm_to_weight(cm, regime_freq=None):
    """Calibrate the simulation ``weight`` parameter from an empirical confusion matrix.

    The simulation's misclassification model sets::

        P(correct | true regime r) = 1 - weight * (1 - 1/R)

    This function inverts that relationship: given a confusion matrix it
    returns the ``weight`` value that reproduces the observed accuracy.

    Parameters
    ----------
    cm : array-like of shape (R, R)
        Confusion matrix in **project convention**: ``cm[i, j]`` =
        P(predicted=i | true=j), i.e. columns sum to 1 (or to raw counts per
        true class).  This matches ``cm_4``/``cm_10`` from ``ml_switching_reg.cm``
        and the column-normalised matrices used throughout the codebase.
        (If you have a scikit-learn confusion matrix where rows = true class,
        pass ``cm.T``.)
    regime_freq : array-like of shape (R,), optional
        Weights for averaging per-regime accuracy.  Defaults to equal weights.
        Pass the empirical regime frequencies (e.g. class sizes) for a
        frequency-weighted average.

    Returns
    -------
    weight : float in [0, 1]
        Calibrated misclassification weight.
    info : dict
        ``per_regime_accuracy`` — P(correct | true regime r) for each r.
        ``mean_accuracy``       — (weighted) average accuracy.
        ``R``                   — number of regimes.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_switching_reg.cm import cm_4
    >>> weight, info = cm_to_weight(cm_4)   # no .T needed
    """
    cm = np.asarray(cm, dtype=float)
    R = cm.shape[0]
    if cm.shape != (R, R):
        raise ValueError(f"cm must be square; got shape {cm.shape}")

    # Column-normalise: cm[:, j] = P(predicted | true=j), diagonal = P(correct | true=r)
    col_sums = cm.sum(axis=0, keepdims=True)
    if np.any(col_sums == 0):
        raise ValueError("One or more true-class columns sum to zero.")
    cm_norm = cm / col_sums

    per_regime_acc = np.diag(cm_norm)  # P(correct | true=r) for each r

    if regime_freq is None:
        regime_freq = np.ones(R) / R
    else:
        regime_freq = np.asarray(regime_freq, dtype=float)
        regime_freq = regime_freq / regime_freq.sum()

    mean_acc = float(per_regime_acc @ regime_freq)

    # Invert: mean_acc = 1 - weight * (1 - 1/R)
    weight = (1.0 - mean_acc) / (1.0 - 1.0 / R)
    weight = float(np.clip(weight, 0.0, 1.0))

    info = {
        "per_regime_accuracy": per_regime_acc,
        "mean_accuracy": mean_acc,
        "R": R,
    }
    return weight, info

def lagged_drought_df(data=None, drought_cols=None, shift=None, dropna=False, groupby_index=['hashed_driver_uuid'], date_col= None, assign_only=False):
    
    if shift is None:
        return data
    
    def lag_drought_col(d, shift):

        return lambda df: df.groupby(groupby_index)[d].shift(shift)        
    
    assign_dict = {
        f"lagged_{s_name}_{d}": lag_drought_col(d, s)
        for d in drought_cols
        for s, s_name in zip(shift, [f"neg_{str(i)[1:]}" if i<0 else i for i in shift])
    }
    
    if assign_only:
        return list(assign_dict.keys())
    
    df = (
        data.set_index([groupby_index, date_col])
        .sort_index()
        .assign(**assign_dict)
        .reset_index()
    )
    
    if dropna:
        df = df.dropna(subset=assign_dict.keys())

    return df

def set_covariance(x, diag=1, size=2):
    """Given a sizexsize matrix with diagonal `diag`
    increase covariance
    """
    
    if isinstance(x, (int, float)):
        x = [x]
    
    X = np.diag([diag]*size)
    
    # fill off-diagonal
    
    return np.where(X==0, x, X)

def create_list_covariance_matrices(num = 21, size = 2, **kwargs):
    """Creates a list of covariance matrices

    Args:
        r (iterable, optional)
    """
        
    # r = np.repeat(np.linspace(0,1,num),size).reshape(num,size)
    r = np.linspace(0,1,num=num)
    
    mat_list = []
    
    for i in r:
        
        mat = set_covariance([i]*size, size=size,**kwargs)
        # Check if invertible
        try:
            numpy.linalg.inv(mat)
        except numpy.linalg.LinAlgError:
            print(f"setting covariance at {i} led to singular matrix, skipping...")
            continue
        
        mat_list.append(mat)
    
    return mat_list