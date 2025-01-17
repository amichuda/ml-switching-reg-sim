import numpy as np
import numpy.linalg

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