import numpy as np
import numpy.linalg

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