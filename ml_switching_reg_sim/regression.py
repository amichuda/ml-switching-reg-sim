import statsmodels.formula.api as smf
import pandas as pd
import numpy as np


def get_reg_effect(res):

    beta0_reg = res.params.loc[lambda df: ~df.index.str.contains("drought_")].values
    # beta1_main_reg = res.params.loc[lambda df: df.index.str.contains("^drought_")].values 

    beta1_reg = res.params.loc[lambda df: df.index.str.contains("misclass_regime_[0-9]:drought_")].values 

    # beta1_reg = beta1_main_reg + beta1_int_reg
        
    return beta0_reg, beta1_reg, np.append(beta0_reg, beta1_reg)


def uber_regression(data: pd.DataFrame, 
                    endog_col: str,
                    classifier_cols: list, 
                    drought_cols: list,
                    show_formula = False):
        
    formula_str = f'{endog_col} ~ -1 +'
    
    formulas = []
        
    for c in classifier_cols:
        
        formulas.append(f"{c}")
    
        
    for c, d in zip(classifier_cols, drought_cols):
        
        if isinstance(d, list):
            for i in d:
                formulas.append(f"{c}:{i}")
        else:
            formulas.append(f"{c}:{d}")
        
    formula_str = formula_str + '+ '.join(formulas)
    
    mod = smf.ols(formula_str, 
                  data = data
                  )
    
    if show_formula:
        return mod, formula_str
    
    return mod
    
    