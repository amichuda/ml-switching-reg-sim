
import pandas as pd
import numpy as np
from functools import reduce

class UberDatasetCreator:
    
    def __init__(self,
                 drivers = 275, 
                 regimes = 4,
                 time_periods = 10,
                 seed = None):
        
        self.regimes = regimes
        self.drivers = drivers
        self.time_periods = time_periods
        self.N = drivers*regimes*time_periods
        self.seed = seed
        
        # Create new random generator instance
        self.random = np.random.default_rng(seed=seed)
    
    def _create_drought_index(self,
                            mean: list = None,
                            cov =  None, 
                            random_walk=False):
        """
        Creates the drought index, optionally with correlation
        (and maybe seasonality)
        """
        
        if mean is None:
            mean = [-1]*self.regimes
            
        if cov is None:
            cov = [1]*self.regimes
        
        if isinstance(cov[0], (list, np.ndarray)):

            df = pd.DataFrame(
                self.random.multivariate_normal(mean, cov, size=self.time_periods, 
                                                check_valid = 'raise', 
                                                method = 'eigh'), 
                    columns = [f"drought_{i}" for i in range(self.regimes)]
                    )
        elif isinstance(cov[0], (int, float)):
            df = pd.DataFrame(
                {f'drought_{i}' : self.random.normal(m, sd, size= self.time_periods)\
                    for i, (m, sd) in enumerate(zip(mean, cov))}
            )
        
        df.index.names = ['time']
        
        return df
    
    def _create_drought_index_with_driver(self, 
                                          mean = None, 
                                          cov = None):
        
        df_list = []
        
        # Create drought data
        drought_df = self._create_drought_index(mean = mean, cov=cov)
        
        # Create list of the same data but with driver ids
        for i in range(self.drivers):
            df_list.append(drought_df.assign(driver = i))

        driver_time_df = pd.concat(df_list).set_index('driver', append=True)
        
        driver_time_df.index.names = ['time', 'driver']
        
        return driver_time_df.reorder_levels(['driver', 'time'])

    
    def _create_driver_index(self):
        
        return pd.DataFrame({'driver' : list(range(self.drivers))})
    
    def _create_regime_index(self, p = None):
        
        return pd.DataFrame({'regime': self.random.choice(list(range(self.regimes)),
                                   size= self.drivers, p=p)})
        
    def misclassification_weight(self, regimes, high, low=0.):
        """
        Starts with an identity matrix and then adds a 
        random number with range from `low` to `high`.
        `high` is the weight. So if weight is high, then
        chances are, you will have more misclassification 
        """
        
        result = np.identity(regimes) + self.random.uniform(low=low, high=high, size=(regimes, regimes))
        result /= result.sum(axis=1, keepdims=1)

        return result
    
    def _misclassify_regime(self, x, mw):
        
        for i, p_vec in enumerate(mw):
            
            if x == i:
                                
                return self.random.choice(range(self.regimes), 
                                            p=p_vec)        
                
        
    def _create_y(self, data, beta1, beta0 = None, name = 'y', sd= None):
        
        if sd is None:
            sd = [1]*self.regimes
        if isinstance(beta0, int):
            beta0 = [beta0]*self.regimes
        if isinstance(beta1, int):
            beta1 = [beta1]*self.regimes
            
        def y_lambda(i):
            return lambda df: beta0[i] + beta1[i]*df[f'drought_{i}'] + self.random.normal(0, 
                                                                                          sd[i], 
                                                                                          self.time_periods*self.drivers)
            
        df = (
            data
            .assign(**{name + f'_{i}' : y_lambda(i) for i in range(self.regimes)})
        )
        
        return df
        
    def construct(self, 
                  seed=None,
                  y_sd = None,
                  drought_mean = None,
                  drought_cov = None,
                  beta0 = 12,
                  beta1 = 2,
                  y_name = 'y',
                  weight = 0.9,
                  reg_ready = False,
                  output_true_beta = False,
                  output_sigma = False,
                  jittered = True,
                  ):
        
        if seed is not None:
            self.random = np.random.default_rng(seed=seed)
        
        # Create drought index
        drought = self._create_drought_index_with_driver(mean = drought_mean,
                                                         cov = drought_cov)
        
        # Create randomly assigned regime membership
        regime = pd.concat([self._create_driver_index(), 
                             self._create_regime_index()], 
                           axis=1)
        
        # Get weight matrix for misclassification
        mw = self.misclassification_weight(self.regimes, weight)

        if jittered:
            
            m = MisclassificationCreator(self.regimes)  
                      
            # Create wrong regime variable
            regime_with_misclassified = (
                regime
                .assign(misclass_regime = lambda df: df['regime']\
                    .apply(lambda x: m.noisify_matrix(extent=weight, index=x))
                    )
            )
            
            regime_dummies = (
                pd.concat([pd.get_dummies(regime_with_misclassified, columns=['regime']), 
                           regime_with_misclassified
                           .apply(lambda x: x['misclass_regime'], 
                                  result_type='expand', 
                                  axis=1)
                           ], 
                          axis=1)
                .rename({old: f"misclass_regime_{old}" for old in range(self.regimes)}, axis=1)
                .assign(max_misclass_regime = lambda df: df['misclass_regime'].apply(lambda x: x.argmax()))
                .drop(['misclass_regime'], axis=1)
            )
        else:
            mw = self.misclassification_weight(self.regimes, weight)
            
            # Create wrong regime variable
            regime_with_misclassified = (
                regime
                .assign(misclass_regime = lambda df: df['regime']\
                    .apply(self._misclassify_regime, mw = mw)
                    )
            )
        
            # Now create dummies for regime and wrong regime
            regime_dummies = pd.get_dummies(regime_with_misclassified,
                                            columns = ['regime', 'misclass_regime'])
        
        # Create y-variable and merge in regime
        df = (
            drought
            .join(regime_dummies.set_index('driver'))
            .join(regime_with_misclassified.set_index('driver'))
            .pipe(self._create_y, 
                  beta0 = beta0, 
                  beta1=beta1, 
                  name = y_name, 
                  sd= y_sd)
            .pipe(pd.get_dummies, columns=['max_misclass_regime'])
        )
        
        for r in range(self.regimes):
            
            df.loc[lambda df: df['regime'] == r, y_name] = df[f'{y_name}_{r}']
            
        if reg_ready:
            
            # Get regime columns
            regime_cols = df.columns[df.columns.str.contains("^regime")].tolist()
            
            # Get y columns
            y_cols = df.columns[df.columns.str.contains(f"{y_name}_")].tolist()
            
            df = df.drop(regime_cols + y_cols + ['regime', 'misclass_regime'],axis=1)
        
        if output_true_beta:
            if output_sigma:
                return df, mw, [beta0, beta1], y_sd
            else:
                return df, mw, [beta0, beta1]
                    
        return df, mw
    
class MisclassificationCreator:
    
    def __init__(self, regimes, seed=None):
        """Creates a matrix of misclassification, ranging from:
        - 0, no misclassification
        - 1, completely uninformative, more or less equal to 1/regimes (with a small jitter so the max function can work)

        """
        
        if seed is None:
            seed=1234
        
        self.regimes = regimes
        self.random = np.random.default_rng(seed=seed)
        
    def _no_misclass_matrix(self):
        
        return np.identity(self.regimes)
    
    def _index_to_misclassification(self, extent):
        """Puts misclassification domain (0,1) to domain of misclassification function

        Args:
            extent (float): The extent of misclassification

        Returns:
            float: The amount of misclassification in terms of the regimes
        """
        
        return extent* (1-(1./self.regimes))
    
    def noisify_matrix(self, extent, index):
        
        extent = self._index_to_misclassification(extent)
        
        if self.regimes == 2:
            new_array = np.array([1-extent])
            
            return np.insert(new_array, index, extent)
        
        def _recursive_fill_in(extent):
            
            
            if extent.shape == ():
                extent_diff = extent.sum()
            else:
                # Get what the next index will receive
                extent_diff = reduce(lambda x,y: x-y, extent)
                
            if extent.shape != () and extent.shape[0] == self.regimes-1:
                
                return np.append(extent, extent_diff)
                
            new_extent = np.random.uniform(0, extent_diff)
            
            return _recursive_fill_in(np.append(extent, new_extent))
        
        new_vec = np.delete(_recursive_fill_in(np.array(extent)), 0)
        
        self.random.shuffle(new_vec)
        
        return np.insert(new_vec, index, 1-extent)
        
        
    
    
        
    
        
        
        