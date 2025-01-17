
import pandas as pd
import numpy as np
from functools import reduce
from .utils import lagged_drought_df
from mockseries.trend import LinearTrend, Switch
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.noise import RedNoise
from mockseries.transition import LinearTransition
from datetime import timedelta


class UberDatasetCreator:
    
    def __init__(self,
                 drivers = 275, 
                 regimes = 4,
                 time_periods = 10,
                 lags = None,
                 seed = None):
        
        self.regimes = regimes
        self.drivers = drivers
        self.time_periods = time_periods
        self.N = drivers*regimes*time_periods
        self.seed = seed
        self.lags = lags
        
        # Create new random generator instance
        self.random = np.random.default_rng(seed=seed)
        
        self.dates = pd.date_range(start="2016-01-01",
              periods=self.time_periods,
              freq='M')
    
    def _create_drought_index(self,
                            mean: list = None,
                            cov =  None, 
                            mock=False,
                            random_walk=False,
                            mock_dict=None):
        """
        Creates the drought index, optionally with correlation
        (and maybe seasonality)
        """
        
        regime_ids = range(self.regimes)

        alpha_r = {k:v for k,v in zip(regime_ids, self.random.normal(0,1, size=self.regimes))}
                 
        if mock:
            if mock_dict is None:
                raise TypeError("Need mock dict in order to use mockseries")
            
            bases = mock_dict.get('bases')
            amplitudes=mock_dict.get('amplitudes')
            offsets=mock_dict.get('offsets')
            shock_times = mock_dict.get('shock_times')
            
            if len(bases) != len(amplitudes) != self.regimes:
                raise ValueError("mock dict parameters must be equal and equal to number of regimes.")
            
            df = pd.DataFrame(
                index=pd.DatetimeIndex(self.dates, name='time'),
                columns=[f'drought_{i}' for i in range(self.regimes)],
            )
            
            for shock, base, amplitude, offset, r, alpha_r in \
                zip(shock_times, bases, amplitudes, offsets, range(self.regimes), alpha_r.values()):
                trend = LinearTrend(coefficient=0, time_unit=timedelta(days=30), #.1
                                                flat_base=base) #1
                seasonality = SinusoidalSeasonality(amplitude=amplitude, period=timedelta(days=90), offset=offset) #2
                noise = RedNoise(mean=0, std=1, correlation=.5)
                            
                trans = LinearTransition(transition_window=timedelta(days=30), stop_window=timedelta(days=5))

                switch = Switch(
                    start_time = self.dates[shock[0]],
                    base_value= 0,
                    switch_value= -5,
                    stop_time=self.dates[shock[1]],
                    transition=trans
                )

                timeseries = trend + seasonality + noise + switch

                df[f"drought_{r}"] = timeseries.generate(self.dates.tolist()) + alpha_r
            
            return df

            
        else:
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
                                          cov = None,
                                          mock=False,
                                          mock_dict=None):
        
        df_list = []
        
        # Create drought data
        drought_df = self._create_drought_index(mean = mean, cov=cov, mock=mock, mock_dict=mock_dict)
        
        drought_cols = [f"drought_{i}" for i in range(self.regimes)]
        
        # Create list of the same data but with driver ids
        for i in range(self.drivers):
            df_list.append(drought_df.assign(driver = i))

        driver_time_df = pd.concat(df_list).set_index('driver', append=True)
        
        driver_time_df.index.names = ['time', 'driver']
        
        return (
            driver_time_df
            .reset_index()
            .pipe(lagged_drought_df, drought_cols, 
                          shift=self.lags, groupby_index='driver', date_col='time')
            .drop(columns=[f"drought_{i}" for i in range(self.regimes)])
            .set_index(['driver', 'time'])
            .reorder_levels(['driver', 'time'])
            )
    
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
            if self.lags is not None:
                raise ValueError("Must provide a list of size `len(lags)`")
            else:
                beta1 = [beta1]*self.regimes
        else:
            if len(self.lags) != len(beta1):
                raise ValueError("size of betas and number of lags must be equal")
            beta1 = np.array(beta1)
        
        def y_lambda(i):
            return lambda df: beta0[i] + df.filter(regex=rf"lagged_\d_drought_{i}").values @ beta1 \
                + self.random.normal(0, sd[i], self.time_periods*self.drivers) #TODO: Is this heteroskedasticity?
        
        assign_dict = {name + f'_{i}' : y_lambda(i) for i in range(self.regimes)}
        
        df = (
            data
            .assign(**assign_dict)
        )
        
        return df
    
    def regime_dummies(self, regime, weight):
        
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
        
        return regime_dummies
        
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
                  driver_fe=False,
                  time_fe=False,
                  month_year_fe=False,
                  shift=None,
                  mock=False,
                  mock_dict=None
                  ):
        
        if seed is not None:
            self.random = np.random.default_rng(seed=seed)
        
        # Create drought index
        drought = self._create_drought_index_with_driver(mean = drought_mean,
                                                         cov = drought_cov,
                                                         mock=mock,
                                                         mock_dict=mock_dict)
        
        # Create randomly assigned regime membership
        regime = pd.concat([self._create_driver_index(), 
                             self._create_regime_index()], 
                           axis=1)
        
        # Get weight matrix for misclassification
        mw = self.misclassification_weight(self.regimes, weight)

        if jittered:
            regime_dummies = self.regime_dummies(self, regime, weight)
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
            
        driver_ids = list(range(self.drivers))
        time_period_list = self.dates
        month_list = self.dates.month
        year_list = self.dates.year
            
        # Create fixed effects
        if driver_fe:
            alpha_i = {k:v for k,v in zip(driver_ids, self.random.normal(0,1, size=self.drivers))}
        else:
            alpha_i = {k:0 for k in driver_ids}
        if time_fe:
            alpha_t = {k:v for k,v in zip(time_period_list, self.random.normal(0,1, size=self.time_periods))}
        else:
            alpha_t = {k:0 for k in time_period_list}
            
        if month_year_fe:
            alpha_m = {k:v for k,v in zip(month_list, self.random.normal(0,1, size=len(month_list)))}
            alpha_y = {k:v for k,v in zip(year_list, self.random.normal(0,1, size=len(year_list)))}
        else:
            alpha_m = {k:0 for k in month_list}
            alpha_y = {k:0 for k in year_list}
           
        df = (
            df
            .assign(alpha_t = df.index.get_level_values('time').map(alpha_t),
                    alpha_i = df.index.get_level_values('driver').map(alpha_i),
                    alpha_m = df.index.get_level_values('time').month.map(alpha_m),
                    alpha_y = df.index.get_level_values('time').year.map(alpha_y),
                    y = lambda df: df['y'] + df['alpha_i'] + df['alpha_t'] + df['alpha_m'] + df['alpha_y']
                    )
        )
            
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
        

class UberDatasetCreatorHet:
    
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
        
        
    
    
        
    
        
        
        
    
        
    
        
        
        