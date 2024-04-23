import matplotlib.pyplot as plt
import numpy as np
from .regression import get_reg_effect
from ml_switching_reg.mle import get_mle_betas

class Visualizer:
    
    def __init__(self, data, reg, mle, true_beta0, true_beta1, *other_mle):
        
        self.reg = reg
        self.mle = mle
        
        self.data = data
        
        self.beta0_reg, self.beta1_reg, _ =  get_reg_effect(self.reg)
        
        self.n_regimes = self.mle.model.n_regimes
        
        self.beta0_mle, self.beta1_mle, _ = get_mle_betas(self.mle, regimes = self.n_regimes)
                
        
        self.true_beta0 = true_beta0
        self.true_beta1 = true_beta1
        
               
    def _plot_line(self, ax, beta0, beta1, labels, linestyle, color='black'):
        
        # Create x data
        x = np.linspace(*ax.get_xlim(), num=5)
        
        for b0, b1, l in zip(beta0, beta1, labels):
            
            ax.plot(x, b0 + b1*x, label = l, linestyle = linestyle, color=color)
            
    def _plot_reg_line(self, ax, labels, linestyle):
        """Plots regression lines after estimation

        Args:
            ax (matplotlib axis object): The plot with the scatter
        """
        
        # Assuming here that misclass_regime_{i} is the beta0 and misclass_regime_{i}:drought_{i}
        # is the beta1
        
        self._plot_line(ax=ax, 
                        beta0 = self.beta0_reg, 
                        beta1=self.beta1_reg, 
                        labels = labels, 
                        linestyle=linestyle,
                        color='red')
        
    def _plot_mle_line(self, ax, labels, linestyle):
        
        self._plot_line(ax=ax, 
                        beta0 = self.beta0_mle, 
                        beta1=self.beta1_mle, 
                        labels = labels, 
                        linestyle=linestyle,
                        color='green')
 
        
    def _create_scatter(self, ax, data, y_name = None):
        
        # First create composite drought variable
        
        ## Create string for df.eval
        
        eval_string = f"drought = "
        
        eval_list = []
                
        for i in range(self.n_regimes):
            
            eval_list.append(f"regime_{i}*drought_{i}")
        
        df = data.eval(eval_string + '+ '.join(eval_list))
        
        df.plot.scatter(
            x = 'drought',
            y = y_name,
            c = 'regime',
            colormap = 'tab10',
            colorbar = False,
            alpha=0.2,
            ax=ax
            )
        
    def _create_table(self, ax, other_beta0=None, other_beta1=None, other_title=None):
        
        rows = ['True', 'Reg', 'MLE']
        
        columns = ['$\\beta_0$', '$\\beta_1$']
                
        cell_text = [
            [self.true_beta0, self.true_beta1],
            np.round([self.beta0_reg, self.beta1_reg], 3),
            np.round([self.beta0_mle, self.beta1_mle],3)
        ]
        
        if other_title is not None:
            rows.append(other_title)
            cell_text.append(np.round([other_beta0, other_beta1],3))
        
        table = ax.table(
            rowLabels = rows,
            colLabels = columns,
            cellText = cell_text,
            loc='center'
        )
        
        table.auto_set_column_width(col = [0,1])
        # table.auto_set_font_size(False)
        # table.set_fontsize(12)
        table.scale(2,2)
        
        
    def plot(self, 
             plot_true = False, 
             plot_reg = False, 
             plot_mle = False, 
             ax=None, 
             fig=None, 
             y_name = 'y',
             other_beta0=None,
             other_beta1=None,
             other_title=None):
        
        if ax is None:
            fig, ax = plt.subplots(ncols=2, figsize = (10,6),
                                            gridspec_kw=dict(width_ratios=[3,1]))                    
        # Create Scatter
        self._create_scatter(ax=ax[0], data=self.data, y_name = y_name)
        
        if plot_true:
        # Plots true lines
            self._plot_line(ax=ax[0], 
                            beta0 = self.true_beta0, 
                            beta1 = self.true_beta1,
                            labels = [f"True {i}" for i in range(self.n_regimes)],
                            linestyle='--')
            
        if plot_reg:
            self._plot_reg_line(ax=ax[0], 
                                labels = [f"Reg {i}" for i in range(self.n_regimes)],
                                linestyle='-.'
                                )
        
        if plot_mle:
            self._plot_mle_line(ax=ax[0], 
                                labels = [f"MLE {i}" for i in range(self.n_regimes)],
                                linestyle= ':'
                                )
            
        if other_beta0 is not None:
            self._plot_line(ax=ax[0], 
                            beta0 = other_beta0, 
                            beta1 = other_beta1,
                            labels = [f"{other_title} {i}" for i in range(self.n_regimes)],
                            linestyle='--',
                            color='blue')
        
        self._create_table(ax=ax[1], other_beta0=other_beta0, other_beta1=other_beta1, other_title=other_title)
        
        ax[1].axis("off")
                
        fig.legend()
                
        return ax
        
            
    