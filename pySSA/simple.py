import pandas as pd

from pySSA.core import MSSA
from pySSA.plotting import plot


class SSA(object):
    '''
    External wrapper for MSSA method with graphs from plotly
    '''
    def __init__(self, timeseries):
        self.time_series = timeseries
        self.mssa = MSSA(timeseries)
        self.n_series = self.mssa.ts_s
        self.steps = None
        self.forecast = None
        self.conf_intervals = None

    def decompose(self, embedding_dimension=None, return_df=False):
        self.mssa.embed(embedding_dimension=embedding_dimension)
        self.mssa.decompose()
        decomposed_ts = self.mssa.diag_procedure()
        if return_df:
            return pd.DataFrame(decomposed_ts)

    def RLforecast(self, groups, steps, conf_int=False):
        self.steps = steps
        self.mssa.group_components(groups)
        self.forecast = self.mssa.L_reccurent_forecast(steps)
        self.conf_intervals = self.mssa.conf_int()
        if conf_int:
            return pd.concat([self.forecast, self.conf_intervals], axis=1)
        else:
            return self.forecast

    def RKforecast(self, groups, steps, conf_int=False):
        self.steps = steps
        self.mssa.group_components(groups)
        self.forecast = self.mssa.K_reccurent_forecast(steps)
        self.conf_intervals = self.mssa.conf_int()
        if conf_int:
            return pd.concat([self.forecast, self.conf_intervals])
        else:
            return self.forecast

    def plot(self,
             plot_series,
             test=None,
             index=None,
             title='Plot',
             x_ax='x',
             y_ax='y',
             name_train='Real_',
             name_train_foreacst='Trainig_',
             name_test='Test_',
             name_test_forecast='Forecast_',
             plot_conf_int=True,
             save_html_path=None):

        plot(ssa=self,
             plot_series=plot_series,
             test=test,
             index=index,
             title=title,
             x_ax=x_ax,
             y_ax=y_ax,
             name_train=name_train,
             name_train_foreacst=name_train_foreacst,
             name_test=name_test,
             name_test_forecast=name_test_forecast,
             plot_conf_int=plot_conf_int,
             save_html_path=save_html_path)