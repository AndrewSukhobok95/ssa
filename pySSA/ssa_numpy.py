import pandas as pd
import numpy as np
from numpy import matrix as m
from scipy import linalg
from scipy.stats import norm

from plotly.graph_objs import *
import plotly.offline as offline

class MSSA(object):
    '''Multi-channel Singular Spectrum Analysis object

    SSA class take one positional argument – timeseries.

    :param timeseries: type can be pandas.DataFrame, pandas.Series, numpy.array, numpy.matrix, list'''

    def __init__(self, time_series):
        self.ts_df = pd.DataFrame(time_series).reset_index(drop=True)
        self.ts = m(self.ts_df)
        self.ts_N = self.ts.shape[0]
        self.ts_s = self.ts.shape[1]
        self.ts_name = self.ts_df.columns.tolist()

    def _hankelSeries(self, series):
        '''Perform hankelization procedure for given series'''
        return linalg.hankel(series, np.zeros(self.embedding_dimension)).T[:, :self.K]

    def embed(self, embedding_dimension=None):
        '''Compute the trajectory matrix of given time series.

        :param embedding_dimension: How many components to compute from the original series. Default is N//2, where N stands for length of series.'''
        if not embedding_dimension:
            self.embedding_dimension = self.ts_N // 2
        else:
            self.embedding_dimension = embedding_dimension
        self.K = self.ts_N - self.embedding_dimension + 1
        series = np.hsplit(self.ts, self.ts_s)
        X = np.hstack(map(self._hankelSeries, series))
        self.X = m(X)
        self.trajectory_dimentions = X.shape

    def decompose(self):
        '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace.'''
        X = self.X
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
        self.d = np.linalg.matrix_rank(X)
        self.Vs = (X.T * self.U) / self.s
        U_s = m(np.array(self.U) * self.s)
        Xs = np.empty((self.embedding_dimension, 0))
        for i in range(self.d):
            Xs = np.hstack((Xs, U_s[:, i] * self.Vs[:, i].T))
        self.Xs = Xs

    @staticmethod
    def _diagonal_averaging(hankel_matrix):
        '''Performs anti-diagonal averaging from given hankel matrix

        :param embedding_dimension: Trajectory matrix of one-dimensional time series.

        :return: pandas.DataFrame with decomposed series.'''
        mat = m(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L, K), max(L, K)
        new = np.zeros((L, K))
        if L > K:
            mat = mat.T
        ret = []
        # Diagonal Averaging
        for k in range(1 - K_star, L_star):
            mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
            mask_n = sum(sum(mask))
            ma = np.ma.masked_array(mat.A, mask=1 - mask)
            ret += [ma.sum() / mask_n]
        return ret

    def _d_series_diag(self, Xd):
        '''Diagonal averaging for d series matrix'''
        sreries = list(map(self._diagonal_averaging, np.hsplit(Xd, self.ts_s)))
        return np.hstack(sreries)

    def diag_procedure(self):
        '''Performs anti-diagonal averaging for multidimensional time series.'''
        _hankel_list = np.hsplit(self.Xs, self.d)
        _big_vector = np.vstack(list(map(self._d_series_diag, _hankel_list))).T
        return np.hstack(np.vsplit(_big_vector, self.ts_s))

    def group_components(self, r, return_df=False):
        '''Compute the sum of first r chosen components from decomposed series (reconstruction procedure).

        :param r: The number of components for series reconstruction.

        :param return_df: If True, then return the pandas.DataFrame with reconstructed series. 

        :return: pandas.DataFrame with reconstructed series.'''
        self.r = r
        C = self.diag_procedure()
        C_grouped = np.hsplit(C, self.ts_s)
        res = []
        for i in range(len(C_grouped)):
            res.append(
                np.sum(C_grouped[i][:, :r], axis=1)
            )

        ### Resids part ###############
        resids = []
        for i in range(len(C_grouped)):
            resids.append(
                np.sum(C_grouped[i][:, r:], axis=1)
            )
        self.resids = m(resids)
        ###############################

        self.C_grouped = m(res)
        if return_df == True:
            return pd.DataFrame(res, index=['Grouped_component_' + str(i) for i in self.ts_name]).T

    def L_reccurent_forecast(self, steps_ahead):
        '''Compute the recurrent forecast based on columns (MSSA-L).

        :param steps_ahead: The length of the forecast (how many steps ahead to compute).

        :return: pandas.DataFrame with reconstructed series and their forecaasts.'''

        r = self.r
        v_2 = 0
        for i in range(r):
            v_2 += (self.U[-1, i]) ** 2
        R_L_sum = 0
        for i in range(r):
            R_L_sum += self.U[-1, i] * self.U[:-1, i]
        R_L = (1 / (1 - v_2)) * R_L_sum
        C_grp_forc = self.C_grouped
        N = self.ts_N
        for i in range(steps_ahead):
            Z = C_grp_forc[:, N - self.embedding_dimension + 1:]
            R_N = np.dot(Z, R_L)
            C_grp_forc = np.hstack((C_grp_forc, R_N))
            N += 1

        self.forc = C_grp_forc
        self.reccurent_coef = R_L
        self.steps_ahead = steps_ahead
        return pd.DataFrame(C_grp_forc, index=['Forecast_' + str(i) for i in self.ts_name]).T

    def K_reccurent_forecast(self, steps_ahead):
        '''Compute the recurrent forecast based on rows (MSSA-K).

        :param steps_ahead: The length of the forecast (how many steps ahead to compute).

        :return: pandas.DataFrame with reconstructed series and their forecasts.'''
        r = self.r
        K = self.K
        # V = np.hstack([i[1] for i in sorted(self.Vs.items(), key=lambda x: x[0])])
        V = self.Vs
        S = m([]).reshape((0, r))
        for i in range(self.ts_s):
            S = np.vstack((S, V[(i + 1) * K - 1, :r]))
        V_dropped = np.delete(V[:, :r], [i * K - 1 for i in range(1, self.ts_s + 1)], 0)

        I = np.zeros((self.ts_s, self.ts_s), int)
        np.fill_diagonal(I, 1)
        inv_part_R_K = linalg.inv(I - S * S.T)
        R_K = inv_part_R_K * S * V_dropped.T
        self.reccurent_coef = R_K
        C_grp_forc = self.C_grouped
        N = self.ts_N
        for i in range(steps_ahead):
            Z_m = C_grp_forc[:, N - K + 1:]
            Z = Z_m.reshape((Z_m.shape[0] * Z_m.shape[1], 1))
            R_N = np.dot(R_K, Z)
            C_grp_forc = np.hstack((C_grp_forc, R_N))
            N += 1
        self.forc = C_grp_forc
        return pd.DataFrame(C_grp_forc, index=['Forecast_' + str(i) for i in self.ts_name]).T

    @staticmethod
    def _recursive_coef_calc(a, steps):
        a = np.array(sum(a.tolist(), []))
        psy_list = []
        psy = np.zeros((len(a), 1))
        psy[0] = 1
        for i in range(steps):
            psy_j = np.dot(a, psy).prod()
            psy = np.roll(psy, 1)
            psy[0] = psy_j
            psy_list.append(psy_j ** 2)
        return psy_list

    def conf_int(self):
        '''
        Build the confidence intervals for all forecasted series

        :return: pandas DataFrame with conf. intervals in the same order as series in original table
        '''
        intervals = []
        for r in range(len(self.resids)):
            mu, std = norm.fit(np.squeeze(np.asarray(self.resids[r])))
            p5 = np.percentile(np.squeeze(np.asarray(self.resids[r])) - mu, 5)
            p95 = np.percentile(np.squeeze(np.asarray(self.resids[r])) - mu, 95)

            recursive_coefs = self._recursive_coef_calc(self.reccurent_coef, self.steps_ahead)
            recursive_coefs = [1] + recursive_coefs
            forc_interval = []
            for i in range(self.steps_ahead):
                forc_interval.append(
                    sum(recursive_coefs[:i+1])*(std**2)
                )

            upper_bound_in = (np.squeeze(np.array(self.forc[r, :-self.steps_ahead])) + p95).tolist()
            lower_bound_in = (np.squeeze(np.array(self.forc[r, :-self.steps_ahead])) + p5).tolist()
            upper_bound_out = (np.squeeze(np.array(self.forc[r, -self.steps_ahead:])) + np.sqrt(np.array(forc_interval))*1.96).tolist()
            lower_bound_out = (np.squeeze(np.array(self.forc[r, -self.steps_ahead:])) - np.sqrt(np.array(forc_interval))*1.96).tolist()
            intervals.append(lower_bound_in + lower_bound_out)
            intervals.append(upper_bound_in + upper_bound_out)
        return pd.DataFrame(intervals).T



class SSA(object):
    '''external cover for MSSA method with graphs from plotly'''

    def __init__(self, timeseries):
        self.time_series = timeseries
        self.ssa = MSSA(timeseries)
        self.n_series = self.ssa.ts_s

    def decompose(self, embedding_dimension=None, return_df=False):
        self.ssa.embed(embedding_dimension=embedding_dimension)
        self.ssa.decompose()
        decomposed_ts = self.ssa.diag_procedure()
        if return_df:
            return decomposed_ts

    def RLforecast(self, groups, steps, conf_int=False):
        self.steps = steps
        self.ssa.group_components(groups)
        self.forecast = self.ssa.L_reccurent_forecast(steps)
        self.conf_intervals = self.ssa.conf_int()
        if conf_int:
            return pd.concat([self.forecast, self.conf_intervals])
        else:
            return self.forecast

    def RKforecast(self, groups, steps, conf_int=False):
        self.steps = steps
        self.ssa.group_components(groups)
        self.forecast = self.ssa.K_reccurent_forecast(steps)
        self.conf_intervals = ssa.conf_int()
        if conf_int:
            return pd.concat([self.forecast, self.conf_intervals])
        else:
            return self.forecast

    def plot(self, plot_series, test=None, index=None,
             title='Plot', x_ax='', y_ax='',
             name_train='Real_', name_train_foreacst='Trainig_', name_test='Test_', name_test_forecast='Forecast_',
             plot_conf_int=True,
             output_folder='', file_name='plot'):

        train = self.ssa.ts_df.copy()
        train_forecast = self.forecast.iloc[:-self.steps, :].copy()
        test_forecast = self.forecast.iloc[-self.steps:, :].copy()
        if plot_conf_int:
            conf_int_df = self.conf_intervals

        if index is not None:
            index_train = index[:-self.steps]
            index_test = index[-self.steps:]
            train.index = index_train
            train_forecast.index = index_train
            test_forecast.index = index_test
            if test is not None:
                test.index = index_test

        trace_list = []

        for i in plot_series:
            trace_train = Scatter(
                y=train.iloc[:, i],
                x=train.iloc[:, i].index,
                mode='lines+markers',
                name=name_train + str(train.iloc[:, i].name)
            )
            trace_train_forecast = Scatter(
                y=train_forecast.iloc[:, i],
                x=train_forecast.iloc[:, i].index,
                mode='lines+markers',
                name=name_train_foreacst + str(train.iloc[:, i].name)
            )
            trace_test_forecast = Scatter(
                y=test_forecast.iloc[:, i],
                x=test_forecast.iloc[:, i].index,
                mode='lines+markers',
                name=name_test_forecast + str(train.iloc[:, i].name)
            )
            if test is not None:
                trace_test = Scatter(
                    y=test.iloc[:, i],
                    x=test.iloc[:, i].index,
                    mode='lines+markers',
                    name=name_test + str(train.iloc[:, i].name)
                )
                trace_list.append(trace_test)
            trace_list = trace_list + [trace_train, trace_train_forecast, trace_test_forecast]
            if plot_conf_int:
                trace_conf_5 = Scatter(
                    y=conf_int_df.iloc[:, 2 * i],
                    x=train_forecast.index.tolist() + test_forecast.index.tolist(),
                    line=dict(
                        color=('black'),
                        width=1,
                        dash='dash'),
                    name='Conf. Int. 5%'
                )
                trace_conf_95 = Scatter(
                    y=conf_int_df.iloc[:, 2 * i + 1],
                    x=train_forecast.index.tolist() + test_forecast.index.tolist(),
                    line=dict(
                        color=('black'),
                        width=1,
                        dash='dash'),
                    name='Conf. Int. 95%'
                )
                trace_list += [trace_conf_5, trace_conf_95]

        layout = Layout(
            title=title,
            xaxis=dict(title=x_ax),
            yaxis=dict(title=y_ax),
            legend=dict(
                font=dict(
                    size=12
                )
            )
        )
        fig = Figure(data=trace_list, layout=layout)
        offline.plot(fig, filename=output_folder + file_name + '.html')



if __name__=="__main__":

    ts = pd.read_csv('../data/air_ssa.csv', parse_dates=True, index_col=0)
    ssa = SSA(ts)
    ssa.decompose(20)
    ssa.RLforecast(7, steps=50)
    ssa.plot(plot_series=[0], title='Plot Name', x_ax='Day', y_ax='Value', output_folder='../data/', file_name='plot1')

    ts = pd.read_csv('../data/art_series.csv', parse_dates=True, index_col=0)
    ssa = SSA(ts)
    ssa.decompose(20)
    ssa.RLforecast(7, steps=50)
    ssa.plot(plot_series=[0, 1, 2], title='Plot Name', x_ax='Day', y_ax='Value', output_folder='../data/', file_name='plot2')

    # ts = pd.read_csv('data/air_ssa.csv', parse_dates=True, index_col=0)
    # ts = pd.read_csv('data/art_series.csv', parse_dates=True, index_col=0)
    # ssa = MSSA(ts)
    # ## Decomposition
    # ssa.embed(embedding_dimension=2)
    # ssa.decompose()
    # b = ssa.diag_procedure()
    #
    # ## Forecasting Reccurent L
    # ssa.embed(embedding_dimension=20)
    # ssa.decompose()
    # ssa.group_components(5)
    # forc = ssa.L_reccurent_forecast(40)
    # inter = ssa.conf_int()
    # forc.plot()
    # plt.show(block=False)
    #
    # ###Forecasting Reccurent K
    # ssa.embed(embedding_dimension=10)
    # ssa.decompose()
    # ssa.group_components(1)
    # forc = ssa.K_reccurent_forecast(40)
    # forc.plot()
    # plt.show(block=False)
