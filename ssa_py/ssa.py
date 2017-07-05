import numpy as np
import pandas as pd
from numpy import matrix as m
from pandas import DataFrame as df
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.offline as offline

class MSSA(object):

    '''Multi-channel Singular Spectrum Analysis object'''

    def __init__(self, time_series):
        self.ts = pd.DataFrame(time_series).reset_index(drop=True)
        self.ts_name = self.ts.columns.tolist()
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
        self.ts_M = self.ts.shape[1]

    def view_time_series(self, norm=False):
        '''Plot the time series'''
        if not norm:
            self.ts.plot(title='Original Time Series')
            plt.show(block=False)
        else:
            ts_norm = (ts - ts.mean()) / (ts.max() - ts.min())
            ts_norm.plot(title='Original Time Series')
            plt.show(block=False)

    # @staticmethod
    # def _dot(x, y):
    #     '''Alternative formulation of dot product to allow missing values in arrays/matrices'''
    #     pass

    def embed(self, embedding_dimension=None):
        '''Embed the time series with embedding_dimension window size.'''
        if not embedding_dimension:
            self.embedding_dimension = self.ts_N // 2
        else:
            self.embedding_dimension = embedding_dimension

        self.K = self.ts_N - self.embedding_dimension + 1
        matrix_list = []
        for i in range(self.ts_M):
            subX = m(linalg.hankel(self.ts.iloc[:, i], np.zeros(self.embedding_dimension))).T[:, :self.K]
            matrix_list.append(subX)
        self.X = np.hstack(matrix_list)
        self.X_df = df(self.X)
        self.trajectory_dimentions = self.X_df.shape

    def decompose(self):
        '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace
        Characteristic of projection: the proportion of variance captured in the subspace'''
        X = self.X
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
        self.d = np.linalg.matrix_rank(X)
        Vs, Xs, Ys = {}, {}, {}
        for i in range(self.d):
            Vs[i] = (X.T * self.U[:, i]) / self.s[i]
            Ys[i] = self.s[i] * self.U[:, i]
            Xs[i] = Ys[i] * (m(Vs[i]).T)
        self.Vs, self.Xs = Vs, Xs

    @staticmethod
    def diagonal_averaging(hankel_matrix):
        '''Performs anti-diagonal averaging from given hankel matrix
        Returns: Pandas DataFrame object containing the reconstructed series'''
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
        return df(ret).rename(columns={0: 'Reconstruction'})

    @classmethod
    def sequential_diagonal_averaging(cls, Xs, K):
        '''Perform anti-dioganal averaging for every hankel matrix of all series'''
        decomposed_series = []
        n_series = Xs[0].shape[1] // K
        for i in range(n_series):
            for j in range(len(Xs)):
                hank_matr = Xs[j][:, i * K:(i + 1) * K]
                decomposed_series.append(
                    cls.diagonal_averaging(hank_matr).rename(
                        columns={'Reconstruction': 'Component_' + str(i) + '_' + str(j)})
                )
        return pd.concat(decomposed_series, axis=1)

    def group_components(self, r, return_df=False):
        '''Grouping components for recurrent forecasting.
        Caution: this grouping is using only for forecasting, but
        decomposition (diagonal_averaging step) is applied to simple
        components without grouping
        r - how many components to group'''
        self.r = r
        Xs = self.Xs
        K = self.K
        L = self.embedding_dimension
        n_series = Xs[0].shape[1] // K
        self.C = self.sequential_diagonal_averaging(Xs, K)
        Cm = m(self.C)
        grouped_C = []
        for i in range(n_series):
            hank_matr = Cm[:, i * L:(i + 1) * L]
            grouped_C.append(
                np.squeeze(np.asarray(
                    hank_matr[:, :r].sum(axis=1)
                ))
            )
        self.C_grouped = m(df(grouped_C))
        if return_df == True:
            return df(grouped_C, index=['Grouped_component_'+str(i) for i in self.ts_name]).T

    def L_reccurent_forecast(self, steps_ahead):
        '''Perform MSSA-L reccurent forecast for 'steps_ahead' steps ahead.
        L - usually refer to Column based forecast by left singular vectors U'''
        r = self.r
        v_2 = 0
        for i in range(r):
            v_2 += (self.U[-1, i])**2
        R_L_sum = 0
        for i in range(r):
            R_L_sum += self.U[-1, i]*self.U[:-1, i]
        R_L = (1/(1-v_2))*R_L_sum
        C_grp_forc = self.C_grouped
        N = self.ts_N
        for i in range(steps_ahead):
            Z = C_grp_forc[:, N - self.embedding_dimension + 1:]
            R_N = np.dot(Z,R_L)
            C_grp_forc = np.hstack((C_grp_forc, R_N))
            N += 1
        return df(C_grp_forc, index=['Forecast_'+str(i) for i in self.ts_name]).T

    def K_reccurent_forecast(self, steps_ahead):
        '''Perform MSSA-K reccurent forecast for 'steps_ahead' steps ahead.
        K - usually refer to Row based forecast by right singular vectors V'''
        r = self.r
        K = self.K
        V = np.hstack([i[1] for i in sorted(self.Vs.items(), key=lambda x: x[0])])
        S = m([]).reshape((0,r))
        for i in range(self.ts_M):
            S = np.vstack((S, V[(i+1)*K-1, :r]))
        V_dropped = np.delete(V[:, :r], [i*K - 1 for i in range(1, self.ts_M+1)], 0)
        I = np.zeros((self.ts_M, self.ts_M), int)
        np.fill_diagonal(I, 1)
        inv_part_R_K = linalg.inv(I-S*S.T)
        R_K = inv_part_R_K*S*V_dropped.T
        C_grp_forc = self.C_grouped
        N = self.ts_N
        for i in range(steps_ahead):
            Z_m = C_grp_forc[:, N - K + 1:]
            Z = Z_m.reshape((Z_m.shape[0]*Z_m.shape[1], 1))
            R_N = np.dot(R_K,Z)
            C_grp_forc = np.hstack((C_grp_forc, R_N))
            N += 1
        return df(C_grp_forc, index=['Forecast_'+str(i) for i in self.ts_name]).T

    # def L_vector_forecast(self, steps_ahead):
    #     '''Perform MSSA vector forecast (V-MSSA) by columns for 'steps_ahead' steps ahead.'''
    #     r = self.r
    #     K = self.K
    #     V = np.hstack([i[1] for i in sorted(self.Vs.items(), key=lambda x: x[0])])
    #     U = self.U
    #     s = self.s
    #     #
    #     P = U[:, :r]
    #     Q_list = []
    #     for i in range(r):
    #         Q_list.append(s[i]*V[:, i])
    #     Q = np.hstack(Q_list)
    #     W = Q.T
    #     I = np.zeros((r, r), int)
    #     np.fill_diagonal(I, 1)
    #     pi = P[-1, :]
    #     D = (I - pi*pi.T/(1 - pi.T*pi))*P[:-1,:].T*P[:-1,:]
    #     W_list = []
    #     W_k_1 = W.copy()
    #     for i in range(steps_ahead + self.embedding_dimension - 2):
    #         W_k = np.dot(D,W_k_1)
    #         W_k_1 = W_k.copy()
    #         W_list.append(W_k)
    #     Z = np.dot(P, np.hstack([W]+W_list))
    #     Z_list = []
    #     for i in range(self.ts_M):
    #         Z_list.append(
    #             self.diagonal_averaging(
    #                 Z[:, i * K * (steps_ahead + self.embedding_dimension - 1):(i + 1) * K * (steps_ahead + self.embedding_dimension - 1) - 1]
    #             )
    #         )
    #     return Z_list


class SSA(object):
    '''external cover for MSSA method with graphs from plotly'''
    def __init__(self, timeseries):
        self.time_series = timeseries
        self.ssa = MSSA(timeseries)
        self.n_series = self.ssa.ts_M

    def decompose(self, embedding_dimension=None, return_df=False):
        self.ssa.embed(embedding_dimension=embedding_dimension)
        self.ssa.decompose()
        decomposed_ts = self.ssa.sequential_diagonal_averaging(self.ssa.Xs, self.ssa.K)
        if return_df:
            return decomposed_ts

    def RLforecast(self, groups, steps):
        self.steps = steps
        self.ssa.group_components(groups)
        self.forecast = self.ssa.L_reccurent_forecast(steps)
        return self.forecast

    def RKforecast(self, groups, steps):
        self.steps = steps
        self.ssa.group_components(groups)
        self.forecast = self.ssa.K_reccurent_forecast(steps)
        return self.forecast

    def plot(self, plot_series, test=None, index=None,
             title='Plot', x_ax='', y_ax='',
             name_train='Real_', name_train_foreacst='Trainig_', name_test='Test_', name_test_forecast='Forecast_',
             output_folder='', file_name='plot'):

        train = self.ssa.ts.copy()
        train_forecast = self.forecast.iloc[:-self.steps, :].copy()
        test_forecast = self.forecast.iloc[-self.steps:, :].copy()

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





# #########################
#
# ts = pd.read_csv('data/art_series.csv', parse_dates=True, index_col=0)
# ssa = MSSA(ts)
#
# # ts = pd.read_csv('data/f2.csv')#, parse_dates=True, index_col=0)
# # ssa = MSSA(ts)
#
#
#
# ssa = MSSA(ts)
#
# ### Decomposition
#
# ssa.embed(embedding_dimension=5)
# ssa.decompose()
# a = ssa.sequential_diagonal_averaging(ssa.Xs, ssa.K)
#
# ### Forecasting Reccurent L
#
# ssa.embed(embedding_dimension=5)
# ssa.decompose()
# ssa.group_components(2)
# a = ssa.L_reccurent_forecast(40)
#
# ###Forecasting Reccurent K
#
# ssa.embed(embedding_dimension=5)
# ssa.decompose()
# ssa.group_components(2)
# a = ssa.K_reccurent_forecast(40)
#
#
#
























# Code from https://github.com/aj-cloete/pySSA
# Current code is partly based on this source

# class mySSA(object):
#     '''Singular Spectrum Analysis object'''
#
#     def __init__(self, time_series):
#
#         self.ts = pd.DataFrame(time_series)
#         self.ts_name = self.ts.columns.tolist()[0]
#         if self.ts_name == 0:
#             self.ts_name = 'ts'
#         self.ts_v = self.ts.values
#         self.ts_N = self.ts.shape[0]
#         self.freq = self.ts.index.inferred_freq
#
#     @staticmethod
#     def _printer(name, *args):
#         '''Helper function to print messages neatly'''
#         print('-' * 40)
#         print(name + ':')
#         for msg in args:
#             print(msg)
#
#     @staticmethod
#     def _dot(x, y):
#         '''Alternative formulation of dot product to allow missing values in arrays/matrices'''
#         pass
#
#     @staticmethod
#     def get_contributions(X=None, s=None, plot=True):
#         '''Calculate the relative contribution of each of the singular values'''
#         lambdas = np.power(s, 2)
#         frob_norm = np.linalg.norm(X)
#         ret = df(lambdas / (frob_norm ** 2), columns=['Contribution'])
#         ret['Contribution'] = ret.Contribution.round(4)
#         if plot:
#             ax = ret[ret.Contribution != 0].plot.bar(legend=False)
#             ax.set_xlabel("Lambda_i")
#             ax.set_title('Non-zero contributions of Lambda_i')
#             vals = ax.get_yticks()
#             ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
#             return ax
#         return ret[ret.Contribution > 0]
#
#     @staticmethod
#     def diagonal_averaging(hankel_matrix):
#         '''Performs anti-diagonal averaging from given hankel matrix
#         Returns: Pandas DataFrame object containing the reconstructed series'''
#         mat = m(hankel_matrix)
#         L, K = mat.shape
#         L_star, K_star = min(L, K), max(L, K)
#         new = np.zeros((L, K))
#         if L > K:
#             mat = mat.T
#         ret = []
#
#         # Diagonal Averaging
#         for k in range(1 - K_star, L_star):
#             mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
#             mask_n = sum(sum(mask))
#             ma = np.ma.masked_array(mat.A, mask=1 - mask)
#             ret += [ma.sum() / mask_n]
#
#         return df(ret).rename(columns={0: 'Reconstruction'})
#
#     def view_time_series(self):
#         '''Plot the time series'''
#         self.ts.plot(title='Original Time Series')
#
#     def embed(self, embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False):
#         '''Embed the time series with embedding_dimension window size.
#         Optional: suspected_frequency changes embedding_dimension such that it is divisible by suspected frequency'''
#         if not embedding_dimension:
#             self.embedding_dimension = self.ts_N // 2
#         else:
#             self.embedding_dimension = embedding_dimension
#         if suspected_frequency:
#             self.suspected_frequency = suspected_frequency
#             self.embedding_dimension = (self.embedding_dimension // self.suspected_frequency) * self.suspected_frequency
#
#         self.K = self.ts_N - self.embedding_dimension + 1
#         self.X = m(linalg.hankel(self.ts, np.zeros(self.embedding_dimension))).T[:, :self.K]
#         self.X_df = df(self.X)
#         self.X_complete = self.X_df.dropna(axis=1)
#         self.X_com = m(self.X_complete.values)
#         self.X_missing = self.X_df.drop(self.X_complete.columns, axis=1)
#         self.X_miss = m(self.X_missing.values)
#         self.trajectory_dimentions = self.X_df.shape
#         self.complete_dimensions = self.X_complete.shape
#         self.missing_dimensions = self.X_missing.shape
#         self.no_missing = self.missing_dimensions[1] == 0
#
#         if verbose:
#             msg1 = 'Embedding dimension\t:  {}\nTrajectory dimensions\t: {}'
#             msg2 = 'Complete dimension\t: {}\nMissing dimension     \t: {}'
#             msg1 = msg1.format(self.embedding_dimension, self.trajectory_dimentions)
#             msg2 = msg2.format(self.complete_dimensions, self.missing_dimensions)
#             self._printer('EMBEDDING SUMMARY', msg1, msg2)
#
#         if return_df:
#             return self.X_df
#
#     def decompose(self, verbose=False):
#         '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace
#         Characteristic of projection: the proportion of variance captured in the subspace'''
#         X = self.X_com
#         self.S = X * X.T
#         self.U, self.s, self.V = linalg.svd(self.S)
#         self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
#         self.d = np.linalg.matrix_rank(X)
#         Vs, Xs, Ys, Zs = {}, {}, {}, {}
#         for i in range(self.d):
#             Zs[i] = self.s[i] * self.V[:, i]
#             Vs[i] = X.T * (self.U[:, i] / self.s[i])
#             Ys[i] = self.s[i] * self.U[:, i]
#             Xs[i] = Ys[i] * (m(Vs[i]).T)
#         self.Vs, self.Xs = Vs, Xs
#         self.s_contributions = self.get_contributions(X, self.s, False)
#         self.r = len(self.s_contributions[self.s_contributions > 0])
#         self.r_characteristic = round((self.s[:self.r] ** 2).sum() / (self.s ** 2).sum(), 4)
#         self.orthonormal_base = {i: self.U[:, i] for i in range(self.r)}
#
#         if verbose:
#             msg1 = 'Rank of trajectory\t\t: {}\nDimension of projection space\t: {}'
#             msg1 = msg1.format(self.d, self.r)
#             msg2 = 'Characteristic of projection\t: {}'.format(self.r_characteristic)
#             self._printer('DECOMPOSITION SUMMARY', msg1, msg2)
#
#     def view_s_contributions(self, adjust_scale=False, cumulative=False, return_df=False):
#         '''View the contribution to variance of each singular value and its corresponding signal'''
#         contribs = self.s_contributions.copy()
#         contribs = contribs[contribs.Contribution != 0]
#         if cumulative:
#             contribs['Contribution'] = contribs.Contribution.cumsum()
#         if adjust_scale:
#             contribs = (1 / contribs).max() * 1.1 - (1 / contribs)
#         ax = contribs.plot.bar(legend=False)
#         ax.set_xlabel("Singular_i")
#         ax.set_title('Non-zero{} contribution of Singular_i {}'. \
#                      format(' cumulative' if cumulative else '', '(scaled)' if adjust_scale else ''))
#         if adjust_scale:
#             ax.axes.get_yaxis().set_visible(False)
#         vals = ax.get_yticks()
#         ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])
#         if return_df:
#             return contribs
#
#     @classmethod
#     def view_reconstruction(cls, *hankel, names=None, return_df=False, plot=True, symmetric_plots=False):
#         '''Visualise the reconstruction of the hankel matrix/matrices passed to *hankel'''
#         hankel_mat = None
#         for han in hankel:
#             if isinstance(hankel_mat, m):
#                 hankel_mat = hankel_mat + han
#             else:
#                 hankel_mat = han.copy()
#         hankel_full = cls.diagonal_averaging(hankel_mat)
#         title = 'Reconstruction of signal'
#         if names or names == 0:
#             title += ' associated with singular value{}: {}'
#             title = title.format('' if len(str(names)) == 1 else 's', names)
#         if plot:
#             ax = hankel_full.plot(legend=False, title=title)
#             if symmetric_plots:
#                 velocity = hankel_full.abs().max()[0]
#                 ax.set_ylim(bottom=-velocity, top=velocity)
#         if return_df:
#             return hankel_full
#
#     def _forecast_prep(self, singular_values=None):
#         self.X_com_hat = np.zeros(self.complete_dimensions)
#         self.verticality_coefficient = 0
#         self.forecast_orthonormal_base = {}
#         if singular_values:
#             try:
#                 for i in singular_values:
#                     self.forecast_orthonormal_base[i] = self.orthonormal_base[i]
#             except:
#                 if singular_values == 0:
#                     self.forecast_orthonormal_base[0] = self.orthonormal_base[0]
#                 else:
#                     raise ('Please pass in a list/array of singular value indices to use for forecast')
#         else:
#             self.forecast_orthonormal_base = self.orthonormal_base
#         self.R = np.zeros(self.forecast_orthonormal_base[0].shape)[:-1]
#         for Pi in self.forecast_orthonormal_base.values():
#             self.X_com_hat += Pi * Pi.T * self.X_com
#             pi = np.ravel(Pi)[-1]
#             self.verticality_coefficient += pi ** 2
#             self.R += pi * Pi[:-1]
#         self.R = m(self.R / (1 - self.verticality_coefficient))
#         self.X_com_tilde = self.diagonal_averaging(self.X_com_hat)
#
#     def forecast_recurrent(self, steps_ahead=12, singular_values=None, plot=False, return_df=False, **plotargs):
#         '''Forecast from last point of original time series up to steps_ahead using recurrent methodology
#         This method also fills any missing data from the original time series.'''
#         try:
#             self.X_com_hat
#         except(AttributeError):
#             self._forecast_prep(singular_values)
#         self.ts_forecast = np.array(self.ts_v[0])
#         for i in range(1, self.ts_N + steps_ahead):
#             try:
#                 if np.isnan(self.ts_v[i]):
#                     x = self.R.T * m(self.ts_forecast[max(0, i - self.R.shape[0]): i]).T
#                     self.ts_forecast = np.append(self.ts_forecast, x[0])
#                 else:
#                     self.ts_forecast = np.append(self.ts_forecast, self.ts_v[i])
#             except(IndexError):
#                 x = self.R.T * m(self.ts_forecast[i - self.R.shape[0]: i]).T
#                 self.ts_forecast = np.append(self.ts_forecast, x[0])
#         self.forecast_N = i + 1
#         new_index = pd.date_range(start=self.ts.index.min(), periods=self.forecast_N, freq=self.freq)
#         forecast_df = df(self.ts_forecast, columns=['Forecast'], index=new_index)
#         forecast_df['Original'] = np.append(self.ts_v, [np.nan] * steps_ahead)
#         if plot:
#             forecast_df.plot(title='Forecasted vs. original time series', **plotargs)
#         if return_df:
#             return forecast_df
#
#
# #if __name__ == '__main__':
# from pandas import DataFrame as df
# import pandas as pd
# import numpy as np
# from matplotlib.pylab import rcParams
#
# # Construct the data with gaps
# ts = pd.read_csv('data/air_ssa.csv', parse_dates=True, index_col='Month')
# ts_ = ts.copy()
# ts_.ix[67:79] = np.nan
# ts_ = ts_.set_value('1961-12-01', '#Passengers', np.nan).asfreq('MS')
# ssa = mySSA(ts_)
#
# # Plot original series for reference
# ssa.view_time_series()
# plt.show(block=False)
#
# ssa.embed(embedding_dimension=3, verbose=True)
# ssa.decompose(True)
# ssa.view_s_contributions(adjust_scale=True)
# plt.show(block=False)
#
# # Component Signals
# components = [i for i in range(3)]
# rcParams['figure.figsize'] = 11, 2
# for i in range(3):
#     ssa.view_reconstruction(ssa.Xs[i], names=i, symmetric_plots=i != 0)
#     plt.show(block=False)
# rcParams['figure.figsize'] = 11, 4
#
# # RECONSTRUCTION
# ssa.view_reconstruction(*[ssa.Xs[i] for i in components], names=components)
# plt.show(block=False)
#
# # FORECASTING
# ssa.forecast_recurrent(steps_ahead=48, plot=True)
# plt.show(block=False)