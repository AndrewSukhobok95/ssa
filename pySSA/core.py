import pandas as pd
import numpy as np
from scipy import linalg
from scipy.stats import norm


class MSSA(object):
    """
    Multi-channel Singular Spectrum Analysis object
    SSA class take one positional argument – timeseries.
    :param timeseries: type can be pandas.DataFrame, pandas.Series, numpy.array, numpy.matrix, list
    """
    def __init__(self, time_series):
        self.ts_df = pd.DataFrame(time_series).reset_index(drop=True)
        self.ts = np.matrix(self.ts_df)
        self.ts_N = self.ts.shape[0]
        self.ts_s = self.ts.shape[1]
        self.ts_name = self.ts_df.columns.tolist()

    def _hankelSeries(self, series):
        """
        Perform hankelization procedure for given series
        """
        return linalg.hankel(series, np.zeros(self.embedding_dimension)).T[:, :self.K]

    def embed(self, embedding_dimension=None):
        """
        Compute the trajectory matrix of given time series.
        :param embedding_dimension: How many components to compute from the original series.
                                    Default is N//2, where N stands for length of series.
        """
        if not embedding_dimension:
            self.embedding_dimension = self.ts_N // 2
        else:
            self.embedding_dimension = embedding_dimension
        self.K = self.ts_N - self.embedding_dimension + 1
        series = np.hsplit(self.ts, self.ts_s)
        X = np.hstack(list(map(self._hankelSeries, series)))
        self.X = np.matrix(X)
        self.trajectory_dimentions = X.shape

    def decompose(self):
        """
        Perform the Singular Value Decomposition and identify the rank of the embedding subspace.
        """
        X = self.X
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = np.matrix(self.U), np.sqrt(self.s), np.matrix(self.V)
        self.d = np.linalg.matrix_rank(X)
        self.Vs = (X.T * self.U) / self.s
        U_s = np.matrix(np.array(self.U) * self.s)
        Xs = np.empty((self.embedding_dimension, 0))
        for i in range(self.d):
            Xs = np.hstack((Xs, U_s[:, i] * self.Vs[:, i].T))
        self.Xs = Xs

    @staticmethod
    def _diagonal_averaging(hankel_matrix):
        """
        Performs anti-diagonal averaging from given hankel matrix
        :param embedding_dimension: Trajectory matrix of one-dimensional time series.
        :return: pandas.DataFrame with decomposed series.
        """
        mat = np.matrix(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L, K), max(L, K)
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
        self.resids = np.matrix(resids)
        ###############################

        self.C_grouped = np.matrix(res)
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
        S = np.matrix([]).reshape((0, r))
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
        out_col_names = []
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

            out_col_names += ["conf. 5%", "conf. 95%"]
        conf_int_df = pd.DataFrame(intervals).T
        conf_int_df.columns = out_col_names
        return conf_int_df

