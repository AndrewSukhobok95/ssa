from ssa_py.ssa import *

#####################################################

ts = pd.read_csv('data/art_series.csv', index_col=0)
ssa = SSA(ts)
decomp = ssa.decompose(20, return_df=True)
# decomp.to_csv('decomp.csv')




#####################################################

ssa = SSA(ts)
ssa.decompose(20)
forecast = ssa.RLforecast(5, 40)
forecast




#####################################################

ssa = SSA(ts)
ssa.decompose(20)
forecast = ssa.RLforecast(8, 40)
ssa.plot(plot_series=[0,1,2], title='Plot Name', x_ax='Day', y_ax='Value', output_folder='data/', file_name='plot')

ssa.plot(plot_series=[1], title='Plot Name', x_ax='Day', y_ax='Value', output_folder='data/', file_name='plot')


#####################################################

train = ts.iloc[:-40, :].copy()
test = ts.iloc[-40:, :].copy()

ssa = SSA(train)
ssa.decompose(20)
forecast = ssa.RLforecast(5, 40)
ssa.plot(plot_series=[0], title='Plot Name', x_ax='Day', y_ax='Value', output_folder='data/', file_name='plot', test=test)




#####################################################

ts = pd.read_csv('data/art_series.csv', index_col=0)

train = ts.iloc[:-40, :].copy()
test = ts.iloc[-40:, :].copy()

from datetime import timedelta
day = [pd.to_datetime('2017-01-01')+timedelta(days=i) for i in range(200)]

ssa = SSA(train)
ssa.decompose(20)
forecast = ssa.RLforecast(5, 100)
ssa.plot(plot_series=[0], title='Plot Name', x_ax='Day', y_ax='Value', output_folder='data/', file_name='plot')

#####################################################


