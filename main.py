import pandas as pd
import matplotlib.pyplot as plt

from pySSA.ssa_numpy import SSA, MSSA

DATA_DIR = "./data/"
IMG_DIR = "./imgs/"
DATASET_AIR = DATA_DIR + 'air_ssa.csv'
DATASET_ART = DATA_DIR + 'art_series.csv'

########################### SSA ###########################

ts = pd.read_csv(DATASET_AIR, parse_dates=True, index_col=0)
ssa = SSA(ts)
ssa.decompose(20)
ssa.RLforecast(7, steps=50)
ssa.plot(plot_series=[0],
         title='Plot Name',
         x_ax='Day',
         y_ax='Value',
         output_folder=IMG_DIR,
         file_name='plot1')

ts = pd.read_csv(DATASET_ART, parse_dates=True, index_col=0)
ssa = SSA(ts)
ssa.decompose(20)
ssa.RLforecast(7, steps=50)
ssa.plot(plot_series=[0, 1, 2],
         title='Plot Name',
         x_ax='Day',
         y_ax='Value',
         output_folder=IMG_DIR,
         file_name='plot2')

########################### MSSA ###########################

ts = pd.read_csv(DATASET_AIR, parse_dates=True, index_col=0)
# ts = pd.read_csv(DATASET_ART, parse_dates=True, index_col=0)

ssa = MSSA(ts)
## Decomposition
ssa.embed(embedding_dimension=2)
ssa.decompose()
b = ssa.diag_procedure()

## Forecasting Reccurent L
ssa.embed(embedding_dimension=20)
ssa.decompose()
ssa.group_components(5)
forc = ssa.L_reccurent_forecast(40)
inter = ssa.conf_int()
plt.figure()
forc.plot()
plt.savefig(IMG_DIR + "fig_L_reccurent_forecast.png")

###Forecasting Reccurent K
ssa.embed(embedding_dimension=10)
ssa.decompose()
ssa.group_components(5)
forc = ssa.K_reccurent_forecast(40)
plt.figure()
forc.plot()
plt.savefig(IMG_DIR + "fig_K_reccurent_forecast.png")