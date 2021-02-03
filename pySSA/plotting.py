from plotly.graph_objs import *
import plotly.offline as offline


def plot(ssa,
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

    train = ssa.mssa.ts_df.copy()
    train_forecast = ssa.forecast.iloc[:-ssa.steps, :].copy()
    test_forecast = ssa.forecast.iloc[-ssa.steps:, :].copy()

    if index is not None:
        index_train = index[:-ssa.steps]
        index_test = index[-ssa.steps:]
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
            conf_int_df = ssa.conf_intervals
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
    if save_html_path is None:
        fig.show()
    else:
        offline.plot(fig, filename=save_html_path)

