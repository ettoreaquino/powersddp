import plotly.graph_objects as go
import pandas as pd

from plotly.subplots import make_subplots


def sdp(operation: pd.DataFrame):
    
    df = operation.drop(columns=["hydro_units", "thermal_units"], axis=1)
    df_mean = df.groupby(["stage", "initial_volume"]).mean().reset_index().sort_values(by=['stage', 'initial_volume'], ascending=[False, True])
    n_stages = df["stage"].unique().size

    fig = make_subplots(rows=n_stages, cols=1)

    for i, stage in enumerate(df["stage"].unique()):
        stage_df = df.loc[df["stage"] == stage]
        stage_mean = df_mean.loc[df_mean["stage"] == stage]

        fig.add_trace(go.Scatter(x=stage_mean["initial_volume"],
                                 y=stage_mean["total_cost"],
                                 mode="lines",
                                 name="Stage {}".format(stage)),
                      row=i + 1,
                      col=1)
        for j, scenario in enumerate(stage_df["scenario"].unique()[::-1]): 
            scenario_df = stage_df.loc[stage_df["scenario"] == scenario]
            if j == len(df["scenario"].unique()) - 1:
                fig.add_trace(go.Scatter(x=scenario_df["initial_volume"],
                                         y=scenario_df["total_cost"],
                                         mode="lines",
                                         line=dict(width=0),
                                         marker=dict(color="#444"),
                                         name="Scenario {}".format(scenario),
                                         fillcolor='rgba(163, 172, 247, 0.3)',
                                         fill='tonexty',
                                         showlegend=False),
                              row=i + 1,
                              col=1)
            else:
                fig.add_trace(go.Scatter(x=scenario_df["initial_volume"],
                                         y=scenario_df["total_cost"],
                                         mode="lines",
                                         line=dict(width=0),
                                         marker=dict(color="#444"),
                                         name="Scenario {}".format(scenario),
                                         showlegend=False),
                              row=i + 1,
                              col=1)

    fig.update_xaxes(title_text="Final Volume [hm3]")
    fig.update_yaxes(title_text="$/MW")

    fig.update_layout(height=300 * n_stages, title_text="Future Cost Function")
    fig.show()


def ulp(operation: pd.DataFrame, yaxis_column: str, yaxis_title: str, plot_title: str):

    n_gu = operation["name"].unique().size

    fig = make_subplots(rows=n_gu, cols=1)

    for i, gu in enumerate(operation["name"].unique()):
        df = operation.loc[operation["name"] == gu]
        fig.add_trace(
            go.Scatter(
                x=df["stage"],
                y=df[yaxis_column],
                mode="lines",
                name="{}".format(gu),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_xaxes(title_text="Stages")
    fig.update_yaxes(title_text=yaxis_title)

    fig.update_layout(height=300 * n_gu, title_text=plot_title)
    fig.show()
