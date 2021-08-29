import plotly.graph_objects as go
import pandas as pd

from plotly.subplots import make_subplots


def sdp(operation: pd.DataFrame):

    n_stages = operation["stage"].unique().size

    fig = make_subplots(rows=n_stages, cols=1)

    for i, stage in enumerate(operation["stage"].unique()):
        stage_df = operation.loc[operation["stage"] == stage]
        fig.add_trace(
            go.Scatter(
                x=stage_df["initial_volume"],
                y=stage_df["average_cost"],
                mode="lines",
                name="Stage {}".format(stage),
            ),
            row=i + 1,
            col=1,
        )

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
