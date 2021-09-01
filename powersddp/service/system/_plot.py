import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def sdp(operation: pd.DataFrame):

    df = operation.drop(columns=["hydro_units", "thermal_units"], axis=1)
    df_mean = (
        df.groupby(["stage", "initial_volume"])
        .mean()
        .reset_index()
        .sort_values(by=["stage", "initial_volume"], ascending=[False, True])
    )
    n_stages = df["stage"].unique().size

    fig = make_subplots(rows=n_stages, cols=1)

    for i, stage in enumerate(df["stage"].unique()):
        stage_df = df.loc[df["stage"] == stage]
        stage_mean = df_mean.loc[df_mean["stage"] == stage]

        fig.add_trace(
            go.Scatter(
                x=stage_mean["initial_volume"],
                y=stage_mean["total_cost"],
                mode="lines",
                name="Stage {}".format(stage),
            ),
            row=i + 1,
            col=1,
        )
        for j, scenario in enumerate(stage_df["scenario"].unique()[::-1]):
            scenario_df = stage_df.loc[stage_df["scenario"] == scenario]
            if j == len(df["scenario"].unique()) - 1:
                fig.add_trace(
                    go.Scatter(
                        x=scenario_df["initial_volume"],
                        y=scenario_df["total_cost"],
                        mode="lines",
                        line=dict(width=0),
                        marker=dict(color="#444"),
                        name="Scenario {}".format(scenario),
                        fillcolor="rgba(163, 172, 247, 0.3)",
                        fill="tonexty",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=scenario_df["initial_volume"],
                        y=scenario_df["total_cost"],
                        mode="lines",
                        line=dict(width=0),
                        marker=dict(color="#444"),
                        name="Scenario {}".format(scenario),
                        showlegend=False,
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


def sdp_2hgu(operation: pd.DataFrame):
    # Adjust dataset for surface plot
    hgus = operation.iloc[0]["hydro_units"]["name"].unique().tolist()
    df = operation.drop(
        [
            "hydro_units",
            "thermal_units",
            "initial_volume",
            "scenario",
            "future_cost",
            "operational_marginal_cost",
            "shortage",
        ],
        axis=1,
    )
    mean_df = df.groupby(["stage", "discretization"]).mean().reset_index()
    mean_df = mean_df.sort_values(
        by=["stage", "discretization"], ascending=[False, True]
    ).reset_index(drop=True)

    # Get stages and discretizations
    stages = mean_df["stage"].unique()
    discretizations = mean_df["discretization"].unique()

    # Creating axis meshgrids
    step = 100 / (len(list(set([disc[0] for disc in discretizations]))) - 1)
    xaxis, yaxis = np.meshgrid(
        np.arange(0, 100 + step, step), np.arange(0, 100 + step, step)
    )

    # Building costs mesh grids
    costs = []
    for i, stage in enumerate(stages):
        stage_df = mean_df.loc[mean_df["stage"] == stage]
        zaxis = np.array(stage_df["total_cost"].to_list()).reshape(3, 3).T
        costs.append(
            {
                "stage": stage,
                "zaxis": zaxis,
            }
        )

    costs = pd.DataFrame(costs)

    # Plotting
    n_stages = costs["stage"].unique().size # type: ignore

    fig = make_subplots(
        rows=n_stages,
        cols=1,
        specs=[[{"type": "surface"}]] * n_stages,
        subplot_titles=["Stage {}".format(stage + 1) for stage in range(n_stages)],
    )

    for i, stage in enumerate(costs["stage"].unique()): # type: ignore
        stage_df = costs.loc[costs["stage"] == stage] # type: ignore
        fig.add_trace(
            go.Surface(
                x=xaxis,
                y=yaxis,
                z=stage_df["zaxis"].values[0],
                showscale=False,
                colorscale="Viridis",
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="{} Initial Volume [hm3]".format(hgus[0]),
            yaxis_title="{} Initial Volume [hm3]".format(hgus[1]),
            zaxis_title="$/MW",
        ),
        scene2=dict(
            xaxis_title="{} Initial Volume [hm3]".format(hgus[0]),
            yaxis_title="{} Initial Volume [hm3]".format(hgus[1]),
            zaxis_title="$/MW",
        ),
        scene3=dict(
            xaxis_title="{} Initial Volume [hm3]".format(hgus[0]),
            yaxis_title="{} Initial Volume [hm3]".format(hgus[1]),
            zaxis_title="$/MW",
        ),
        height=900 * n_stages,
        width=1500,
        title_text="Future Cost Function",
        autosize=False,
    )
    fig.show()
