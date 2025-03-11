import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def discretize_time_results(matrix, max_time, nx, ny, time_period):
    """
    Discretize time results from a matrix into a tensor.

    Args:
        matrix (numpy array): Matrix of time results
        max_time (int): Maximum time
        nx (int): Number of x coordinates
        ny (int): Number of y coordinates
        time_period (int): Time period to discretize

    Returns:
        numpy array: Discretized time results tensor
    """
    tensor = np.empty((0, nx, ny), int)
    for t in range(0, max_time, time_period):
        filtered_matrix = np.copy(matrix)
        filtered_matrix[matrix > t + time_period] = -1
        tensor = np.concatenate((tensor, filtered_matrix[None]), axis=0)
    return tensor


def discretize_alpha(wind_alphas, wind_speeds):
    """
    Discretize wind speeds based on wind direction angles.

    Args:
        wind_alphas (pd.Series): Series of wind direction angles in radians.
        wind_speeds (pd.Series): Series of corresponding wind speeds.

    Returns:
        list: A list of 8 averaged wind speeds, each corresponding to a
              45-degree sector centered on the standard compass directions.
    """
    sum_speeds = [0] * 8
    count = [0] * 8
    wind_alphas = wind_alphas.to_list()
    wind_speeds = wind_speeds.to_list()

    for i in range(8):
        lower_bound = -np.pi / 8 + (i * np.pi / 4)
        upper_bound = np.pi / 8 + (i * np.pi / 4)
        for alpha, speed in zip(wind_alphas, wind_speeds):
            if lower_bound <= alpha <= upper_bound:
                sum_speeds[i] += speed
                count[i] += 1

    discretized = [sum_speeds[i] / count[i] if count[i] != 0 else 0 for i in range(8)]
    return discretized


def discretize_time_wind(wind_alphas, wind_speeds, max_time):
    """
    Discretize wind alphas and wind speeds into time steps.

    Args:
        wind_alphas (pd.Series): Series of wind direction angles in radians.
        wind_speeds (pd.Series): Series of corresponding wind speeds.
        max_time (int): Maximum time in seconds.

    Returns:
        tuple: Two DataFrames, the first with discretized wind alphas and the
               second with discretized wind speeds.
    """
    time_steps = range(0, max_time, 1)
    discretized_alphas = pd.DataFrame({'x': time_steps, 'y': [0] * len(time_steps)})
    discretized_speeds = pd.DataFrame({'x': time_steps, 'y': [0] * len(time_steps)})

    for t in time_steps:
        filtered_alphas = wind_alphas[(wind_alphas >= t) & (wind_alphas < (t + 1))]
        filtered_speeds = wind_speeds[(wind_speeds >= t) & (wind_speeds < (t + 1))]

        if not filtered_alphas.empty and not filtered_speeds.empty:
            mean_alpha = filtered_alphas.mean()
            mean_speed = filtered_speeds.mean()
            discretized_alphas.loc[discretized_alphas['x'] == t + 1, 'y'] = mean_alpha
            discretized_speeds.loc[discretized_speeds['x'] == t + 1, 'y'] = mean_speed
        else:
            prev_alpha = wind_alphas[wind_alphas < t].iloc[-1]
            prev_speed = wind_speeds[wind_speeds < t].iloc[-1]
            discretized_alphas.loc[discretized_alphas['x'] == t + 1, 'y'] = prev_alpha
            discretized_speeds.loc[discretized_speeds['x'] == t + 1, 'y'] = prev_speed

    return discretized_alphas, discretized_speeds


def plot_results(label, env, t_p=60):
    """
    Visualize the simulation results of a wildfire environment using various plots.

    This function generates an HTML file containing multiple plots to visualize the
    wildfire simulation results, including heatmaps, surface plots, polar bar plots,
    and scatter plots. The visualizations are based on data from the given environment,
    such as fuel maps, wind speeds, wind directions, and terrain altitude.

    Args:
        label (str): The label used for naming the output HTML file.
        env: An instance of the environment class containing simulation data and parameters.
        t_p (int, optional): The time period for discretization and plotting, in seconds. Default is 60.

    Generates:
        An HTML file with the given label containing the plots, which visualize the
        wildfire simulation over time and across the environment grid.
    """
    Nx = env.Nx
    Ny = env.Ny
    results = env.results
    sr = env.sr
    terrain = env.altitude_map
    df_ws = env.df_ws
    df_wa = env.df_wa

    df_wa, df_ws = discretize_time_wind(df_wa, df_ws, env.max_sim_t)

    dic = {"no_fuel": 0, "grass_short": 1, "grass_tall": 2, "brush_not_chaparral": 3, "chaparral": 4
        , "timber_grass_under": 5, "timber_litter": 6, "timber_litter_under": 7,
           "hardwood_litter": 8, "logging_light": 9, "logging_medium": 10, "logging_heavy": 11}

    map_fuel = []
    for i in range(len(env.fuel_map)):
        temp = []
        for j in range(len(env.fuel_map[0])):
            if not env.fuel_map[i][j].critical:
                temp.append(dic[env.fuel_map[i][j].name])
            else:
                temp.append(-11)
        map_fuel.append(temp)

    df_cp = pd.DataFrame([], columns=['x', 'y', 'z'])

    results_stamps = discretize_time_results(results, env.max_sim_t, Nx, Ny, t_p)

    fig = make_subplots(rows=3, cols=4,
                        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'polar'},
                                {'type': 'surface', 'rowspan': 3}],
                               [{'type': 'scatter', 'colspan': 3}, None, None, None],
                               [{'type': 'scatter', 'colspan': 3}, None, None, None]],
                        column_widths=[0.20, 0.20, 0.20, 0.4],
                        row_heights=[0.5, 0.25, 0.25])

    fig_px = px.imshow(results_stamps, x=np.linspace(0, Ny * sr, Ny), y=np.linspace(0, Nx * sr, Nx), animation_frame=0)

    sliders = fig_px.layout.sliders
    update_menus = fig_px.layout.updatemenus

    max_y_ws = df_ws['y'].max()
    min_y_ws = df_ws['y'].min()

    max_y_wa = df_wa['y'].max()
    min_y_wa = df_wa['y'].min()

    frames = [go.Frame(data=[
        go.Heatmap(z=results_stamps[k], visible=True, name=str(k), colorscale='hot', x=np.linspace(0, Ny * sr, Ny),
                   y=np.linspace(0, Nx * sr, Nx), showlegend=False),

        go.Barpolar(r=discretize_alpha(df_wa[(df_wa['x'] >= k * t_p) & (df_wa['x'] < (k + 1) * t_p)]['y'],
                                       df_ws[(df_wa['x'] >= k * t_p) & (df_wa['x'] < (k + 1) * t_p)]['y']),
                    text=['W', 'N-W', 'W', 'N-E', 'E', 'S-E', 'S', 'S-W'], showlegend=False),

        go.Surface(
            colorscale='hot',
            opacity=0.9,
            contours={
                "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
            },
            x=np.linspace(0, Ny * sr, Ny),
            y=np.linspace(0, Nx * sr, Nx),
            z=terrain,
            surfacecolor=results_stamps[k]
        ),

        go.Scatter(x=[k * t_p, k * t_p], y=[max_y_ws, min_y_ws], showlegend=False),

        go.Scatter(x=[k * t_p, k * t_p], y=[max_y_wa, min_y_wa], showlegend=False)
    ],
        traces=[0, 1, 2, 4, 6], name=str(k)) for k in range(results_stamps.shape[0])]  # [0, 1, 2, 4, 6]

    fig.add_trace(
        go.Heatmap(z=results_stamps[0], colorscale='hot', x=np.linspace(0, Ny * sr, Ny), y=np.linspace(0, Nx * sr, Nx)),
        row=1, col=1)

    fig.add_trace(
        go.Barpolar(r=discretize_alpha(df_wa[(df_wa['x'] >= 0) & (df_wa['x'] < t_p)]['y'],
                                       df_ws[(df_wa['x'] >= 0) & (df_wa['x'] < t_p)]['y']),
                    text=['W', 'N-W', 'W', 'N-E', 'E', 'S-E', 'S', 'S-W'], showlegend=False), row=1,
        col=3)

    fig.add_trace(go.Surface(
        colorscale='hot',
        opacity=0.9,
        contours={
            "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
        },
        x=np.linspace(0, Ny * sr, Ny),
        y=np.linspace(0, Nx * sr, Nx),
        z=terrain,
        surfacecolor=results_stamps[0]
    ), row=1, col=4)

    maxN = max(Nx, Ny)

    fig.update_scenes({"aspectratio": {"x": Ny / maxN, "y": Nx / maxN,
                                       "z": (env.altitude_map.max() - env.altitude_map.min()) / env.sr / maxN}})

    fig.add_trace(go.Scatter(x=df_ws['x'], y=df_ws['y'], showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, 0], y=[df_ws['y'].max(), df_ws['y'].min()], showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_wa['x'], y=df_wa['y'], showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=[0, 0], y=[df_wa['y'].max(), df_wa['y'].min()], showlegend=False), row=3, col=1)

    fig.add_trace(go.Scatter3d(x=df_cp['x'],
                               y=df_cp['y'],
                               z=df_cp['z'],
                               mode='markers',
                               marker=dict(size=5, color='blue'),
                               showlegend=False
                               )
                  )

    fig.add_trace(
        go.Heatmap(z=map_fuel, colorscale='RdYlGn', x=np.linspace(0, Ny * sr, Ny), y=np.linspace(0, Nx * sr, Nx),
                   colorbar={'orientation': 'h',
                             'ticktext': ["critical_location", "no_fuel", "grass_short", "grass_tall",
                                          "brush_not_chaparral", "chaparral",
                                          "timber_grass_under",
                                          "timber_litter",
                                          "timber_litter_under",
                                          "hardwood_litter", "logging_light",
                                          "logging_medium", "logging_heavy"],
                             'tickvals': [-11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                             }), row=1, col=2)

    fig.update(frames=frames)
    fig.update_layout(updatemenus=update_menus, sliders=sliders)
    fig.update_layout(
        sliders=[{"currentvalue": {"prefix": "Simulation Time Stamps (TickTime: " + str(t_p) + " s) = "}}])

    fig.show()
    fig.write_html(str(label) + ".html")
