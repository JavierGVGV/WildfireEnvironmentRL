import random

import numpy as np
import pandas as pd
import plots
from wildfire_environment_dt import WildfireEnvironment

"""
Execution example for the WildfireEnvironment class.

This script shows an example of how to use the WildfireEnvironment class to
simulate a wildfire. The simulation is done in a stochastic way, meaning that
the wind direction and velocity are randomly chosen at each time step.

The simulation is done in a loop, where at each iteration, the wind direction
and velocity are randomly chosen, and the `stochastic_step` method is called.
The simulation stops when the `done` attribute of the environment is set to
True.

The resulting simulation is then plotted using the `plot_results` function
from the `plots` module.
"""

random.seed(1)
Nx = 51
Ny = 51
sr = 50
max_sim_t = 7200  # 2 h of maximum simulation time
env = WildfireEnvironment(Nx, Ny, sr, max_sim_t)
env.reset([[25, 25]], new_scenario=True, new_init_points=True)

env.wind_dirs = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
env.wind_velocity_10 = np.random.random() * 3

while True:
    if not env.done:
        # Update wind direction and velocity
        alpha_conv_table = np.array([np.pi / 2, np.pi / 2 + np.pi / 4, np.pi, np.pi + np.pi / 4, 3 * np.pi / 2,
                                     3 * np.pi / 2 + np.pi / 4, 0, np.pi / 4])
        env.df_wa = pd.concat(
            [env.df_wa,
             pd.DataFrame({'x': [env.current_t], 'y': [sum(env.wind_dirs * alpha_conv_table) % (np.pi * 2)]})])
        env.df_ws = pd.concat(
            [env.df_ws, pd.DataFrame({'x': [env.current_t], 'y': [env.wind_velocity_10]})])

    # Take a step in the environment
    env.stochastic_step()

    # Check if the simulation is done
    if env.done:
        # Plot the results
        plots.plot_results("WildfireSimulation", env)
        break
