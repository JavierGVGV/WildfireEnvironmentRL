import math
import uuid
import numpy as np
import pandas as pd
from noise import snoise2
import torch
import torch.nn.functional as F
from collections import deque


class FuelType:
    """
    A class to represent the type of fuel.

    Attributes:
        name (str): Name of the fuel type.
        fuel_cat (int): Fuel category.
        moisture_content (float): Moisture content of the fuel.
        sigma (float): Surface area-to-volume ratio.
        w0 (float): Ovendry fuel load.
        fuel_depth (float): Depth of the fuel.
        ovendry_particle_density (float): Ovendry particle density.
        low_heat_content (float): Low heat content.
        total_mineral_content (float): Total mineral content.
        dead_fuel_moisture_extinction (float): Dead fuel moisture extinction.
        effective_mineral_content (float): Effective mineral content.
        critical (bool): Indicates if the fuel is critical.
        nofuel (bool): Indicates if there is no fuel.
    """

    def __init__(self, name, fuel_cat, moisture_content, sigma, w0, fuel_depth, ovendry_particle_density,
                 low_heat_content, total_mineral_content, dead_fuel_moisture_extinction, effective_mineral_content,
                 critical, nofuel):
        self.name = name
        self.fuel_cat = fuel_cat
        self.moisture_content = moisture_content
        self.sigma = sigma
        self.w0 = w0
        self.fuel_depth = fuel_depth
        self.ovendry_particle_density = ovendry_particle_density
        self.low_heat_content = low_heat_content
        self.total_mineral_content = total_mineral_content
        self.dead_fuel_moisture_extinction = dead_fuel_moisture_extinction
        self.effective_mineral_content = effective_mineral_content
        self.critical = critical
        self.nofuel = nofuel


class Cell:
    """
    A class to represent a cell in the environment.

    Attributes:
        i (int): Row index of the cell.
        j (int): Column index of the cell.
        sr (float): Spatial resolution in meters.
        ID (str): Unique identifier for the cell.
        frt (float): Fire reach time.
        parent_cell (Cell): Parent cell.
        fuel_type (FuelType): Type of fuel in the cell.
        altitude (float): Altitude of the cell.
        ovendry_bulk_density (float): Ovendry bulk density.
        effective_heating_num (float): Effective heating number.
        preignition_heat (float): Preignition heat.
        packing_ratio (float): Packing ratio.
        optimum_packing_ratio (float): Optimum packing ratio.
        C (float): Coefficient C.
        B (float): Coefficient B.
        E (float): Coefficient E.
        reaction_intensity (float): Reaction intensity.
        propagation_flux_ratio (float): Propagation flux ratio.
        wind_limit_velocity_midflame (float): Wind limit velocity at midflame.
        hf (float): Heat flux.
        WAF (float): Wind adjustment factor.
    """
    orientation_neighbour_cell = {
        str([0, 1]): [1, 0, 0, 0, 0, 0, 0, 0],
        str([-1, 1]): [0, 1, 0, 0, 0, 0, 0, 0],
        str([-1, 0]): [0, 0, 1, 0, 0, 0, 0, 0],
        str([-1, -1]): [0, 0, 0, 1, 0, 0, 0, 0],
        str([0, -1]): [0, 0, 0, 0, 1, 0, 0, 0],
        str([1, -1]): [0, 0, 0, 0, 0, 1, 0, 0],
        str([1, 0]): [0, 0, 0, 0, 0, 0, 1, 0],
        str([1, 1]): [0, 0, 0, 0, 0, 0, 0, 1]
    }

    def __init__(self, i, j, sr, frt, environment):
        """
        Initialize a Cell object.

        Args:
            i (int): Row index of the cell.
            j (int): Column index of the cell.
            sr (float): Spatial resolution in meters.
            frt (float): Fire reach time.
            environment (Environment): The environment object.
        """
        self.i = int(i)
        self.j = int(j)
        self.sr = sr
        self.ID = str(uuid.uuid1())
        self.frt = frt
        self.parent_cell = None
        self.fuel_type = environment.fuel_map[i][j]
        self.altitude = environment.altitude_map[i][j]
        self.ovendry_bulk_density = self.fuel_type.w0 / self.fuel_type.fuel_depth
        self.effective_heating_num = math.exp(-138 / self.fuel_type.sigma)
        self.preignition_heat = 250 + 1116 * self.fuel_type.moisture_content
        self.packing_ratio = self.ovendry_bulk_density / self.fuel_type.ovendry_particle_density
        self.optimum_packing_ratio = 3.348 * self.fuel_type.sigma ** (-0.8189)
        self.C = 7.47 * math.exp(-0.133 * self.fuel_type.sigma ** 0.55)
        self.B = 0.02526 * self.fuel_type.sigma ** 0.54
        self.E = 0.715 * math.exp(-0.000359 * self.fuel_type.sigma)
        moisture_damping_coef = (
                (1 - 2.59 * (self.fuel_type.moisture_content / self.fuel_type.dead_fuel_moisture_extinction)
                 + 5.11 * (self.fuel_type.moisture_content / self.fuel_type.dead_fuel_moisture_extinction) ** 2)
                - 3.52 * (self.fuel_type.moisture_content / self.fuel_type.dead_fuel_moisture_extinction) ** 3)
        mineral_damping_coef = 0.174 * self.fuel_type.effective_mineral_content ** (-0.19)
        maximum_reaction_velocity = self.fuel_type.sigma ** 1.5 * (495 + 0.0594 * self.fuel_type.sigma ** 1.5) ** (-1)
        A = 1 / (4.774 * self.fuel_type.sigma ** 0.1 - 7.27)
        optimum_reaction_velocity = (maximum_reaction_velocity * (
                self.packing_ratio / self.optimum_packing_ratio) ** A) * math.exp(
            A * (1 - self.packing_ratio / self.optimum_packing_ratio))
        net_fuel_loading = self.fuel_type.w0 / (1 + self.fuel_type.total_mineral_content)
        self.reaction_intensity = optimum_reaction_velocity * net_fuel_loading * self.fuel_type.low_heat_content * moisture_damping_coef * mineral_damping_coef
        self.propagation_flux_ratio = ((192 + 0.2595 * self.fuel_type.sigma) ** (-1)) * math.exp((
                                                                                                         0.792 + 0.681 * self.fuel_type.sigma ** 0.5) * (
                                                                                                         self.packing_ratio + 0.1))
        self.wind_limit_velocity_midflame = 96.8 * self.reaction_intensity ** (1 / 3)
        if not self.fuel_type.critical:
            self.hf = np.random.uniform(0.08, 0.30)
        else:
            self.hf = np.random.uniform(0.3, 0.4)
        H = self.fuel_type.fuel_depth
        self.WAF = 1.83 / (np.log((20 + 0.36 * H) / (0.13 * H)))

    def __eq__(self, other_cell):
        """
        Check if two cells are equal based on their ID.

        Args:
            other_cell (Cell): The other cell to compare with.

        Returns:
            bool: True if the cells are equal, False otherwise.
        """
        if other_cell is None:
            return False
        return self.ID == other_cell.ID

    def __gt__(self, other):
        """
        Compare two cells based on their fire reach time.

        Args:
            other (Cell): The other cell to compare with.

        Returns:
            bool: True if the current cell's fire reach time is greater, False otherwise.
        """
        return self.frt > other.frt

    def __lt__(self, other):
        """
        Compare two cells based on their fire reach time.

        Args:
            other (Cell): The other cell to compare with.

        Returns:
            bool: True if the current cell's fire reach time is less, False otherwise.
        """
        return self.frt < other.frt

    def get_neighbours(self, environment):
        """
        Get the neighboring cells of the current cell.

        Args:
            environment (Environment): The environment object.

        Returns:
            list: List of neighboring cells.
        """
        neighbour_cells = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni = self.i + di
                nj = self.j + dj
                if (ni >= 0) and (ni < environment.Nx) and (nj >= 0) and (nj < environment.Ny) and not (
                        (ni == self.i) and (nj == self.j)):
                    if environment.burned_map[ni][nj] != 1:
                        neighbour_cell = Cell(ni, nj, self.sr, 0, environment)
                        frt = self.calculate_frt(neighbour_cell, environment.wind_dirs, environment.wind_velocity_10)
                        neighbour_cell.frt = frt
                        neighbour_cell.parent_cell = self
                        neighbour_cells.append(neighbour_cell)
        return neighbour_cells

    def calculate_frt(self, neighbour_cell, wind_dirs, wind_velocity_10):
        """
        Calculate the fire reach time for a neighboring cell.

        Args:
            neighbour_cell (Cell): The neighboring cell.
            wind_dirs (list): List of wind directions.
            wind_velocity_10 (float): Wind velocity at 10 meters.

        Returns:
            float: Fire reach time for the neighboring cell.
        """
        adjusted_wind_velocity = max(wind_velocity_10 * (self.altitude / 10) ** self.hf, wind_velocity_10)
        d = np.array([(neighbour_cell.j - self.j) * self.sr, (neighbour_cell.i - self.i) * self.sr])
        distance = (d[0] ** 2 + d[1] ** 2) ** (1 / 2)
        slope = ((neighbour_cell.altitude - self.altitude) / distance) * 100
        wind_velocity_midflame = (adjusted_wind_velocity * 196.85) * self.WAF
        orientation = str([(neighbour_cell.j - self.j), (neighbour_cell.i - self.i)])

        if self.orientation_neighbour_cell[orientation] == list(wind_dirs):
            spread_rate = self.calculate_spread_rate(slope, wind_velocity_midflame) * 2.236936
        elif ((wind_dirs[(self.orientation_neighbour_cell[orientation].index(1) + 1) % 8] == 1) or
              (wind_dirs[(self.orientation_neighbour_cell[orientation].index(1) - 1) % 8] == 1)):
            m = 1
            V_head = self.calculate_spread_rate(slope, wind_velocity_midflame) * 2.236936
            V_back = self.calculate_spread_rate(-slope, 0) * 2.236936
            b = (V_head + V_back) / 2
            c = V_head - b
            LW_ratio = min(
                0.936 * math.exp(0.2566 * (wind_velocity_midflame * 0.0113636)) + 0.461 * math.exp(
                    -0.1548 * (wind_velocity_midflame * 0.0113636)) - 0.397, 8)
            a = b / LW_ratio
            b_e = ((2 * m * c) / (b ** 2))
            a_e = ((1 / (a ** 2)) + ((m ** 2) / (b ** 2)))
            c_e = ((c ** 2) / (b ** 2)) - 1
            x1 = (-b_e + (b_e ** 2 - (4 * a_e * c_e)) ** (1 / 2)) / (2 * a_e)
            x2 = (-b_e - (b_e ** 2 - (4 * a_e * c_e)) ** (1 / 2)) / (2 * a_e)
            if x2 != 0:
                y = m * x2 + c
                x = x2
            else:
                y = m * x1 + c
                x = x1
            spread_rate = (((x - 0) ** 2) + ((y - c) ** 2)) ** (1 / 2)
        elif ((wind_dirs[(self.orientation_neighbour_cell[orientation].index(1) + 2) % 8] == 1) or
              (wind_dirs[(self.orientation_neighbour_cell[orientation].index(1) - 2) % 8] == 1)):
            m = 0
            V_head = self.calculate_spread_rate(slope, wind_velocity_midflame) * 2.236936
            V_back = self.calculate_spread_rate(-slope, 0) * 2.236936
            b = (V_head + V_back) / 2
            c = V_head - b
            LW_ratio = min(
                0.936 * math.exp(0.2566 * (wind_velocity_midflame * 0.0113636)) + 0.461 * math.exp(
                    -0.1548 * (wind_velocity_midflame * 0.0113636)) - 0.397, 8)
            a = b / LW_ratio
            b_e = ((2 * m * c) / (b ** 2))
            a_e = ((1 / (a ** 2)) + ((m ** 2) / (b ** 2)))
            c_e = ((c ** 2) / (b ** 2)) - 1
            x1 = (-b_e + (b_e ** 2 - (4 * a_e * c_e)) ** (1 / 2)) / (2 * a_e)
            x2 = (-b_e - (b_e ** 2 - (4 * a_e * c_e)) ** (1 / 2)) / (2 * a_e)
            if x2 != 0:
                y = m * x2 + c
                x = x2
            else:
                y = m * x1 + c
                x = x1
            spread_rate = (((x - 0) ** 2) + ((y - c) ** 2)) ** (1 / 2)
        else:
            spread_rate = self.calculate_spread_rate(slope, 0) * 2.236936

        spread_rate = spread_rate * 0.44704
        return self.frt + (distance / spread_rate)

    def calculate_spread_rate(self, slope, wind_velocity_midflame):
        """
        Calculate the spread rate of fire.

        Args:
            slope (float): Slope percentage.
            wind_velocity_midflame (float): Wind velocity at midflame in ft/min.

        Returns:
            float: Spread rate in m/s.
        """
        if wind_velocity_midflame > self.wind_limit_velocity_midflame:
            wind_velocity_midflame = self.wind_limit_velocity_midflame
        slope_factor = 5.275 * self.packing_ratio ** (-0.3) * (max(slope, 0) / 100) ** 2
        wind_coef = self.C * (max(float(wind_velocity_midflame), 0) ** self.B) * (
                self.packing_ratio / self.optimum_packing_ratio) ** (-self.E)
        R = ((self.reaction_intensity * self.propagation_flux_ratio * (1 + wind_coef + slope_factor)) /
             (self.ovendry_bulk_density * self.effective_heating_num * self.preignition_heat))
        return R / 196.85


class FRTSortedQueueCells:
    """
    A class to represent a sorted queue of cells based on their fire reach time (FRT).

    Attributes:
        cells (list): List of cells in the queue.
        env (Environment): The environment object.
    """

    def __init__(self, env):
        """
        Initialize an FRTSortedQueueCells object.

        Args:
            env (Environment): The environment object.
        """
        self.cells = []
        self.env = env

    def __len__(self):
        """
        Get the number of cells in the queue.

        Returns:
            int: Number of cells in the queue.
        """
        return len(self.cells)

    def add_cell(self, cell):
        """
        Add a cell to the queue in sorted order based on its fire reach time (FRT).

        Args:
            cell (Cell): The cell to be added.
        """
        index = np.searchsorted(self.cells, cell)
        self.cells.insert(index, cell)

    def clean_burned(self):
        """
        Remove burned cells from the queue.
        """
        self.cells = [c for c in self.cells if self.env.burned_map[c.i][c.j] != 1]

    def pop_cell(self):
        """
        Pop the cell with the smallest fire reach time (FRT) from the queue.

        Returns:
            Cell: The cell with the smallest FRT, or None if the queue is empty.
        """
        if len(self.cells) > 0:
            return self.cells.pop(0)
        else:
            return None


class WildfireEnvironment:
    """
    A class to represent the environment in which the simulation takes place.

    Attributes:
        Nx (int): Number of quadrats per side in the x-direction.
        Ny (int): Number of quadrats per side in the y-direction.
        sr (float): Spatial resolution in square meters.
        max_sim_t (int): Maximum simulation time.
        mode (int): Mode of the environment (0 for RL, 1 for ST).
        processed_cells (int): Number of processed cells.
        init_cells_points (list): Initial points of cells.
        z_scale (float): Scale for altitude normalization.
        score (float): Score of the simulation.
        current_cell (Cell): Current cell being processed.
        max_fuel (float): Maximum fuel.
        results (np.ndarray): Results of the simulation.
        burned_map (np.ndarray): Map indicating burned cells.
        active_cells (FRTSortedQueueCells): Queue of active cells.
        altitude_map (np.ndarray): Map of altitudes.
        fuel_map (list): Map of fuel types.
        wind_dirs (np.ndarray): Array of wind directions.
        wind_velocity_10 (float): Wind velocity at 10 meters.
        done (bool): Indicates if the simulation is done.
        df_wa (pd.DataFrame): DataFrame for wind adjustment.
        df_ws (pd.DataFrame): DataFrame for wind speed.
        critical_locations (list): List of critical locations.
        fuel_cat_map_norm (np.ndarray): Normalized fuel category map.
        automate_state_norm (np.ndarray): Normalized automate state map.
        altitude_map_norm (np.ndarray): Normalized altitude map.
    """

    def __init__(self, Nx=100, Ny=100, sr=10, max_sim_t=3600, mode=0):
        """
        Initialize an Environment object.

        Args:
            Nx (int): Number of quadrats per side in the x-direction.
            Ny (int): Number of quadrats per side in the y-direction.
            sr (float): Spatial resolution in squared meters.
            max_sim_t (int): Maximum simulation time.
            mode (int): Mode of the environment (0 for RL, 1 for ST).
        """
        self.processed_cells = 0
        self.init_cells_points = None
        self.z_scale = None
        self.score = 0
        self.current_cell = None
        self.max_fuel = None
        self.results = None
        self.burned_map = None
        self.active_cells = None
        self.altitude_map = None
        self.fuel_map = None
        self.wind_dirs = None
        self.wind_velocity_10 = None
        self.done = False
        self.Nx = Nx
        self.Ny = Ny
        self.sr = sr
        self.max_sim_t = max_sim_t
        self.current_t = 0
        self.df_wa = None
        self.df_ws = None
        self.mode = mode
        self.critical_locations = []
        self.fuel_cat_map_norm = None
        self.automate_state_norm = None
        self.altitude_map_norm = None

    def reset(self, init_cells_points=None, new_scenario=False, new_init_points=False):
        """
        Reset the environment to its initial state.

        Args:
            init_cells_points (list): Initial points of cells.
            new_scenario (bool): Indicates if a new scenario should be created.
            new_init_points (bool): Indicates if new initial points should be generated.
        """
        self.processed_cells = 0
        self.current_t = 0
        self.wind_velocity_10 = 2
        self.burned_map = np.zeros(shape=(self.Nx, self.Ny))
        self.results = np.full((self.Nx, self.Ny), -1)
        self.done = False
        self.df_wa = pd.DataFrame({'x': [], 'y': []})
        self.df_ws = pd.DataFrame({'x': [], 'y': []})
        self.wind_dirs = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        dirt = np.random.randint(0, 8)
        self.wind_dirs[dirt] = 1

        if new_init_points:
            if init_cells_points is None:
                self.init_cells_points = []
                r = np.random.randint(1, 4)
                i = 0
                while i < r:
                    x = np.random.randint(0, self.Nx)
                    y = np.random.randint(0, self.Ny)
                    if not self.fuel_map[x][y].critical and not self.fuel_map[x][y].nofuel:
                        self.init_cells_points.append([x, y])
                        i += 1
            else:
                self.init_cells_points = init_cells_points

        if new_scenario:
            self.critical_locations = []
            if self.mode == 0:
                self.z_scale = 1000
                self.altitude_map_norm = self.synthetic_altitude_map()
                self.altitude_map = self.altitude_map_norm * self.z_scale
                self.fuel_map = self.synthetic_fuel_map()
                self.fuel_cat_map_norm = np.empty((self.Nx, self.Ny), dtype=float)
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        self.fuel_cat_map_norm[i][j] = self.fuel_map[i][j].fuel_cat
                self.fuel_cat_map_norm = self.fuel_cat_map_norm / np.max(self.fuel_cat_map_norm)
            else:
                self.altitude_map_norm = self.altitude_map / np.max(self.altitude_map)
                self.fuel_cat_map_norm = np.empty((self.Nx, self.Ny), dtype=float)
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        self.fuel_cat_map_norm[i][j] = self.fuel_map[i][j].fuel_cat
                self.fuel_cat_map_norm = self.fuel_cat_map_norm / np.max(self.fuel_cat_map_norm)

            self.automate_state_norm = np.full((self.Nx, self.Ny), 0.333, dtype=float)
            for i in range(self.Nx):
                for j in range(self.Ny):
                    if self.fuel_map[i][j].critical:
                        self.automate_state_norm[i][j] = 1
                    elif self.fuel_map[i][j].nofuel:
                        self.automate_state_norm[i][j] = 0

        self.active_cells = FRTSortedQueueCells(self)
        for i, j in self.init_cells_points:
            init_cell = Cell(i, j, self.sr, 0, self)
            self.active_cells.add_cell(init_cell)

    def synthetic_altitude_map(self):
        """
        Generate a random altitude normalized map.

        Returns:
            np.ndarray: Normalized altitude map.
        """
        terrain = np.zeros((self.Nx, self.Ny))
        scale = np.random.randint(min(self.Nx, self.Ny), 20 * min(self.Nx, self.Ny))
        octaves = 8
        persistence = np.random.uniform(0.4, 0.6)
        lacunarity = 2.0
        for i in range(self.Nx):
            for j in range(self.Ny):
                terrain[i][j] = snoise2(i / scale, j / scale, octaves=octaves, persistence=persistence,
                                        lacunarity=lacunarity, repeatx=1024, repeaty=1024)
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
        terrain = terrain * np.random.random()
        rand_rot = np.random.choice([1, 2, 3, 4])
        if rand_rot == 1:
            terrain = np.flip(terrain, axis=0)
        elif rand_rot == 2:
            terrain = np.flip(terrain, axis=1)
        elif rand_rot == 3:
            terrain = np.flip(terrain, axis=0)
            terrain = np.flip(terrain, axis=1)
        return terrain

    def can_reach_critical(self, synthetic_data, start_x, start_y):
        """
        Check if a critical location can be reached from a starting point.

        Args:
            synthetic_data (np.ndarray): Synthetic data map.
            start_x (int): Starting x-coordinate.
            start_y (int): Starting y-coordinate.

        Returns:
            bool: True if a critical location can be reached, False otherwise.
        """
        Nx, Ny = synthetic_data.shape
        queue = deque([(start_x, start_y)])
        visited = set()
        while queue:
            x, y = queue.popleft()
            if synthetic_data[x, y] == -11:
                return True
            if synthetic_data[x, y] > 0 and (x, y) not in visited:
                visited.add((x, y))
                if 0 <= x - 1 < Nx and 0 <= y < Ny:
                    queue.append((x - 1, y))
                if 0 <= x + 1 < Nx and 0 <= y < Ny:
                    queue.append((x + 1, y))
                if 0 <= x < Nx and 0 <= y - 1 < Ny:
                    queue.append((x, y - 1))
                if 0 <= x < Nx and 0 <= y + 1 < Ny:
                    queue.append((x, y + 1))
                if 0 <= x - 1 < Nx and 0 <= y - 1 < Ny:
                    queue.append((x - 1, y - 1))
                if 0 <= x - 1 < Nx and 0 <= y + 1 < Ny:
                    queue.append((x - 1, y + 1))
                if 0 <= x + 1 < Nx and 0 <= y - 1 < Ny:
                    queue.append((x + 1, y - 1))
                if 0 <= x + 1 < Nx and 0 <= y + 1 < Ny:
                    queue.append((x + 1, y + 1))
        return False

    # Method to create fuel map with fuel categories random between 1 and 13
    def synthetic_fuel_map(self):
        """
        Create a synthetic fuel map with random fuel categories.

        Returns:
            list: Synthetic fuel map.
        """
        total_loadings = {
            "no_fuel": 1, "grass_short": 0.75, "grass_tall": 3, "brush_not_chaparral": 6, "chaparral": 25,
            "timber_grass_under": 4, "timber_litter": 15, "timber_litter_under": 30, "hardwood_litter": 15,
            "logging_light": 40, "logging_medium": 120, "logging_heavy": 200
        }
        sigma_fuels = {
            "no_fuel": 1, "grass_short": 3500, "grass_tall": 1500, "brush_not_chaparral": 2000, "chaparral": 2000,
            "timber_grass_under": 3000, "timber_litter": 2000, "timber_litter_under": 2000, "hardwood_litter": 2500,
            "logging_light": 1500, "logging_medium": 1500, "logging_heavy": 1500
        }
        w0_fuels = {
            "no_fuel": 1, "grass_short": 0.034, "grass_tall": 0.138, "brush_not_chaparral": 0.046, "chaparral": 0.23,
            "timber_grass_under": 0.092, "timber_litter": 0.069, "timber_litter_under": 0.138, "hardwood_litter": 0.134,
            "logging_light": 0.069, "logging_medium": 0.184, "logging_heavy": 0.322
        }
        fuels_depth = {
            "no_fuel": 1, "grass_short": 1, "grass_tall": 2.5, "brush_not_chaparral": 2, "chaparral": 6,
            "timber_grass_under": 1.5, "timber_litter": 0.2, "timber_litter_under": 1, "hardwood_litter": 0.2,
            "logging_light": 1, "logging_medium": 2.3, "logging_heavy": 3
        }
        fuels_categories = {
            0: "no_fuel", 1: "grass_short", 2: "grass_tall", 3: "brush_not_chaparral", 4: "chaparral",
            5: "timber_grass_under", 6: "timber_litter", 7: "timber_litter_under", 8: "hardwood_litter",
            9: "logging_light", 10: "logging_medium", 11: "logging_heavy"
        }

        while True:
            synthetic_data = np.zeros((self.Nx, self.Ny))
            scale = np.random.randint(min(self.Nx, self.Ny), 20 * min(self.Nx, self.Ny))
            octaves = 8
            persistence = np.random.uniform(0.4, 0.6)
            lacunarity = 2.0
            for i in range(self.Nx):
                for j in range(self.Ny):
                    synthetic_data[i][j] = snoise2(i / scale, j / scale, octaves=octaves, persistence=persistence,
                                                   lacunarity=lacunarity, repeatx=1024, repeaty=1024)
            synthetic_data = synthetic_data * np.random.random()
            synthetic_data = (synthetic_data - np.min(synthetic_data))
            synthetic_data = synthetic_data / (np.max(synthetic_data) - np.min(synthetic_data))
            synthetic_data = np.round(synthetic_data * 11)
            synthetic_data = np.array(synthetic_data, dtype=int)
            empty_index = np.random.randint(3, 8)
            apply_empty = np.random.random()
            random_no_fuel_degradation = np.random.random()
            for i in range(synthetic_data.shape[0]):
                for j in range(synthetic_data.shape[1]):
                    if synthetic_data[i][j] == empty_index and apply_empty < 0.5:
                        if np.random.random() < random_no_fuel_degradation:
                            synthetic_data[i][j] = 0
                    elif synthetic_data[i][j] == 0 or synthetic_data[i][j] == 1:
                        if ((np.random.random() < np.random.random())
                                and ((0 <= i <= int(self.Nx * 0.2)) or (
                                        self.Nx - int(self.Nx * 0.2) <= i < int(self.Nx)))
                                and ((0 <= j <= int(self.Ny * 0.2)) or (
                                        self.Ny - int(self.Ny * 0.2) <= j < int(self.Ny)))):
                            synthetic_data[i][j] = -11
                        else:
                            synthetic_data[i][j] = 1
            synthetic_data[np.random.choice([0, self.Nx - 1])][np.random.choice([0, self.Ny - 1])] = -11
            reach_critical = False
            for ip in self.init_cells_points:
                if self.can_reach_critical(synthetic_data, ip[0], ip[1]):
                    reach_critical = True
                    break
            if reach_critical:
                break
            rand_rot = np.random.choice([1, 2, 3, 4])
            if rand_rot == 1:
                synthetic_data = np.flip(synthetic_data, axis=0)
            elif rand_rot == 2:
                synthetic_data = np.flip(synthetic_data, axis=1)
            elif rand_rot == 3:
                synthetic_data = np.flip(synthetic_data, axis=0)
                synthetic_data = np.flip(synthetic_data, axis=1)

        fuel_moisture_map = np.zeros((self.Nx, self.Ny))
        scale = np.random.randint(min(self.Nx, self.Ny), 20 * min(self.Nx, self.Ny))
        octaves = 8
        persistence = np.random.uniform(0.4, 0.6)
        lacunarity = 2.0
        for i in range(self.Nx):
            for j in range(self.Ny):
                fuel_moisture_map[i][j] = snoise2(i / scale, j / scale, octaves=octaves, persistence=persistence,
                                                  lacunarity=lacunarity, repeatx=1024, repeaty=1024)
        fuel_moisture_map = (fuel_moisture_map - np.min(fuel_moisture_map))
        fuel_moisture_map = fuel_moisture_map / (np.max(fuel_moisture_map) - np.min(fuel_moisture_map))
        fuel_moisture_map = fuel_moisture_map * np.random.uniform(0.05, 0.25)
        rand_rot = np.random.choice([1, 2, 3, 4])

        if rand_rot == 1:
            fuel_moisture_map = np.flip(fuel_moisture_map, axis=0)
        elif rand_rot == 2:
            fuel_moisture_map = np.flip(fuel_moisture_map, axis=1)
        elif rand_rot == 3:
            fuel_moisture_map = np.flip(fuel_moisture_map, axis=0)
            fuel_moisture_map = np.flip(fuel_moisture_map, axis=1)

        fuel_map = []
        total_mineral_content = 0.0555
        moisture_content = 0.06
        ovendry_particle_density = 32
        low_heat_content = 8000
        dead_fuel_moisture_extinction = 0.3
        effective_mineral_content = 0.01

        for i in range(0, self.Nx):
            temp = []
            for j in range(0, self.Ny):
                if synthetic_data[i][j] > 0:
                    fuel = fuels_categories[synthetic_data[i][j]]
                    fuel_cat = synthetic_data[i][j]
                    critical = False
                    no_fuel = False
                elif synthetic_data[i][j] == 0:
                    fuel = fuels_categories[0]
                    fuel_cat = synthetic_data[i][j]
                    critical = False
                    no_fuel = True
                else:
                    fuel_cat = np.random.choice([k for k in range(1, len(fuels_categories))])
                    fuel = fuels_categories[fuel_cat]
                    critical = True
                    self.critical_locations.append((i, j))
                    no_fuel = False

                sigma = sigma_fuels[fuel]
                w0 = w0_fuels[fuel]
                fuel_depth = fuels_depth[fuel]
                temp.append(FuelType(fuel, fuel_cat, moisture_content, sigma, w0, fuel_depth,
                                     ovendry_particle_density, low_heat_content, total_mineral_content,
                                     dead_fuel_moisture_extinction, effective_mineral_content, critical, no_fuel))
            fuel_map.append(temp)
        return fuel_map

    def get_state(self):
        """
        Get the current state of the environment.

        Returns:
            torch.Tensor: State tensor.
        """
        state = np.stack((self.automate_state_norm, self.fuel_cat_map_norm, self.altitude_map_norm), axis=-1)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.permute(2, 0, 1)
        state_tensor = F.interpolate(state_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        return state_tensor[0]

    def step(self, wind_dirs):
        """
        Perform a step in the environment.

        Args:
            wind_dirs (list): List of wind directions.

        Returns:
            tuple: Reward, done flag, and score.
        """
        if list(self.wind_dirs) != list(wind_dirs):
            self.wind_dirs = list(wind_dirs)
            self.update_environment_frts()
        new_processed_cell = self.wildfire_propagation()
        reward = 0
        if self.current_cell.fuel_type.critical and new_processed_cell:
            self.done = True
            reward = 10
        self.score = (self.max_sim_t - self.current_t) / self.max_sim_t
        if self.current_t >= self.max_sim_t:
            self.done = True
            reward = -10
            self.score = 0
        return reward, self.done, self.score

    def stochastic_step(self):
        """
        Perform a stochastic step in the environment.

        Returns:
            tuple: Done flag and current time.
        """
        cdir = np.where(self.wind_dirs == 1)[0]
        dev = np.random.choice([0, 1, -1, 2, -2, 3, -3, 4], 1,
                               p=[0.5, 0.125, 0.125, 0.0625, 0.0625, 0.03125, 0.03125, 0.0625])
        wind_dirs = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        if self.mode == 1:
            self.wind_velocity_10 = (np.random.uniform() * 4.5) + 1.5
        wind_dirs[(cdir + dev[0]) % 8] = 1
        if list(self.wind_dirs) != list(wind_dirs):
            self.wind_dirs = wind_dirs
            self.update_environment_frts()
        new_processed_cell = self.wildfire_propagation()
        if self.current_cell.fuel_type.critical and new_processed_cell:
            self.done = True
        return self.done, self.current_t

    def update_environment_frts(self):
        """
        Update the fire reach times (FRTs) in the environment.
        """
        if (self.processed_cells % min(self.Nx, self.Ny)) == 0:
            self.active_cells.clean_burned()
        temp_cells = []
        if len(self.active_cells) > 0:
            for c in self.active_cells.cells:
                if c.parent_cell is not None:
                    frt = c.parent_cell.calculate_frt(c, self.wind_dirs, self.wind_velocity_10)
                    c.frt = (((c.frt - self.current_t) / c.frt) * frt) + self.current_t
                temp_cells.append(c)
        self.active_cells = FRTSortedQueueCells(self)
        for c in temp_cells:
            self.active_cells.add_cell(c)

    def wildfire_propagation(self):
        """
        Perform wildfire propagation in the environment.

        Returns:
            int: 1 if a new cell is processed, 0 otherwise.
        """
        if len(self.active_cells) == 0:
            self.done = True
        else:
            self.current_cell = self.active_cells.pop_cell()
            self.processed_cells += 1
            if self.burned_map[self.current_cell.i][self.current_cell.j] != 1:
                self.current_t = self.current_cell.frt
                self.results[self.current_cell.i][self.current_cell.j] = self.current_cell.frt
                self.burned_map[self.current_cell.i][self.current_cell.j] = 1
                self.automate_state_norm[self.current_cell.i][self.current_cell.j] = 0
                current_cell_neighbours = self.current_cell.get_neighbours(self)
                for cell in current_cell_neighbours:
                    if self.burned_map[cell.i][cell.j] != 1 and not cell.fuel_type.nofuel:
                        self.active_cells.add_cell(cell)
                        self.automate_state_norm[cell.i][cell.j] = 0.666
                return 1
        return 0