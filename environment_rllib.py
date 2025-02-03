import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
# from gym import Env
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import time

from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

from gymnasium import spaces

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.load_capacity = 0

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"

class ForagingEnv_r(MultiAgentEnv):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self
    
    def __init__(
        self,
        players,
        field_size,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward,
        grid_observation,
        penalty,
        spawn_resources_random,
        randomize,
        change_to_random,
        change_number,
        network_cost,
        num_storage,
        num_network,
        storage_level,
        network_level,
        min_consumption,
        ):
        
        super(MultiAgentEnv).__init__()
    
        # Logging and random seed
        self.logger = logging.getLogger(__name__)
        self.seed()
    
        # Player and field setup
        self.players = [Player() for _ in range(players)]
        self.agents_id = ['p' + str(k) for k in range(players)]
        self._agent_ids = self.agents_id
        self.field = np.zeros(field_size, np.int32)
    
        # General environment parameters
        self.max_player_level = 0
        self.sight = sight
        self.force_coop = force_coop
        self.penalty = penalty
        self._game_over = None
        self._max_episode_steps = max_episode_steps
        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation
        self._valid_actions = None
        self._rendering_initialized = False
        self.randomize = randomize
        self.current_iter = 0  # 
    
        # Storage and Network fruit settings
        self.spawn_resources_random = spawn_resources_random
        self.change_to_random = change_to_random
        self.change_number = change_number
        self.storage_layer = np.zeros(field_size, np.int32)
        self.network_layer = np.zeros(field_size, np.int32)
        self.network_cost = network_cost
        self.num_storage = num_storage
        self.num_network = num_network
        self.storage_level = storage_level
        self.network_level = network_level
        self.min_consumption = min_consumption
    
        # Action and observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = self._get_observation_space()
    
        # Player initialization
        self.viewer = None
        self.n_agents = len(self.players)
    
        # Spawn players and fruits
        self.spawn_players(self.max_player_level, self.randomize)
        self.spawn_resources(self.randomize)
        
        # Add this in ForagingEnv_r.__init__()
        self.custom_metrics = {
            "coop_penalty_applied": 0,
            "storage_consumed": 0,
            "network_consumed": 0
        }


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
    
        if self._grid_observation: #if grid observation is true
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)
    
            # Layer for agent presence: values between 0 and max player level	
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.full(grid_shape, np.inf, dtype=np.float32)
    
            # Layer for storage fruits: values between 0 and storage level
            storage_min = np.zeros(grid_shape, dtype=np.float32)
            storage_max = np.ones(grid_shape, dtype=np.float32) * self.storage_level
    
            # Layer for network fruits: values between 0 and network level
            network_min = np.zeros(grid_shape, dtype=np.float32)
            network_max = np.ones(grid_shape, dtype=np.float32) * self.network_level
    
            # Layer for access (binary mask): 0 for blocked, 1 for accessible
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)
    
            # Stack all layers
            min_obs = np.stack([agents_min, storage_min, network_min, access_min])
            max_obs = np.stack([agents_max, storage_max, network_max, access_max])
    
        else:
            # Flat observation: [agent info] + [storage fruit info] + [network fruit info]
            field_x, field_y = self.field.shape
    
            # Agents: (x, y, level) per agent            
            agents_min = [-1, -1, 0] * len(self.players)
            agents_max = [field_x - 1, field_y - 1, np.inf] * len(self.players)
    
            # Storage fruits: (x, y, level) per storage      
            storage_min = [-1, -1, 0] * self.num_storage
            storage_max = [field_x - 1, field_y - 1, self.storage_level] * self.num_storage
    
            # Network fruits: (x, y, level) per network        
            network_min = [-1, -1, 0] * self.num_network
            network_max = [field_x - 1, field_y - 1, self.network_level] * self.num_network
    
            # Concatenate all layers for flat observation
            min_obs = agents_min + storage_min + network_min
            max_obs = agents_max + storage_max + network_max
    
        return gym.spaces.Box(low=np.array(min_obs), high=np.array(max_obs), dtype=np.float32)


    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }
        
    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        # Combine storage, network, and field layers
        combined_field = (
            self.field
            + self.storage_layer
            + self.network_layer
        )
    
        if not ignore_diag:
            # Include diagonals in the neighborhood
            return combined_field[
                max(row - distance, 0): min(row + distance + 1, self.rows),
                max(col - distance, 0): min(col + distance + 1, self.cols)
            ]
        
        # Only include vertical and horizontal neighbors
        return (
            combined_field[
                max(row - distance, 0): min(row + distance + 1, self.rows), col
            ].sum()
            + combined_field[
                row, max(col - distance, 0): min(col + distance + 1, self.cols)
            ].sum()
        )


    def adjacent_resource(self, row, col):
        return (
        self.storage_layer[max(row - 1, 0), col] > 0 or  # Above
        self.storage_layer[min(row + 1, self.rows - 1), col] > 0 or  # Below
        self.storage_layer[row, max(col - 1, 0)] > 0 or  # Left
        self.storage_layer[row, min(col + 1, self.cols - 1)] > 0 or  # Right
        self.network_layer[max(row - 1, 0), col] > 0 or  # Above
        self.network_layer[min(row + 1, self.rows - 1), col] > 0 or  # Below
        self.network_layer[row, max(col - 1, 0)] > 0 or  # Left
        self.network_layer[row, min(col + 1, self.cols - 1)] > 0  # Right
        )
        
    def adjacent_resource_location(self, row, col):
        # Check adjacent cells for Storage fruits
        if row > 1 and self.storage_layer[row - 1, col] > 0:
            return row - 1, col # Return the coordinates if resource is found above
        elif row < self.rows - 1 and self.storage_layer[row + 1, col] > 0:
            return row + 1, col # Return the coordinates if resource is found below
        elif col > 1 and self.storage_layer[row, col - 1] > 0:
            return row, col - 1 # Return the coordinates if food is resource to the left
        elif col < self.cols - 1 and self.storage_layer[row, col + 1] > 0:
            return row, col + 1  # Return the coordinates if resource is found to the right
    
    
        # Check adjacent cells for Network fruits
        if row > 1 and self.network_layer[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.network_layer[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.network_layer[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.network_layer[row, col + 1] > 0:
            return row, col + 1
    
        return None  # No adjacent resource found

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]
    
    def spawn_resources(self, randomize):
        
        if self.change_to_random:  # Check if switching to random mode is enabled
            if self.current_iter <= self.change_number:  # Predefined placement
                fixed_positions = [(1, 1), (1, 4), (4, 1), (4, 4)]
                self.storage_layer = np.zeros(self.field.shape, np.int32)
                self.network_layer = np.zeros(self.field.shape, np.int32)
    
                storage_count = 0
                network_count = 0
    
                for i, (row, col) in enumerate(fixed_positions):
                    if storage_count < self.num_storage:
                        level = self.storage_level
                        self.storage_layer[row, col] = level
                        storage_count += 1
                    elif network_count < self.num_network:
                        level = self.network_level
                        self.network_layer[row, col] = level
                        network_count += 1
    
                    if storage_count >= self.num_storage and network_count >= self.num_network:
                        break
            else:  # Switch to random placement after `change_number` iterations
                while True:
                    self.storage_layer = np.zeros(self.field.shape, np.int32)
                    self.network_layer = np.zeros(self.field.shape, np.int32)
    
                    total_required_resources = self.min_consumption * len(self.players)
                    total_spawned_resources = 0
                    storage_count = 0
                    network_count = 0
                    attempts = 0
    
                    while (total_spawned_resources < total_required_resources or
                           storage_count < self.num_storage or
                           network_count < self.num_network) and attempts < 2000:
    
                        attempts += 1
                        row, col = np.random.randint(1, self.field.shape[0] - 1), np.random.randint(1, self.field.shape[1] - 1)
    
                        if (
                            np.sum(self.storage_layer[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) == 0 and
                            np.sum(self.network_layer[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) == 0 and
                            self._is_empty_location(row, col)
                        ):
                            if storage_count < self.num_storage:
                                level = np.random.randint(1, self.storage_level + 1) if randomize else self.storage_level
                                self.storage_layer[row, col] = level
                                total_spawned_resources += level
                                storage_count += 1
                                continue
    
                            if network_count < self.num_network:
                                level = np.random.randint(1, self.network_level + 1) if randomize else self.network_level
                                self.network_layer[row, col] = level
                                total_spawned_resources += level
                                network_count += 1
    
                    if attempts >= 2000:
                        print("Warning: Max attempts reached. Resources may be insufficient.")
    
                    if (total_spawned_resources >= total_required_resources and
                            storage_count >= self.num_storage and
                            network_count >= self.num_network):
                        break
        else:  # Original logic remains as it is
            if self.spawn_resources_random:  # Randomized resource spawning
                while True:
                    self.storage_layer = np.zeros(self.field.shape, np.int32)
                    self.network_layer = np.zeros(self.field.shape, np.int32)
    
                    total_required_resources = self.min_consumption * len(self.players)
                    total_spawned_resources = 0
                    storage_count = 0
                    network_count = 0
                    attempts = 0
    
                    while (total_spawned_resources < total_required_resources or
                           storage_count < self.num_storage or
                           network_count < self.num_network) and attempts < 2000:
    
                        attempts += 1
                        row, col = np.random.randint(1, self.field.shape[0] - 1), np.random.randint(1, self.field.shape[1] - 1)
    
                        if (
                            np.sum(self.storage_layer[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) == 0 and
                            np.sum(self.network_layer[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) == 0 and
                            self._is_empty_location(row, col)
                        ):
                            if storage_count < self.num_storage:
                                level = np.random.randint(1, self.storage_level + 1) if randomize else self.storage_level
                                self.storage_layer[row, col] = level
                                total_spawned_resources += level
                                storage_count += 1
                                continue
    
                            if network_count < self.num_network:
                                level = np.random.randint(1, self.network_level + 1) if randomize else self.network_level
                                self.network_layer[row, col] = level
                                total_spawned_resources += level
                                network_count += 1
    
                    if attempts >= 2000:
                        print("Warning: Max attempts reached. Resources may be insufficient.")
    
                    if (total_spawned_resources >= total_required_resources and
                            storage_count >= self.num_storage and
                            network_count >= self.num_network):
                        break
            else:  # Fixed position spawning
                fixed_positions = [(1, 1), (1, 4), (4, 1), (4, 4)]
                self.storage_layer = np.zeros(self.field.shape, np.int32)
                self.network_layer = np.zeros(self.field.shape, np.int32)
    
                storage_count = 0
                network_count = 0
    
                for i, (row, col) in enumerate(fixed_positions):
                    if storage_count < self.num_storage:
                        level = self.storage_level
                        self.storage_layer[row, col] = level
                        storage_count += 1
                    elif network_count < self.num_network:
                        level = self.network_level
                        self.network_layer[row, col] = level
                        network_count += 1
    
                    if storage_count >= self.num_storage and network_count >= self.num_network:
                        break
              
    def _is_empty_location(self, row, col):
        # Block cells with resources
        if self.storage_layer[row, col] > 0 or self.network_layer[row, col] > 0:
            return False  
    
        # Block cells with agents
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False
    
        return True



    def spawn_players(self, max_player_level, randomize):
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                # row = np.random.randint(0, self.rows)
                # col = np.random.randint(0, self.cols)
                row = np.random.randint(0, self.rows)
                col = np.random.randint(0, self.cols)
                
                if self._is_empty_location(row, col):
                    agent_level = np.random.randint(0, max_player_level + 1) if randomize else max_player_level
                    player.setup(
                        (row, col),
                        agent_level,
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        row, col = player.position
    
        if action == Action.NONE:
            return True
    
        elif action == Action.NORTH:
            return row > 0 and self._is_empty_location(row - 1, col)
    
        elif action == Action.SOUTH:
            return row < self.rows - 1 and self._is_empty_location(row + 1, col)
    
        elif action == Action.WEST:
            return col > 0 and self._is_empty_location(row, col - 1)
    
        elif action == Action.EAST:
            return col < self.cols - 1 and self._is_empty_location(row, col + 1)
    
        elif action == Action.LOAD:
            # Check only adjacent cells for Storage or Network resources
            return self.adjacent_resource(row, col)
        
        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")    

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood( # Get relative position of player 'a'
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0 # Check if they are within the visible area (non-negative)
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight # Ensure they are within the sight window
            ],
            
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over, # Whether the game has ended
            sight=self.sight, # How far the player can see
            current_step=self.current_step, # Current step in the episode
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            # Initialize the observation array with zeros
            obs_length = self.observation_space.shape[0]
            obs = np.zeros(obs_length, dtype=np.float32)
        
            # Populate agent observations dynamically
            for i, player in enumerate(self.players):
                pos = player.position
                level = player.level 
                idx = i * 3
                obs[idx] = pos[0]  # Row (x-coordinate)
                obs[idx + 1] = pos[1]  # Column (y-coordinate)
                obs[idx + 2] = level  # Actual agent level
        
            # Populate storage fruit observations
            offset = len(self.players) * 3  # Offset for storage data
            for i, (x, y) in enumerate(zip(*np.where(self.storage_layer > 0))):
                idx = offset + i * 3
                obs[idx] = x  # Row
                obs[idx + 1] = y  # Column
                obs[idx + 2] = self.storage_layer[x, y]  # Storage level
        
            # Populate network fruit observations
            offset += self.num_storage * 3  # Offset for network data
            for i, (x, y) in enumerate(zip(*np.where(self.network_layer > 0))):
                idx = offset + i * 3
                obs[idx] = x  # Row
                obs[idx + 1] = y  # Column
                obs[idx + 2] = self.network_layer[x, y]  # Network level
        
            return obs

    
        def make_global_grid_arrays():
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)
    
            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self.sight, player_y + self.sight] = player.level
    
            storage_layer = np.zeros(grid_shape, dtype=np.float32)
            for x, y in zip(*np.where(self.storage_layer > 0)):
                storage_layer[x + self.sight, y + self.sight] = self.storage_layer[x, y]
    
            network_layer = np.zeros(grid_shape, dtype=np.float32)
            for x, y in zip(*np.where(self.network_layer > 0)):
                network_layer[x + self.sight, y + self.sight] = self.network_layer[x, y]
    
            access_layer = np.ones(grid_shape, dtype=np.float32)
            access_layer[:self.sight, :] = 0.0
            access_layer[-self.sight:, :] = 0.0
            access_layer[:, :self.sight] = 0.0
            access_layer[:, -self.sight:] = 0.0
            
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
                
            # Block cells occupied by storage resources
            for x, y in zip(*np.where(self.storage_layer > 0)):
                access_layer[x + self.sight, y + self.sight] = 0.0
    
            # Block cells occupied by network resources
            for x, y in zip(*np.where(self.network_layer > 0)):
                access_layer[x + self.sight, y + self.sight] = 0.0
            
            return np.stack([agents_layer, storage_layer, network_layer, access_layer])
    
        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1
    
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward
        
        # Generate observations for all players
        observations = [self._make_obs(player) for player in self.players]
        
        # Handle observations differently depending on the observation type
        if self._grid_observation:
            # Generate global grid layers for all agents
            layers = make_global_grid_arrays()
            
            # Get grid bounds for each player based on their position
            agents_bounds = [get_agent_grid_bounds(p.position[0], p.position[1]) for p in self.players]
            
            # Slice the global grid to match each player's field of view
            nobs = [layers[:, sx:ex, sy:ey] for sx, ex, sy, ey in agents_bounds]
        else:
            # Use flat observation format if grid observation is disabled
            nobs = [make_obs_array(obs) for obs in observations]
        
        # Extract rewards for each player from their observations
        nreward = [get_player_reward(obs) for obs in observations]
        
        # Check if the game is over for each player
        ndone = [obs.game_over for obs in observations]
        
        # Organize observations, rewards, and done flags into dictionaries
        nobs_dict = {self.agents_id[k]: nobs[k] for k in range(len(self.agents_id))}
        nreward_dict = {self.agents_id[k]: nreward[k] for k in range(len(self.agents_id))}
        ndone_dict = {self.agents_id[k]: ndone[k] for k in range(len(self.agents_id))}
        
        # Determine if all agents are done
        ndone_dict['__all__'] = all(ndone_dict.values())
        
        
        # Validate that all observations are within the allowed observation space
        for agent_id, obs in nobs_dict.items():
            assert self.observation_space.contains(obs), \
                f"obs space error for {agent_id}: obs: {obs}, obs_space: {self.observation_space}"
        
        # Return observations, rewards, done flags, and an empty info dictionary
        return nobs_dict, nreward_dict, ndone_dict, {}


    def reset(self):
        
        self.field = np.zeros(self.field_size, np.int32)  # Reset the grid
        # Reset player positions
        
        self.current_iter += 1 # Add iteration for Curriculum training
        print('current iter:', self.current_iter)
        
        # Spawn fruits (Storage and Network)
        self.spawn_resources(self.randomize)
    
        self.spawn_players(self.max_player_level, self.randomize)
    
        self.current_step = 0  # Reset step counter
        self._game_over = False  # Reset game-over flag
        self._gen_valid_moves()  # Update valid moves
        
        for player in self.players:
            player.starting_level = player.level  # Track initial level
    
        # Generate and return initial observations
        nobs, _, _, _ = self._make_gym_obs()
        return nobs

    def step(self, actions):
        actions = tuple(actions.values())  
        self.current_step += 1
        
    
        # Reset player rewards
        for player in self.players:
            player.reward = 0
        
        # Validate actions
        actions = [
            Action(action) if Action(action) in self._valid_actions[player] else Action.NONE
            for player, action in zip(self.players, actions)
        ]
    
        # Validate actions before applying them
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    f"{player.name} at {player.position} attempted invalid action {action}."
                )
                actions[i] = Action.NONE  # Replace invalid action with NONE
        
                
        loading_players = set()  
        collisions = defaultdict(list)
        
    
    
        # Determine target positions for each action
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)
    
        # Handle collisions and move non-colliding players
        for position, players in collisions.items():
            if len(players) > 1:
                continue  # Multiple players cannot move to the same position
            row, col = position
            
            players[0].position = position  # Move the player
    
    
        # Handle loading actions
        while loading_players:
            player = loading_players.pop()
            row, col = player.position
            
            # Find the exact adjacent resource location
            resource_pos = self.adjacent_resource_location(row, col)
        
            if resource_pos:
                r_row, r_col = resource_pos
                player.load_capacity = max(self.min_consumption - player.level, 0)
                
                if self.storage_layer[r_row, r_col] > 0:
                    if self.force_coop:
                        adjacent_players = self.adjacent_players(r_row, r_col)
                        total_players = adjacent_players   # Include the current player
                    
                        required_players = 2  # Minimum players required for cooperation
                    
                        if len(total_players) >= required_players:
                            # Calculate each player's remaining capacity
                            total_capacity = sum(max(self.min_consumption - p.level, 0) for p in total_players)
                    
                            if total_capacity > 0:
                                total_resource = self.storage_layer[r_row, r_col]
                    
                                # Distribute resources without prematurely clearing them
                                distributed_total = 0
                    
                                            
                    
                                for p in total_players:
                                    #import pdb
                                    #pdb.set_trace()
                                    remaining_capacity = max(self.min_consumption - p.level, 0)
                                    proportion = remaining_capacity / total_capacity
                                    amount_collected = min(round(total_resource * proportion), remaining_capacity)
                    
                                    p.level += amount_collected
                                    p.reward += amount_collected
                                    distributed_total += amount_collected
                    
                                # Deduct the actual distributed amount from the resource
                                self.storage_layer[r_row, r_col] -= distributed_total

                                    
                            else:
                                # Apply penalty for not loading cooperatively
                                for p in total_players:
                                    p.reward -= self.penalty
                                
                        else:
                            # Apply penalty if not enough players are cooperating
                            for p in total_players:
                                p.reward -= self.penalty
                            
                        
                    else:
                        amount = min(self.storage_layer[r_row, r_col], player.load_capacity)
                        self.storage_layer[r_row, r_col] -= amount
                        player.level += amount
                        reward = amount
                        player.reward += reward              
                
                elif self.network_layer[r_row, r_col] > 0:
                    amount = min(self.network_layer[r_row, r_col], player.load_capacity)
                    self.network_layer[r_row, r_col] -= amount
                    player.level += amount
                    reward = amount - (self.network_cost * amount)
                    player.reward += reward              
                    
             
    
        # Check if all players reached the min_consumption
        all_met_min_consumption = all(player.level >= self.min_consumption for player in self.players)
        
        # Update game-over condition
        self._game_over = (
            self.storage_layer.sum() + self.network_layer.sum() == 0  
            or self.current_step >= self._max_episode_steps           
            or all_met_min_consumption                              
        )
        
        if self._game_over:
            for player in self.players:
                # Apply penalty if player didn't meet the minimum consumption
                if player.level < self.min_consumption:
                    # Penalty scaled
                    penalty = self.penalty 
                    player.reward -= penalty
                    
                    

        self._gen_valid_moves()
    
        # Update player rewards
        for player in self.players:
            
            if self._normalize_reward:
                player.reward /= (self.min_consumption * len(self.players))

            player.score += player.reward  
            
            #print('total score:', player.score)
        
        return self._make_gym_obs()

    def _init_render(self):
        from rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True
        
    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
