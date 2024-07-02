import gym
import numpy as np
from gym3.types import Discrete


'''class MazeActionSpace(gym.spaces.Discrete):
    def __init__(self, env):
        num_actions = 4
        super().__init__(num_actions)
        self.env = env
    def get_valid_actions(self, observation):
        valid_actions = []
        player_object = np.where(observation == self.env.maze_gen.grid.get())        # Convert agent index to x, y coordinates


        min_x = 0
        max_x = self.env.main_width - 1
        min_y = 0
        max_y = self.env.main_height - 1

        # Check if the agent can move up
            if agent_y > min_y and self.env.maze_gen.grid.get(agent_x, agent_y - 1) != self.env.WALL_OBJ:
            valid_actions.append(0)

        # Check if the agent can move down
        if agent_y < max_y and self.env.maze_gen.grid.get(agent_x, agent_y + 1) != self.env.WALL_OBJ:
            valid_actions.append(1)

        # Check if the agent can move left
        if agent_x > min_x and self.env.maze_gen.grid.get(agent_x - 1, agent_y) != self.env.WALL_OBJ:
            valid_actions.append(2)

        # Check if the agent can move right
        if agent_x < max_x and self.env.maze_gen.grid.get(agent_x + 1, agent_y) != self.env.WALL_OBJ:
            valid_actions.append(3)

        return valid_actions

    def sample(self, observation):
        valid_actions = self.get_valid_actions(observation)
        return np.random.choice(valid_actions)'''