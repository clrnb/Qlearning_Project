import gym
from gym.core import RewardWrapper_maze, RewardWrapper_leaper, ObservationWrapper_maze, ObservationWrapper_leaper
from Q_Agent import Q_Agent
import matplotlib.pyplot as plt

max_steps = 500
#total_episodes = 2000
"""
Default actions in procgen:
actions = [
    ("LEFT", "DOWN"),
    ("LEFT",), 
    ("LEFT", "UP"),
    ("DOWN",), 
    (),
    ("UP",), 
    ("RIGHT", "DOWN"),
    ("RIGHT",), 
    ("RIGHT", "UP"),
    ("D",),
    ("A",),
    ("W",),
    ("S",),
    ("Q",),
    ("E",),
]
"""
action_mapping = {0: 1, 1: 3, 2: 5, 3: 7}


# Function for debugging
def print_state(box):
    assert box.shape == (64, 64), "Input array must be of shape (64, 64)"
    for row in box:
        print(" ".join(map(str, row)))


def maze_game():
    env = gym.make("procgen:procgen-maze-v0", start_level=0, num_levels=200, distribution_mode="easy",
                   use_backgrounds=False, render_mode='human', restrict_themes=True)
    env = RewardWrapper_maze(env)
    env = ObservationWrapper_maze(env)
    total_episodes = 1000
    agent = Q_Agent(env, max_steps, total_episodes)

    for i_episode in range(total_episodes):
        print("episode ", i_episode)
        state = env.reset()

        for t in range(max_steps):
            # action = env.action_space.sample()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action_mapping[action])  # action
            env.render()
            agent.update_Q(state, action, reward, next_state, done, i_episode)
            state = next_state
            if done:
                print(f"Step {t}, Action: {action}, Reward: {reward}, Done: {done}")
                break
    env.close()


def leaper_game():
    env = gym.make("procgen:procgen-leaper-v0", start_level=0, num_levels=200, distribution_mode="easy",
                   use_backgrounds=False, render_mode='human', restrict_themes=True)
    env = RewardWrapper_leaper(env)
    env = ObservationWrapper_leaper(env)
    total_episodes = 2000
    agent = Q_Agent(env, max_steps, total_episodes)

    for i_episode in range(total_episodes):
        print("episode ", i_episode)
        state = env.reset()
        for t in range(max_steps):
            # action = env.action_space.sample()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action_mapping[action])
            env.render()
            agent.update_Q(state, action, reward, next_state, done, i_episode)
            state = next_state
            if done:
                print(f"Step {t}, Action: {action}, Reward: {reward}, Done: {done}")
                break
    env.close()


# Performance code
def parse_file(filename, total_episodes):
    episodes = []
    steps = []
    with open(filename, 'r') as file:
        current_episode = -1
        for line in file:
            if line.startswith("episode"):
                current_episode = int(line.split()[1])
            elif "Done: True" in line and "Reward: 10.0" in line:
                step_info = line.split(", ")
                step_number = int(step_info[0].split()[1])
                episodes.append(current_episode)
                steps.append(step_number)

            if current_episode >= total_episodes:
                break
    return episodes, steps


def calculate_average_steps(episodes, steps, total_episodes):
    total_steps = 0
    completed_episodes = 0

    for i in range(total_episodes):
        if i in episodes:
            idx = episodes.index(i)
            total_steps += steps[idx]
            completed_episodes += 1
        else:
            total_steps += 500  # episode not completed

    if completed_episodes > 0:
        average_steps = total_steps / total_episodes
    else:
        average_steps = 0

    return average_steps

def calculate_cumulative_goals(episodes, total_episodes):
    goals = [0] * total_episodes
    for ep in episodes:
        if ep < total_episodes:
            goals[ep] += 1
    cumulative_goals = [sum(goals[:i + 1]) for i in range(total_episodes)]
    return cumulative_goals

def plot_performance(random_episodes_maze, random_episodes_leaper, qlearning_episodes_maze, qlearning_episodes_leaper):
    random_cumulative_goals_maze = calculate_cumulative_goals(random_episodes_maze, 1000)
    random_cumulative_goals_leaper = calculate_cumulative_goals(random_episodes_leaper, 2000)
    qlearning_cumulative_goals_maze = calculate_cumulative_goals(qlearning_episodes_maze, 1000)
    qlearning_cumulative_goals_leaper = calculate_cumulative_goals(qlearning_episodes_leaper, 2000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(range(1000), random_cumulative_goals_maze, label='Random - Maze', color='blue')
    ax1.plot(range(1000), qlearning_cumulative_goals_maze, label='Q-Learning - Maze', color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Goals')
    ax1.legend()
    ax1.set_title('Cumulative Goals Reached Over Episodes (Maze)')

    ax2.plot(range(2000), random_cumulative_goals_leaper, label='Random - Leaper', color='green')
    ax2.plot(range(2000), qlearning_cumulative_goals_leaper, label='Q-Learning - Leaper', color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Goals')
    ax2.legend()
    ax2.set_title('Cumulative Goals Reached Over Episodes (Leaper)')
    plt.tight_layout()
    plt.show()


def performance_comparison():
    # Q learning
    qlearning_file_maze = 'performance_maze_qlearning2.txt'
    qlearning_file_leaper = 'performance_leaper_qlearning_2000_4.txt'
    # Random
    random_file_maze = 'performance_maze_random.txt'
    random_file_leaper = 'performance_leaper_random2000.txt'

    random_episodes_maze, random_steps_maze = parse_file(random_file_maze, 1000)
    random_episodes_leaper, random_steps_leaper = parse_file(random_file_leaper, 2000)
    qlearning_episodes_maze, qlearning_steps_maze = parse_file(qlearning_file_maze, 1000)
    qlearning_episodes_leaper, qlearning_steps_leaper = parse_file(qlearning_file_leaper, 2000)

    plot_performance(random_episodes_maze, random_episodes_leaper, qlearning_episodes_maze, qlearning_episodes_leaper)

    print("Average steps in random (maze): ", calculate_average_steps(random_episodes_maze, random_steps_maze, 1000))
    print("Average steps in qlearning (maze): ", calculate_average_steps(qlearning_episodes_maze, qlearning_steps_maze, 1000))
    print("Average steps in random (leaper): ", calculate_average_steps(random_episodes_leaper, random_steps_leaper, 2000))
    print("Average steps in qlearning (leaper): ",
          calculate_average_steps(qlearning_episodes_leaper, qlearning_steps_leaper, 2000))

if __name__ == "__main__":
    performance_comparison()
    # game = input("Insert the game name you want to train (maze, leaper): ")
    # if game == "maze":
    #    maze_game()
    # else:
    #    leaper_game()
