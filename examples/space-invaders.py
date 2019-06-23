import gym

from baselines import deepq

env = gym.make('SpaceInvaders-v0')
env = deepq.wrap_atari_dqn(env)

model = deepq.learn(
    env,
    "conv_only",
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    hiddens=[256],
    dueling=True,
    total_timesteps=0
)

count = 0
while True:
    count = count + 1
    flag = count % 100 >= 0 and count % 100 <= 6
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        if flag:
            env.render()
        obs, rew, done, _ = env.step(model(obs[None])[0])
        episode_rew += rew
    print(count, "Episode reward", episode_rew)
env.close()