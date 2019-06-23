import gym

from args import parse_arguments
from models.obspolicy import ObsPolicyModel

def run():
    args = parse_arguments()

    print(args)

    env = gym.make(args.environment)
    env.seed(args.seed or 0)

    reward = 0
    done = False

    agent = None
    if args.model == 'obs-policy':
        agent = ObsPolicyModel(args.convs or 1, 16,
                               env.observation_space.shape,
                               env.action_space.n,
                               args.layers or 1,
                               args.neurons or 32)
        print('observ_space:', env.observation_space.shape)
        print('action_space:', env.action_space.n)
        agent.build()

    if agent == None:
        raise('You have chosen an unimplented model/agent.')

    for i in range(args.episode_count):
        total_reward = 0
        print(f'Episode #{str(i)} reward: {total_reward}')
        obs = env.reset()
        agent.reset()
        render_episode = i % args.render == 0 if args.render else False
        while True:
            if render_episode:
                env.render()
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                agent.act(obs, reward, done)
                break

     # Close the env and write monitor result info to disk
    env.close()



if __name__ == '__main__':
    run()
