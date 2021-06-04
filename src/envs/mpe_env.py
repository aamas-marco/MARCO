from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np


class EnvWrapper:
    def __init__(self, env, bridge=False):
        self.env = env
        self.episode_limit = 25
        self.max_reward = 0
        self.n = 0
        self.last_action = np.zeros((2,))
        self.mf = True
        self.bridge = bridge

    def step(self, actions):
        if self.bridge:
            act = []
            for i in range(2):
                if np.abs(self.last_pos[i][0]) < 0.125 and np.abs(self.last_pos[i][1] < 0.125) and not (
                        self.last_pos[i][1] >= 0 and np.abs(self.last_pos[i][0]) < 0.1):
                    self.env.world.agents[i].movable = False

                if self.env.world.agents[i].movable:
                    act.append([actions[i] // 10, actions[i] % 10])
                else:
                    act.append([actions[i] % 10])
            obs, reward, term, info = self.env.step(act)
        else:
            obs, reward, term, info = self.env.step(
                [[actions[0] // 10, actions[0] % 10], [actions[1] // 10, actions[1] % 10]])
        self.last_obs = obs
        self.last_pos = self._get_agent_pos()
        self.n += 1
        if self.n >= self.episode_limit:
            term[0] = True

        self.last_action = actions.cpu().numpy()
        # import pdb; pdb.set_trace()
        return reward[0], term[0], {}

    def _get_agent_pos(self):
        return [a.state.p_pos for a in self.env.world.agents]

    def reset(self):
        self.last_obs = self.env.reset()
        self.env.world.agents[0].movable = True
        self.env.world.agents[1].movable = True
        self.last_pos = self._get_agent_pos()
        self.n = 0
        return self.last_obs

    def get_state(self):
        if not self.bridge:
            return np.concatenate(self.last_obs)
        else:
            return np.concatenate(self.last_obs + self.last_pos)

    def get_obs(self):
        return self.last_obs

    def get_avail_actions(self):
        return [[1] * 50] * 2

    def close(self):
        self.env.close()

    def get_env_info(self):
        if not self.bridge:
            return {
                "n_agents": 2,
                "n_actions": 50,
                "state_shape": 42,
                "obs_shape": 21,
                "episode_limit": self.episode_limit,
            }
        else:
            return {
                "n_agents": 2,
                "n_actions": 50,
                "state_shape": 46,
                "obs_shape": 21,
                "episode_limit": self.episode_limit,
            }


def make_env(scenario_name, bridge=False, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    try:
        world = scenario.make_world(bridge=bridge)
    except:
        world = scenario.make_world()

    # world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return EnvWrapper(env, bridge=bridge)