import gym

class SkipObs(gym.Wrapper):
    """Wrapper class that repeats the chosen action for a given amount of frames
    """

    def __init__(self, env=None, skip: int=4, reward_scale_factor: int=1):
        """Initialize the wrapper

        Arguments:
            env (gym): Environment the wrapper should be applied to
            skip (int): Number of frames to skip
            reward_scale_factor (int): Factor to scale the rewards with
        """

        super(SkipObs, self).__init__(env)

        self.skip = skip
        self.reward_scale_factor = reward_scale_factor


    def step(self, action: int):
        """ Function to overwrite the environments step function

        Arguments:
            action (int): The chosen action

        Returns:
            obs (nd.array): Last observation from the environment
            total_reward (float): Total reward from all steps (including skipped ones)
            done (boolean): Whether the episode is finished or not
            info (dict): Dictionary containing information about the environments state
        """

        total_reward = 0.0
        done = None

        # Apply the same action to "skip" many frames and calculate the total reward
        for _ in range(self.skip):

            obs, reward, done, info = self.env.step(action)

            # Reward scaling
            total_reward += reward / self.reward_scale_factor

            if done:
                break

        return obs, total_reward, done, info
