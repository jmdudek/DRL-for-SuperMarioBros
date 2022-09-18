import time
import imageio
import numpy as np
import tensorflow as tf

import gym
from gym.wrappers import FrameStack

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# Custom Module
from wrapper_skip_obs import SkipObs


# For environment creation
def make_env(level: str, movement_type: list, num_skip: int=4, num_stack: int=4, reward_scale_factor: int=1):
    """Function to apply all wrappers to the environment

    Arguments:
        level (str): Level to create
        movement_type (list): Defines the joypad space e.g. allow movement to the right only
        num_skip (int): Number of frames to skip
        num_stack (int): Number of frames stack
        reward_scale_factor (int): Factor to scale the rewards with 

    Returns:
        env (gym): Wrapped environment
    """

    # Create Env from given level
    env = gym_super_mario_bros.make(level)

    # Select joypad space 
    env = JoypadSpace(env=env, actions=movement_type)

    # Skip "num_skip" frames
    env = SkipObs(env=env, skip=num_skip, reward_scale_factor=reward_scale_factor)

    # Wrapper that stacks "num_stack" frames resulting in observations being of size (num_stack,240,256,3) 
    env = FrameStack(env=env, num_stack=num_stack) 
   
    return env


@tf.function
def preprocess_obs(obs, resize_env):
    """ Function to preprocess the observations before they are fed into the network 
    1. Rescale the pixel values between [0,1]
    2. Convert from RGB to greyscale
    3. Resize to given resize_env value
    4. Transpose stacked images such that consecutive ones are within the channel dimension

    Arguments:
        obs (gym.wrappers.LazyFrames): Stacked observations from the environment
        resize_env (tuple): Tuple containing the new size to which the observation should be adjusted

    Returns:
        preprocessed_obs (tf.tensor): Preprocessed observation
    """

    preprocessed_obs = tf.convert_to_tensor(obs/255)
    preprocessed_obs = tf.image.rgb_to_grayscale(preprocessed_obs)
    preprocessed_obs = tf.image.resize(preprocessed_obs, size=resize_env)
    preprocessed_obs = tf.transpose(tf.squeeze(preprocessed_obs), perm=[1,2,0])
    
    return preprocessed_obs


# Functions for visualizing training performance:
def create_policy_eval_video(env, policy, model_network, resize_env, path: str, num_episodes: int=1, fps: int=30, max_steps: int=1000):
    """Function to save an mp4 file of the given models performance
    
     Arguments:
            env (gym): Environment the agent interacts with
            policy (Policy): Policy to perform the actions in the environment
            model_network (tf.keras.Model): Model to calculate the Q-values
            resize_env (tuple): Tuple containing the new size to which the observation should be adjusted
            path (str): Directory to which the video should be saved to
            num_episodes (int): Number of to be saved episodes 
            fps (int): Frames per second of the to be saved video
            max_steps (int): Maximum amount of steps per episode
    """
    
    path += ".mp4"

    # Use imageio library for creating and saving an .mp4 video
    with imageio.get_writer(path, fps=fps) as video:


        for _ in range(num_episodes):
            obs = env.reset()
            video.append_data(env.render(mode='rgb_array'))
            done = False
            counter = 0

            while not done:
                action = policy(tf.squeeze(model_network(tf.expand_dims(preprocess_obs(obs=obs, resize_env=resize_env), axis=0))))
                obs, _, done, _  = env.step(action)
                video.append_data(env.render(mode='rgb_array'))
                counter += 1

                if counter > max_steps:
                    done = True
    env.reset()


def timing(start):
    """Function to time the duration of each epoch

    Arguments:
        start (time): Start time needed for computation 
    
    Returns:
        time_per_training_step (time): Rounded time in seconds 
    """
    
    now = time.time()
    time_per_training_step = now - start
    
    return round(time_per_training_step, 4)