import csv
import time 
import numpy as np

import tensorflow as tf
tfkl = tf.keras.layers

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

from IPython.display import clear_output
from IPython import display as ipythondisplay

# Custom Module
from helper_functions import preprocess_obs, create_policy_eval_video, timing


# Policy Class
# Doesn't use any tf functions as numpy turned out to be faster, even compared to graph mode
class Policy():
    """Epsilon greedy policy used to sample an action given the q-values of the current state
    """

    def __init__(self, eps: float=0.2):
        """Initialize the policy

        Arguments:
            eps (float): Parameter to control the sampling probabilities
        """
        
        self.epsilon = eps 

  
    def __call__(self, q_values):
        """Perform epsilon greedy sampling 
        
        Arguments:
            q_values (tf.Tensor): q-values of the current state given by the model network

        Returns:
            action (int): To-be-performed action
        """   

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(len(q_values))
        else:
            action = np.argmax(q_values).astype(np.int32)        
        
        return action


# DDQN Class
class DDQN(tf.keras.Model):
    """DDQN to calculate Q-values for every possible action given the current state"""


    def __init__(self, env):
        """Initialize the DDQN

        Arguments:
            env (gym): Environment the agent interacts with - used to dynamically determine the number of output neurons
        """
        
        super(DDQN, self).__init__()

        self.conv2D1 = tfkl.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding="valid", activation="relu")
        self.conv2D2 = tfkl.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding="valid", activation="relu")
        self.maxpool1 = tfkl.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')

        self.conv2D3 = tfkl.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="valid", activation="relu")
        self.conv2D4 = tfkl.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="valid", activation="relu")
        self.maxpool2 = tfkl.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')

        self.conv2D5 = tfkl.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding="valid", activation="relu")
        self.conv2D6 = tfkl.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding="valid", activation="relu")
        self.flatten = tfkl.Flatten()

        self.dense = tfkl.Dense(units = 64, activation="relu")

        self.out = tfkl.Dense(units=env.action_space.n, activation="linear")


    @tf.function() 
    def call(self, x):
        """Feed input through the network layer by layer to obtain the Q-value estimates
        
        Arguments:
            x (tf.Tensor): Tensor containg the input to the network

        Returns:
            q_values (tf.Tensor): Q-value estimates for all actions
        """   
        
        x = self.conv2D1(x)
        x = self.conv2D2(x)
        x = self.maxpool1(x)

        x = self.conv2D3(x)
        x = self.conv2D4(x)
        x = self.maxpool1(x)

        x = self.conv2D5(x)
        x = self.conv2D6(x)
        x = self.flatten(x)

        x = self.dense(x)

        q_values = self.out(x)

        return q_values


# DDQN Training Functions
@tf.function()
def gradient_step(model_network, s, a, target, importance_sampling_weights, loss_function, optimizer):
    """Perform a gradient step for the given Network by
    1. Propagating the input through the network
    2. Calculating the loss between the networks output and the true targets + applying importance sampling weights
    3. Performing Backpropagation and updating the trainable variables with the calculated gradients 

    Arguments:
        model_network (tf.keras.Model): Given instance of an initialised Network with all its parameters
        s (tf.Tensor): Tensor containing the states 
        a (tf.Tensor): Tensor containing the actions
        target (tf.Tensor): Tensor containing the targets
        importance_sampling_weights (list): List of importance sampling weights applied to be applied within the loss function 
        loss_function (tf.keras.losses): Loss function for training the model 
        optimizer (tf.keras.optimizers): Function from keras defining the to-be-applied optimizer during learning 

    Returns:
        loss (tf.Tensor): Tensor containing the loss of the Network 
        prediction (tf.tensor): Tensor containing the q-values for a given batch of actions and states
    """

    with tf.GradientTape() as tape:

        # 1.
        prediction = tf.gather(model_network(s), a, batch_dims=1)

        # 2.
        loss = loss_function(tf.expand_dims(target, axis=1), tf.expand_dims(prediction, axis=1), sample_weight=importance_sampling_weights)

    # 3.
    gradients = tape.gradient(loss, model_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_network.trainable_variables))

    return loss, prediction


def train_DDQN(env, 
               model_network, 
               target_network, 
               policy, 
               buffer, 
               optimizer,
               resize_env,
               num_epochs: int, 
               env_steps_per_epoch: int, 
               batch_size: int, 
               discount_factor: float, 
               epsilon_decay_factor: float, 
               min_epsilon: float, 
               tau: float, 
               path: str,
               video_steps: int, 
               saving_epoch: int,
               plotting_epoch: int,
               transfer: bool):
    
    """Function that implements the training algorithm for the given Network. 
    Prints out useful information and visualizations per epoch.

    Arguments:
        env (gym): Environment the agent interacts with
        model_network (tf.keras.Model): Given instance of an initialised Network with all its parameters
        target_network (tf.keras.Model): Copy of the model network
        policy (Policy): Policy to perform the actions in the environment
        buffer (PERB): Fifo Prioritized Experience Replay Buffer containing all generated experience samples
        optimizer (tf.keras.optimizers): Function from keras defining the to-be-applied optimizer during learning
        resize_env (tuple): Tuple containing the new size to which the observation should be adjusted
        num_epochs (int): Number of epochs to train
        env_steps_per_epoch (int): Number of environment steps per epoch 
        batch_size (int): Batch size of the dataset
        discount_factor (float): Discount factor for calculating the targets
        epsilon_decay_factor (float): Decay factor for epsilon
        min_epsilon (float): Minimum epsiolon to ensure a base exploration rate
        tau (float): Polyak averaging parameter
        path (str): Path to safe the weights, evaluation videos and training stats to
        video_steps (int): Max number of steps performed to create an evaluation video 
        saving_epoch (int): Number of epochs before saving evaluation video, model weights and training stats
        plotting_epoch (int): Number of epochs before plotting training stats
        transfer (bool): Whether to fill the buffer randomly or with samples generated by a trained model (Transfer Learning) 
    """
    
    tf.keras.backend.clear_session()

    dataset = None
    cumul_reward = 0.0

    wins = []
    train_losses = []
    episode_rewards = []
    
    huber_loss_buffer = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    huber_loss = tf.keras.losses.Huber()

    # Reset the environment and preprocess the observation
    obs_rescaled = preprocess_obs(obs=env.reset(), resize_env=resize_env)
    

    # Warm start the buffer by filling it completely with random samples 
    while buffer.available_samples < (buffer.size-1): 

        # Either randomly fill the buffer or use a trained network
        if transfer:
            action = policy(q_values=tf.squeeze(model_network(tf.expand_dims(obs_rescaled, axis=0))))
        else:
            action = env.action_space.sample()

        # Make a step and preprocess the subsquent observation
        obs_prime, reward, terminal, info = env.step(action)
        obs_prime_rescaled = preprocess_obs(obs=obs_prime, resize_env=resize_env)

        # Append experience sample to buffer with max priority
        buffer.append(experience=(tf.constant(obs_rescaled), tf.constant(action), tf.constant(reward, dtype=tf.float32), tf.constant(obs_prime_rescaled), tf.constant(np.float32(terminal))), 
                      priority=1.0)

        obs_rescaled = obs_prime_rescaled
        cumul_reward += reward

        # After terminal state has been reached the variables are resetted
        if terminal:
            obs = env.reset() 
            obs_rescaled = preprocess_obs(obs=obs, resize_env=resize_env)
            episode_rewards.append(cumul_reward)
            cumul_reward = 0.0
            wins.append(info["flag_get"])
    

    for epoch in range(num_epochs):

        num_steps = 0
        running_average = 0

        start = time.time()

        # Perform polyak averaging 
        target_network.set_weights((1-tau)*np.array(target_network.get_weights(), dtype=object) + tau*np.array(model_network.get_weights(), dtype=object))


        while num_steps < env_steps_per_epoch:
            
            # Sample and perform action + preprocess subsequent observation
            action = policy(q_values=tf.squeeze(model_network(tf.expand_dims(obs_rescaled, axis=0))))
            obs_prime, reward, terminal, info = env.step(action)
            obs_prime_rescaled = preprocess_obs(obs=obs_prime, resize_env=resize_env)

            # Append experience sample to buffer with max priority
            buffer.append(experience=(tf.constant(obs_rescaled), tf.constant(action), tf.constant(reward, dtype=tf.float32), tf.constant(obs_prime_rescaled), tf.constant(np.float32(terminal))), 
                          priority=1.0)

            obs_rescaled = obs_prime_rescaled
            cumul_reward += reward
            num_steps +=1

            # After terminal state has been reached the variables are resetted and epsilon is decayed
            if terminal: 
                obs = env.reset()
                obs_rescaled = preprocess_obs(obs=obs, resize_env=resize_env)
                episode_rewards.append(cumul_reward)
                cumul_reward = 0.0
                wins.append(info["flag_get"])

                # Decay epsilon but ensure a base exploration rate of 10%
                policy.epsilon *= epsilon_decay_factor
                policy.epsilon = max(min_epsilon, policy.epsilon)
        

        # Delete old dataset and build new one
        del dataset
        dataset, sampled_idxs, importance_sampling_weights = buffer.sample(batch_size)
        
    
        for s, a, r, s_prime, terminal in dataset:
            
            # Interplay of model and target network to combat the overestimation bias
            q_prime_idx = tf.math.argmax(model_network(s_prime), axis=1, output_type=tf.dtypes.int32)
            target = r + (1-terminal) * discount_factor*tf.gather(target_network(s_prime), q_prime_idx, batch_dims=1)

            loss, prediction = gradient_step(model_network=model_network, 
                                             s=s,
                                             a=a,
                                             target=target,
                                             importance_sampling_weights=importance_sampling_weights,
                                             loss_function=huber_loss,
                                             optimizer=optimizer)

            running_average = 0.95 * running_average + (1 - 0.95) * loss
            
            # Calculate the error for the given batch of states and actions for updating the priorities
            error = huber_loss_buffer(tf.expand_dims(target, axis=1), tf.expand_dims(prediction, axis=1))
            # Propagate changes through the buffer
            for i in range(len(sampled_idxs)):
                buffer.update(sampled_idxs[i], error[i])


        train_losses.append(float(running_average))
        
        # Anneal the buffers beta parameter to gradually increase importance sampling weighting
        beta = 0.4 + (epoch / (num_epochs/2)) * (1 - 0.4) if epoch < (num_epochs/2) else 1
        buffer.beta = beta

        
        # Create videos of performance and save them
        if epoch%saving_epoch == 0:

            # Make policy greedy for video evaluation
            epsilon_cache = policy.epsilon
            policy.epsilon = 0
            create_policy_eval_video(env=env, 
                                     policy=policy, 
                                     model_network=model_network, 
                                     resize_env=resize_env,
                                     path=f"{path}/Videos/eval_vid_epoch_{epoch}_eps{policy.epsilon}", 
                                     num_episodes=1,
                                     max_steps=video_steps)
            
            # Set epsilon back to previous value
            policy.epsilon = epsilon_cache

            # Save model weights 
            model_network.save_weights(f"{path}/Model_weights/SuperMarioBrosDDQNWeights_epoch_{epoch}")

            # Save training stats for subsequent analysis
            with open(f"{path}/Plotting/losses_rews_wins.csv", "w", encoding='utf8', newline="") as output_file:
                writer = csv.writer(output_file)
                writer.writerow(train_losses)
                writer.writerow(episode_rewards)
                writer.writerow(wins)

                    
        # Plot stats
        if epoch%plotting_epoch == 0:

            clear_output()

            print(f"The last epoch took: {timing(start)} seconds")
            print()

            # Plot loss and cumulative reward
            fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize = (20, 6))
            ax1[0].plot(train_losses)
            ax1[0].set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {epoch} epochs')
            ax1[1].plot(episode_rewards)
            ax1[1].set(ylabel='Cumul Reward', xlabel='Episodes', title=f'Cumulative Reward over {len(episode_rewards)} episode')

            # Plot average reward over the last 100 episodes
            if len(episode_rewards) > 100:
                ax1[1].plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'))
            
            fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
            ax2.set(ylabel='Win Rate', xlabel='Episodes', title=f'Average win rate over {len(wins)} episodes')

            # Plot average win rate over the last 100 episodes
            if len(wins) > 100:
                ax2.plot(np.convolve(wins, np.ones(100)/100, mode='valid'))

            plt.show()