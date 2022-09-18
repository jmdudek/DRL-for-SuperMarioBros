import numpy as np
import tensorflow as tf


# Sum Tree Class

class Node():
    """Class for creating interdependent nodes in the sum tree
    """
    
    def __init__(self, left, right, is_leaf: bool=False, idx: int=None):
        """Initializes a node that can either be a root, intermediate or leaf node. 

        Arguments:
            left (Node): Left child node
            right (Node): Right child node
            is_leaf (bool): Whether the node is a leaf or not
            idx (int): Index of the corresponding experience sample 
        """
        
        # Set left and right children
        self.left = left
        self.right = right

        # Whether node is leaf or not
        self.is_leaf = is_leaf

        # Value is calculated as the sum of the value of both children
        self.value = sum(n.value for n in (left, right) if n is not None)
        self.parent = None

        # Index of the experience sample (only set for leaf nodes)
        self.idx = idx 

        # Set node to be the parent of both its children 
        if left is not None:
            left.parent = self

        if right is not None:
            right.parent = self


    @classmethod
    def create_leaf(cls, value, idx):
        """Classmethod that creates leaf nodes with given priority and experience sample index

        Arguments:
            value (float): Priority value of the respective experience sample
            idx (int): Index of the corresponding experience sample
        
        Returns:
            leaf (Node): A leaf node 
        """
        
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value

        return leaf


# Auxiliary functions for the Sum Tree:

def create_tree(input: list):
    """Recursively creates tree bottom-up, by creating parents until root is reached

    Arguments:
        input (list): List of nodes to build the tree with 
    
    Returns:
        (Node): Root node 
        leaf_nodes (list): List of all leafe nodes 
    """
    
    # Create leafs for given inputs
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes

    # Run until the root is reached i.e. only one node is left in the list
    while len(nodes) > 1:

        inodes = iter(nodes)

        # Create parents of current nodes
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    
    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    """Function to retrieve a sample from the tree by recursively traversing it according to the priorities

    Arguments:
        value (float): Value for traversing the tree
        node (Node): Current node to be checked
    
    Retruns:
        (Node): Sampled leafe node
    """
    
    # Return node if leaf is reached
    if node.is_leaf:
        return node
    
    # First compare with left child, if value smaller retrieve left node
    if node.left.value >= value: 
        return retrieve(value, node.left)

    # Else retrieve right node and substract left node's value from current one
    else:
        return retrieve(value - node.left.value, node.right)

     
def update(node: Node, new_value: float):
    """Function to update the priority of a leaf node

    Arguments:
        node (Node): Leaf node to change the value for
        new_value (float): New value for the given node 
    """    
    
    change = new_value - node.value
    node.value = new_value

    # Function call to propagate the change
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    """Function to propagate a change in priority of a leaf node up to the root

    Arguments:
        change (float): To-be-added value 
        node (Node): Current node to change the value for
    """
    
    node.value += change

    # Run until root is reached
    if node.parent is not None:
        propagate_changes(change, node.parent)  


# PERB Class

class PERB(object):
    """Class for the the prioritized experience replay buffer
    """
    
    def __init__(self, size: int, img_dim):
        """Initialize PERB with a buffer (save experience) and sum tree (save priorities and sample accordingly)

        Arguments:
            size (int): Size of the buffer
            img_dim (tuple): Tuple containing the dimensions of the preprocessed observations
        """
        
        # Buffer size
        self.size = size

        # Position where new experience samples are placed
        self.curr_write_idx = 0

        # How many samples are in the buffer
        self.available_samples = 0

        # Init buffer with zeros
        self.buffer = [(np.zeros((img_dim[0], img_dim[1], img_dim[2]), dtype=np.float32), 
                        0.0, 
                        0.0, 
                        np.zeros((img_dim[0], img_dim[1], img_dim[2]), dtype=np.float32), 
                        0.0) 
                       for i in range(self.size)]

        # Create sum tree
        self.base_node, self.leaf_nodes = create_tree([0 for i in range(self.size)])

        # Indices of experience sample content
        self.obs_idx = 0
        self.action_idx = 1
        self.reward_idx = 2
        self.obs_prime_idx = 3
        self.terminal_idx = 4

        # Parameters for PERB to control importance sampling weighting 
        # and trade-off between greedy prioritization and uniform sampling
        self.beta = 0.4
        self.alpha = 0.6

        # Ensure non-zero prios
        self.min_priority = 0.01

        
    def append(self, experience: tuple, priority: float):
        """Function to append a new experience sample to the buffer and update the priorities

        Arguments:
            experience (tuple): Experience sample from the environment step
            priority (float): Priority of the experience sample
        """
        
        # Save sample in buffer at the current write index
        self.buffer[self.curr_write_idx] = experience

        # Update correspoding priority
        self.update(self.curr_write_idx, priority)

        # Increase current write index
        self.curr_write_idx += 1

        # Reset the current write index if it is greater than the allowed size
        if self.curr_write_idx >= self.size:
            self.curr_write_idx = 0

        # Increase number of samples accordingly
        if self.available_samples + 1 < self.size:
            self.available_samples += 1
        else:
            self.available_samples = self.size - 1

            
    def update(self, idx: int, priority: float):
        """Function to update the sum tree

        Arguments:
            idx (int): Index of the to-be-updated leaf node
            priority (float): New priority value
        """
        
        # Adjust priority before updating by calling auxiliary update function
        update(self.leaf_nodes[idx], self.adjust_priority(priority))

        
    def adjust_priority(self, priority: float):
        """Function to adjust a priority 

        Arguments:
            priority (float): To-be-adjusted priority

        Returns:
            adjusted_priority (float): Adjusted priority
        """

        # Add min prio to current prio and raise sum to the power of alpha
        adjusted_priority = np.power(priority + self.min_priority, self.alpha)

        return adjusted_priority

    
    def sample(self, num_samples: int):
        """Function for sampling 

        Arguments:
            num_samples (int): Number of samples to draw from the buffer

        Returns:
            dataset (tf.data.Dataset): Dataset containing the sampled experiences 
            sampled_idxs (list): List of the sampled experience indices
            importance_sampling_weights (list): List of the corresponding importance sampling weights 
        """


        sampled_idxs = []
        importance_sampling_weights = []
        sample_no = 0
        
        # Sample "num_samples" indices from the sum tree
        while sample_no < num_samples:

            # Uniformly sample between 0 and the summed priority (stored in the root node) 
            # used for retrieving an experience sample index (i.e., sampling according to priorities)
            sample_val = np.random.uniform(0, self.base_node.value)
            sampled_node = retrieve(sample_val, self.base_node)

            if sampled_node.idx < self.available_samples - 1:

                sampled_idxs.append(sampled_node.idx)

                # Divide the sampled nodes' priority by the summed priority and caluclate the importance sampling weight
                p = sampled_node.value / self.base_node.value
                importance_sampling_weights.append((self.available_samples + 1) * p)

                sample_no += 1
                

        # Apply the beta factor and normalize so that the maximum importance sampling weight < 1
        importance_sampling_weights = np.array(importance_sampling_weights)
        importance_sampling_weights = np.power(importance_sampling_weights, -self.beta)
        importance_sampling_weights = importance_sampling_weights / np.max(importance_sampling_weights)
        
        
        obs, actions, rewards, obs_prime, terminal = [], [], [], [], []
        
        for idx in sampled_idxs:
            # Append the sampled experience components from the buffer to the respective lists
            obs.append(self.buffer[idx][self.obs_idx])
            actions.append(self.buffer[idx][self.action_idx])
            rewards.append(self.buffer[idx][self.reward_idx])
            obs_prime.append(self.buffer[idx][self.obs_prime_idx])
            terminal.append(self.buffer[idx][self.terminal_idx])
    
        dataset = tf.data.Dataset.from_tensor_slices((obs, actions, rewards, obs_prime, terminal)).batch(num_samples)

        return dataset, sampled_idxs, importance_sampling_weights