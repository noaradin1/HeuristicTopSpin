from random import random
from BWAS import *
from topspin import TopSpinState
from heuristics import *
import random


def bellmanUpdateTraining(bellman_update_heuristic):
    pass


def bootstrappingTraining(bootstrapping_heuristic):
    minibatch_size = 10  # Assuming a batch size of 32
    num_iterations = 2  # Number of training iterations
    T = 10  # Initial number of expansions allowed in BWA*
    for _ in range(num_iterations):
        # Generate a minibatch of random states
        random_states = generate_random_states(bootstrapping_heuristic._n,bootstrapping_heuristic._k, minibatch_size,4)
        # minibatch = [TopSpinState(state, bootstrapping_heuristic._k) for state in random_states]
        T = 10
        training_examples, training_outputs = [], []
        for state in random_states:
            while True:
                # Run BWA* on the state
                path, num_expansions = BWAS(start=state, W=1.0, B=len(random_states),heuristic_function=bootstrapping_heuristic.get_h_values, T=T )
                if path is not None:
                    # If BWA* finds a solution, add training examples
                    solution_length = len(path) - 1
                    for i, s in enumerate(path):
                        training_examples.append(s)
                        training_outputs.append(solution_length - i)
                    # Train the heuristic model with the training examples
                    break
                else:
                    # If BWA* fails, double T and retry
                    T *= 2
    bootstrapping_heuristic.train_model(training_examples, training_outputs)
    # Save the trained model
    bootstrapping_heuristic.save_model()
    print('model saved :D')


def generate_random_states(n, k, num_states,possible_actions):
    # Generate random states, e.g., by randomly taking actions from the goal states
    goal_state = TopSpinState(list(range(1, n + 1)), k)
    list_of_states = []
    list_of_actions = ['flip', 'clockwise', 'counterclockwise']
    for _ in range(num_states):
        num_of_actions = random.randint(1, possible_actions)
        curr_state = goal_state
        for _ in range(num_of_actions):
            action = random.choice(list_of_actions)
            if action == 'flip':
                curr_state = TopSpinState(curr_state.preform_flipped_state(),k)
            elif action == 'clockwise':
                curr_state = TopSpinState(curr_state.preform_clockwise_state(),k)
            else:
                curr_state = TopSpinState(curr_state.preform_counterclockwise_state(),k)
        list_of_states.append(curr_state)

    return list_of_states

print(generate_random_states(4, 2,1,4))