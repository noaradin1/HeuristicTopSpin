from random import random
from BWAS import *
from topspin import TopSpinState
from heuristicsdca import *
import random
import numpy as np



def bellmanUpdateTraining(bellman_update_heuristic):
    # Parameters
    # TODO - how we get the batch size paramater if it is sent to BWAS instance
    BATCH_SIZE = 1000  # Assuming a batch size of 32

    # TODO - do we control this paramater? perhaps because it influence the training time?
    NUM_TRAINING_ITERATION = 150
    MAX_MOVES = 100

    for i in range(NUM_TRAINING_ITERATION):
        print(f"starting iteration number {i}")
        # at every training iteration, a minibatch of random states is generated
        # TODO - call noa's implemented method
        base_ = BaseHeuristic(bellman_update_heuristic._n, bellman_update_heuristic._k)
        minibatch = generate_random_states(n=bellman_update_heuristic._n, k=bellman_update_heuristic._k,
                                           num_states=BATCH_SIZE, possible_actions=MAX_MOVES)
        training_labels = []
        for state in minibatch:
            # state.get_neighbors() - our implemented function
            neighbors = state.get_neighbors()
            neighbors_costs = []
            for neighbor, cost in neighbors:
                # TODO - use constant for 1
                neighbor_cost = 1
                # if neighbor is not goal we add the heuristic value, otherwise add 0 - according to formula
                if not neighbor.is_goal():
                    neighbor_cost += bellman_update_heuristic.get_h_values([neighbor])[0]
                neighbors_costs.append(neighbor_cost)
            label = min(neighbors_costs)
            training_labels.append(label)

        # Save the trained model
        if i % 50 == 0:
            bellman_update_heuristic.save_model()
        print("training model...")
        bellman_update_heuristic.train_model(minibatch, training_labels)
        print(np.std(training_labels))
        print(np.max(training_labels))

        # update heuristic after each batch
        # TODO - check when we need to update? do we decide this?

    bellman_update_heuristic.save_model()


def bootstrappingTraining(bootstrapping_heuristic):
    BATCH_SIZE = 1000
    NUM_TRAINING_ITERATION = 50
    W_BWAS = 5
    B_BWAS = 100
    number_of_steps = 100  # initial

    training_examples, training_outputs = [], []
    for iteration in range(NUM_TRAINING_ITERATION):
        print(f'iteration number {iteration}')
        # Generate a minibatch of random states
        random_states = generate_random_states(n=bootstrapping_heuristic._n, k=bootstrapping_heuristic._k,
                                               num_states=BATCH_SIZE, possible_actions=number_of_steps)
        for state in random_states:
            T = 1000  # Initial number of expansions allowed in BWA*
            while True:
                # Run BWA* on the state
                path, num_expansions = BWAS(start=state, W=W_BWAS, B=B_BWAS,
                                            heuristic_function=bootstrapping_heuristic.get_h_values, T=T)
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


def generate_random_states(n, k, num_states, possible_actions):
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
                curr_state = TopSpinState(curr_state.preform_flipped_state(), k)
            elif action == 'clockwise':
                curr_state = TopSpinState(curr_state.preform_clockwise_state(), k)
            else:
                curr_state = TopSpinState(curr_state.preform_counterclockwise_state(), k)
        list_of_states.append(curr_state)

    return list_of_states
