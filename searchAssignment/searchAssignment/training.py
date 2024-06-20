from random import random
from BWAS import *
from topspin import TopSpinState
from heuristics import *
import random
import numpy as np


def bellmanUpdateTraining(bellman_update_heuristic):
    # Parameters
    BATCH_SIZE = 1000
    NUM_TRAINING_ITERATION = 150
    MAX_MOVES = 100

    for i in range(NUM_TRAINING_ITERATION):
        # at every training iteration, a minibatch of random states is generated
        minibatch = generate_random_states(n=bellman_update_heuristic._n, k=bellman_update_heuristic._k,
                                           num_states=BATCH_SIZE, possible_actions=MAX_MOVES)
        training_labels = []
        for state in minibatch:
            neighbors = state.get_neighbors()
            neighbors_costs = []
            for neighbor, cost in neighbors:
                neighbor_cost = 1
                # if neighbor is not goal we add the heuristic value, otherwise add 0 - according to heuristic given formula
                if not neighbor.is_goal():
                    neighbor_cost += bellman_update_heuristic.get_h_values([neighbor])[0]
                neighbors_costs.append(neighbor_cost)
            # label is defined according to the minimum neighbor cost
            label = min(neighbors_costs)
            training_labels.append(label)
        # save the trained model object every 50 iterations
        if i % 50 == 0:
            bellman_update_heuristic.save_model()
        # train the model after each batch of samples (generated states)
        bellman_update_heuristic.train_model(minibatch, training_labels)
        """
        print(np.std(training_labels))
        print(np.max(training_labels))
        """
    bellman_update_heuristic.save_model()


def bootstrappingTraining(bootstrapping_heuristic):
    BATCH_SIZE = 1000
    NUM_TRAINING_ITERATION = 50
    W_BWAS = 5
    B_BWAS = 100
    number_of_steps = 100
    training_examples, training_outputs = [], []
    for iteration in range(NUM_TRAINING_ITERATION):
        # at every training iteration, a minibatch of random states is generated
        random_states = generate_random_states(n=bootstrapping_heuristic._n, k=bootstrapping_heuristic._k,
                                               num_states=BATCH_SIZE, possible_actions=number_of_steps)
        for state in random_states:
            # T - limit on initial number of expansions (in BWA* )
            while True:
                # no failure
                path, num_expansions = BWAS(start=state, W=W_BWAS, B=B_BWAS,
                                            heuristic_function=bootstrapping_heuristic.get_h_values, T=T)
                if path is not None:
                    # If BWA* finds a solution, add training examples
                    solution_length = len(path) - 1
                    for i, s in enumerate(path):
                        training_examples.append(s)
                        training_outputs.append(solution_length - i)
                    break
                else:
                    # If BWA* fails, double T and retry
                    T *= 2
        # after eatch batch of random generated states, train the model.
        bootstrapping_heuristic.train_model(training_examples, training_outputs)
    bootstrapping_heuristic.save_model()


def generate_random_states(n, k, num_states, possible_actions):
    # the functions generated random states by randomly taking 'n' actions from the goal states
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
