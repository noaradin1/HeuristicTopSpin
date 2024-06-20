from heuristics import BaseHeuristic
from heuristics import BellmanUpdateHeuristic, BootstrappingHeuristic
from BWAS import BWAS
from topspin import TopSpinState
from training import *
import time
import pandas as pd
import random

# instances

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance
instance_11 = [11, 4, 3, 9, 10, 2, 1, 5, 6, 7, 8]  # easy instance


"""
# training the models - we load the trained model in this code, assuming it is already trained. 

bootstrappingTraining(BS_heuristic)
bellmanUpdateTraining(BU_heuristics)

"""
final_results = {"Heuristic":[],"Instance":[],"B":[], "W":[], "Total expansions":[],"Path length expansions": [],"Time":[]}

random.seed(1)

# Generate 50 States and Evaluate with 3 heuristics

random_states = [tss.state for tss in generate_random_states(n=11, k=4, num_states=50, possible_actions=100)]

# BellmanUpdateheuristic

for i, instance in enumerate(random_states):
    start2 = TopSpinState(instance, 4)
    BU_heuristic = BellmanUpdateHeuristic(11, 4)
    BU_heuristic.load_model()
    for B_, W_ in [(1,2),(100,2),(1,5),(100,5)]:
        try:
            start_time = time.time()
            path, expansions = BWAS(start2, B=B_, W=W_, heuristic_function=BU_heuristic.get_h_values, T=1000000)
            end_time = time.time()
            path_to = [state.get_state_as_list() for state in path]
            final_results["Heuristic"].append("Bellman")
            final_results["Instance"].append(i)
            final_results["B"].append( B_)
            final_results["W"].append(W_)
            final_results["Total expansions"].append(expansions)
            final_results["Path length expansions"].append(len(path_to))
            final_results["Time"].append(end_time - start_time)
        except:
            print(f"BUG in B: {B_}, W: {W_}")

# BootstrapingHeuristic

for i, instance in enumerate(random_states):
    start2 = TopSpinState(instance, 4)
    BU_heuristic = BootstrappingHeuristic(11, 4)
    BU_heuristic.load_model()
    for B_, W_ in [(1,2),(100,2),(1,5),(100,5)]:
        try:
            start_time = time.time()
            path, expansions = BWAS(start2, B=B_, W=W_, heuristic_function=BU_heuristic.get_h_values, T=1000000)
            end_time = time.time()
            path_to = [state.get_state_as_list() for state in path]
            final_results["Heuristic"].append("Bootstrapping")
            final_results["Instance"].append(i)
            final_results["B"].append( B_)
            final_results["W"].append(W_)
            final_results["Total expansions"].append(expansions)
            final_results["Path length expansions"].append(len(path_to))
            final_results["Time"].append(end_time - start_time)
        except:
            print(f"BUG in B: {B_}, W: {W_}")

# BaseHeuristic

for i, instance in enumerate(random_states):
    start2 = TopSpinState(instance, 4)
    BU_heuristic = BaseHeuristic(11, 4)
    # BU_heuristic.load_model()
    for B_, W_ in [(1,2),(100,2),(1,5),(100,5)]:
        try:
            start_time = time.time()
            path, expansions = BWAS(start2, B=B_, W=W_, heuristic_function=BU_heuristic.get_h_values, T=1000000)
            end_time = time.time()
            path_to = [state.get_state_as_list() for state in path]
            final_results["Heuristic"].append("Base")
            final_results["Instance"].append(i)
            final_results["B"].append( B_)
            final_results["W"].append(W_)
            final_results["Total expansions"].append(expansions)
            final_results["Path length expansions"].append(len(path_to))
            final_results["Time"].append(end_time - start_time)
        except:
            print(f"BUG in B: {B_}, W: {W_}")

df = pd.DataFrame(final_results)

df.groupby(['W', 'B', 'Heuristic'])[['Total expansions', 'Path length expansions', 'Time']].mean()

print(df)