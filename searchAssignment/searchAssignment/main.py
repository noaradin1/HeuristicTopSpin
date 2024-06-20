from heuristics import BaseHeuristic
from heuristics import BellmanUpdateHeuristic, BootstrappingHeuristic
from BWAS import BWAS
from topspin import TopSpinState
from training import *
import time

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance
instance_11 = [11, 4, 3, 9, 10, 2, 1, 5, 6, 7, 8]  # easy instance
#
# start1 = TopSpinState(instance_1, 4)
# base_heuristic = BaseHeuristic(11, 4)
# path, expansions = BWAS(start1, 5, 10, base_heuristic.get_h_values, 1000000)
# print("Path to goal:", [state.get_state_as_list() for state in path])
# print("Total expansions:", expansions)
# # if path is not None:
# #     print(expansions)
# #     for vertex in path:
# #         print(vertex)
# # else:
#     print("unsolvable")

start2 = TopSpinState(instance_1, 4)
BU_heuristic = BellmanUpdateHeuristic(11, 4)
bellmanUpdateTraining(BU_heuristic)
BU_heuristic.load_model()
start_time = time.time()
path, expansions = BWAS(start2, B=1, W=5, heuristic_function=BU_heuristic.get_h_values, T=1000000)
end_time = time.time()
path_to = [state.get_state_as_list() for state in path]
print("Path to goal:", path_to)
print("Total expansions:", expansions)
print("Path length expansions:", len(path_to))
print("Time:", end_time - start_time)

# if path is not None:
#     print(expansions)
#     for vertex in path:
#         print(vertex)
# else:
#     print("unsolvable")
