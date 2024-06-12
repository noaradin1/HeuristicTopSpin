import heapq
import heapq
from topspin import TopSpinState
from heuristics import *
from heuristics import BaseHeuristic
class Node:
    def __init__(self, state, g=0, p=None, f=0):
        self.state = state
        self.g = g
        self.p = p
        self.f = f

    def __lt__(self, other):
        return self.f < other.f

def BWAS(start, W, B, heuristic_function, T):
    OPEN = []
    CLOSED = {}
    UB = float('inf')
    nUB = None
    LB = 0
    expansions = 0

    n_start = Node(state=start, g=0, p=None, f=heuristic_function([start])[0])
    heapq.heappush(OPEN, n_start)

    while OPEN and expansions <= T:
        generated = []
        batch_expansions = 0

        while OPEN and batch_expansions < B and expansions <= T:
            n = heapq.heappop(OPEN)
            expansions += 1
            batch_expansions += 1

            if not generated:
                LB = max(n.f, LB)

            if n.state.is_goal():
                if UB > n.g:
                    UB = n.g
                    nUB = n
                continue

            for neighbor, cost in n.state.get_neighbors():
                new_g = n.g + cost
                if neighbor not in CLOSED or new_g < CLOSED[neighbor].g:
                    CLOSED[neighbor] = new_g
                    generated.append((neighbor, new_g, n))

        if LB >= UB:
            return path_to_goal(nUB), expansions

        generated_states = [g[0] for g in generated]
        heuristics = heuristic_function(generated_states)
        # heuristics = [heuristic_function([s]) for s in generated_states]

        for i in range(len(generated)):
            s, g, p = generated[i]
            h = heuristics[i]
            f = g + W * h
            n_s = Node(state=s, g=g, p=p, f=f)
            heapq.heappush(OPEN, n_s)

    return path_to_goal(nUB), expansions


def path_to_goal(node):
    path = []
    while node:
        path.append(node.state)
        node = node.p
    return path[::-1]

# Example usage with TopSpinState
# start_state = TopSpinState(state=[3, 4, 2, 1, 5], k=3)
# W = 1.5
# B = 5
# T = 1000
# base_heuristic = BaseHeuristic()
#
# path, total_expansions = BWAS(start=start_state, W=W, B=B, heuristic_function=base_heuristic.get_h_values, T=T)
# print("Path to goal:", [state.get_state_as_list() for state in path])
# print("Total expansions:", total_expansions)
