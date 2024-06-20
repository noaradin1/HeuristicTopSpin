# imports

import heapq
from topspin import TopSpinState
from heuristics import *

class Node:
    def __init__(self, state, g=0, p=None, f=0):
        self.state = state
        self.g = g
        self.p = p
        self.f = f

    def __lt__(self, other):
        if self.f == other.f:
            # edge case:  comparison nodes g values are equal
            return self.g > other.g
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
                if neighbor not in CLOSED or new_g < CLOSED[neighbor]:
                    CLOSED[neighbor] = new_g
                    generated.append((neighbor, new_g, n))
        if LB >= UB:
            return path_to_goal(nUB), expansions
        generated_states = [g[0] for g in generated]
        if not generated_states:
            # handles edge cases where thera re no generated_states
            continue

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
