COST = 1
class TopSpinState:

    def __init__(self, state, k=4):
        self.state = state
        self.k = k

    def is_goal(self):
        return self.state == list(range(1, len(self.state) + 1))


    def get_state_as_list(self):
        return self.state

    def preform_clockwise_state(self):
        first_element = self.state[0]
        rest_of_list = self.state[1:]
        clockwise_state = rest_of_list + [first_element]
        return clockwise_state

    def preform_counterclockwise_state(self):
        last_element = self.state[-1]
        beginning_of_list = self.state[:-1]
        counterclockwise_state = [last_element] + beginning_of_list
        return counterclockwise_state

    def preform_flipped_state(self):
        first_k_elements = self.state[:self.k]
        rest_of_list = self.state[self.k:]
        flipped_first_k_elements = list(reversed(first_k_elements))
        flipped_state = flipped_first_k_elements + rest_of_list
        return flipped_state

    def get_neighbors(self):
        neighbors = []

        clockwise_state = self.preform_clockwise_state()
        neighbors.append((TopSpinState(state=clockwise_state, k=self.k), COST))

        preform_counterclockwise_state = self.preform_counterclockwise_state()
        neighbors.append((TopSpinState(state=preform_counterclockwise_state, k=self.k), COST))

        preform_flipped_state = self.preform_flipped_state()
        neighbors.append((TopSpinState(state=preform_flipped_state, k=self.k), COST))

        return neighbors

if __name__ == "__main__":
    initial_state = TopSpinState([1, 2, 3], 2)
    print("Initial State:", initial_state)
    print("Is goal:", initial_state.is_goal())
    print("State as list:", initial_state.get_state_as_list())

    print("\nNeighbors:")
    for action in initial_state.get_neighbors():
        print(f"action: {action}, neighbors: {action[0].state}")

    # # Demonstrating the equality function
    # same_state = TopSpinState([3, 2, 1], 2)
    # different_state = TopSpinState([1, 2, 3], 2)
    #
    # print("\nEquality checks:")
    # print(f"Initial state == Same state: {initial_state == same_state}")
    # print(f"Initial state == Different state: {initial_state == different_state}")