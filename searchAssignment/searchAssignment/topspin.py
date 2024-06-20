COST = 1
class TopSpinState:
    def __init__(self, state, k=4):
        self.state = state
        self.k = k

    def __repr__(self):
        # defines the string representation of State object
        return f"TopSpinState(state={self.state}, k={self.k})"

    def __hash__(self):
        return hash(tuple(self.state))

    def __eq__(self, other):
        return self.state == other.state and self.k == other.k

    def is_goal(self):
        # returns true if the list of integers that represent the states is sorted.
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
        # neighbors from first action - clockwise-state
        clockwise_state = self.preform_clockwise_state()
        neighbors.append((TopSpinState(state=clockwise_state, k=self.k), COST))
        # neighbors from first action - counterclouckwise-state
        preform_counterclockwise_state = self.preform_counterclockwise_state()
        neighbors.append((TopSpinState(state=preform_counterclockwise_state, k=self.k), COST))
        # neighbors from 3rd action - flipped-state
        preform_flipped_state = self.preform_flipped_state()
        neighbors.append((TopSpinState(state=preform_flipped_state, k=self.k), COST))
        return neighbors