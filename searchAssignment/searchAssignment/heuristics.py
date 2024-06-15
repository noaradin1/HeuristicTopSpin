import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from topspin import TopSpinState
class BaseHeuristic:
    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        gaps = []

        for state_as_list in states_as_list:
            gap = 0
            if state_as_list[0] != 1:
                gap = 1

            for i in range(len(state_as_list) - 1):
                if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
                    gap += 1

            gaps.append(gap)

        return gaps

class HeuristicModel(nn.Module):
    def __init__(self, input_dim):
        super(HeuristicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class LearnedHeuristic:
    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k
        self._model = HeuristicModel(n)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = torch.tensor(states)
        with torch.no_grad():
            predictions = self._model(states_tensor).numpy()
        return predictions.flatten()

    def train_model(self, input_data, output_labels, epochs=100):
        print("-------")
        input_as_list = [state.get_state_as_list() for state in input_data]
        inputs = np.array(input_as_list, dtype=np.float32)
        outputs = np.array(output_labels, dtype=np.float32)

        inputs_tensor = torch.tensor(inputs)
        outputs_tensor = torch.tensor(outputs).unsqueeze(1)  # Adding a dimension for the output
        all_losses = []
        for epoch in range(epochs):
            self._model.train()
            self._optimizer.zero_grad()

            predictions = self._model(inputs_tensor)
            loss = self._criterion(predictions, outputs_tensor)
            all_losses.append(loss.detach().item())
            loss.backward()

            self._optimizer.step()
        print(f"loss is: {np.average(all_losses)} :-)")

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path))
        self._model.eval()

class BellmanUpdateHeuristic(LearnedHeuristic):
    def __init__(self, n=11, k=4):
        super().__init__(n, k)
        
    def save_model(self):
        super().save_model('bellman_update_heuristic.pth')

    def load_model(self):
        super().load_model('bellman_update_heuristic.pth')

class BootstrappingHeuristic(LearnedHeuristic):
    def __init__(self, n=11, k=4):
        super().__init__(n, k)

    def save_model(self):
        super().save_model('bootstrapping_heuristic.pth')

    def load_model(self):
        super().load_model('bootstrapping_heuristic.pth')



if __name__ == "__main__":
    states = [TopSpinState([3, 2, 1], 2), TopSpinState([1, 2, 3], 2)]

    # Base heuristic example
    base_heuristic = BaseHeuristic()
    base_h_values = base_heuristic.get_h_values(states)
    print("Base Heuristic h-values:", base_h_values)

    # Learned heuristic example
    learned_heuristic = LearnedHeuristic(n=3, k=2)
    learned_h_values = learned_heuristic.get_h_values(states)
    print("Learned Heuristic h-values:", learned_h_values)

    # Training example
    input_data = [TopSpinState([3, 2, 1], 2), TopSpinState([2, 3, 1], 2)]
    output_labels = [3, 2]
    learned_heuristic.train_model(input_data, output_labels, epochs=10)

    # Save and load model example
    learned_heuristic.save_model('learned_heuristic.pth')
    learned_heuristic.load_model('learned_heuristic.pth')

    # Bellman update heuristic example
    bellman_heuristic = BellmanUpdateHeuristic(n=3, k=2)
    bellman_h_values = bellman_heuristic.get_h_values(states)
    print("Bellman Update Heuristic h-values:", bellman_h_values)
    bellman_heuristic.save_model()
    bellman_heuristic.load_model()

    # Bootstrapping heuristic example
    bootstrap_heuristic = BootstrappingHeuristic(n=3, k=2)
    bootstrap_h_values = bootstrap_heuristic.get_h_values(states)
    print("Bootstrapping Heuristic h-values:", bootstrap_h_values)
    bootstrap_heuristic.save_model()
    bootstrap_heuristic.load_model()