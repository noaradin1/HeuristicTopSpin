# imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from topspin import TopSpinState
import torch.nn.functional as F

class BaseHeuristic:
    def __init__(self, n=11, k=4):
        # n - size of the list, k - smaller disk
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
    # using DeepCubeA architecture for the heuristic model; except we didn't utilize batch normalization.
    def __init__(self, input_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks_num = 4
        self.input_dim = input_dim
        self.one_hot_dim = input_dim**2
        if self.one_hot_dim > 0:
            self.fc1 = nn.Linear(self.input_dim * self.one_hot_dim, 5000)
        else:
            self.fc1 = nn.Linear(self.input_dim, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        for _ in range(self.blocks_num):
            fc1_resnet = nn.Linear(1000, 1000)
            fc2_resnet = nn.Linear(1000, 1000)
            self.blocks.append(nn.ModuleList([fc1_resnet, fc2_resnet]))
        # output features dim is 1 -> heuristic value (an integer that represents the estimated dist from goal node)
        self.fc_out = nn.Linear(1000, 1)

    def forward(self, x):
        if self.one_hot_dim > 0:
            x = F.one_hot(x.long(), self.one_hot_dim).float()
            x = x.view(-1, self.input_dim * self.one_hot_dim)
        else:
            x = x.float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        for block in range(self.blocks_num):
            or_input = x
            x = self.blocks[block][0](x)
            x = F.relu(x)
            x = self.blocks[block][1](x)
            x = F.relu(x + or_input)
        # final heuristic value - network output.
        x = self.fc_out(x)
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
        # print(f"loss is: {np.average(all_losses)} :-)")

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