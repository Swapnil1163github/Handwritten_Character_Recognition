from typing import Tuple

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import pennylane as qml
from pennylane.qnn import TorchLayer


class PCAReducer:
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=True, random_state=42)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        X_np = X.detach().cpu().numpy()
        X_pca = self.pca.fit_transform(X_np)
        return torch.from_numpy(X_pca).float()

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X_np = X.detach().cpu().numpy()
        X_pca = self.pca.transform(X_np)
        return torch.from_numpy(X_pca).float()


class QNNClassifier(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int, n_classes: int, device: torch.device):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        qnode = qml.QNode(circuit, self.dev, interface="torch")
        self.q_layer = TorchLayer(qnode, weight_shapes)
        self.post = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.q_layer(x)
        logits = self.post(z)
        return logits
