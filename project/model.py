from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from project.config import ModelParameters


class GNN(torch.nn.Module):
    def __init__(self, feature_size: int, edge_size: int, params: ModelParameters):
        super().__init__()
        self.params = params

        self.conv_layers = ModuleList([])
        self.transform_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(
            in_channels=feature_size,
            out_channels=self.params.embedding_size,
            heads=self.params.n_heads,
            dropout=self.params.dropout_rate,
            edge_dim=edge_size,
            beta=True,
        )
        self.transform1 = Linear(
            in_features=self.params.embedding_size * self.params.n_heads,
            out_features=self.params.embedding_size,
        )
        self.bn1 = BatchNorm1d(num_features=self.params.embedding_size)

        # Other layers
        for layer in range(self.params.n_layers):
            conv = TransformerConv(
                in_channels=self.params.embedding_size,
                out_channels=self.params.embedding_size,
                heads=self.params.n_heads,
                dropout=self.params.dropout_rate,
                edge_dim=edge_size,
                beta=True,
            )
            self.conv_layers.append(conv)

            transform = Linear(
                in_features=self.params.embedding_size * self.params.n_heads,
                out_features=self.params.embedding_size,
            )
            self.transform_layers.append(transform)

            bn = BatchNorm1d(num_features=self.params.embedding_size)
            self.bn_layers.append(bn)

            if layer % self.params.top_k_every_n == 0:
                pooling = TopKPooling(
                    in_channels=self.params.embedding_size,
                    ratio=self.params.top_k_ratio,
                )
                self.pooling_layers.append(pooling)

        # Linear output layers
        self.linear1 = Linear(self.params.embedding_size * 2, self.params.dense_neurons)
        self.linear2 = Linear(self.params.dense_neurons, int(self.params.dense_neurons / 2))
        self.linear3 = Linear(int(self.params.dense_neurons / 2), 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transform1(x))
        x = self.bn1(x)

        # Hold the intermediate graph representations
        global_representation = []

        # Intermediate blocks
        for layer in range(self.params.n_layers):
            x = self.conv_layers[layer](x, edge_index, edge_attr)
            x = torch.relu(self.transform_layers[layer](x))
            x = self.bn_layers[layer](x)

            # Always aggregate last layer
            if layer % self.params.top_k_every_n == 0 or layer == self.params.n_layers - 1:
                pooling = self.pooling_layers[int(layer / self.params.top_k_every_n)]
                x, edge_index, edge_attr, batch_index, _, _ = pooling(
                    x, edge_index, edge_attr, batch_index
                )
                # Add current representation
                global_representation.append(
                    torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
                )

            x = sum(global_representation)

            # Output block
            x = torch.relu(self.linear1(x))
            x = F.dropout(x, p=self.params.dropout_rate, training=self.training)
            x = torch.relu(self.linear2(x))
            x = F.dropout(x, p=self.params.dropout_rate, training=self.training)
            x = self.linear3(x)

            return x
