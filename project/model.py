from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from project.config import TrainingConfig


class GNN(torch.nn.Module):
    def __init__(self, feature_size: int, edge_size: int, config: TrainingConfig):
        super().__init__()
        self.config = config

        self.conv_layers = ModuleList([])
        self.transform_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(
            in_channels=feature_size,
            out_channels=self.config.embedding_size,
            heads=self.config.n_heads,
            dropout=self.config.dropout_rate,
            edge_dim=edge_size,
            beta=True,
        )
        self.transform1 = Linear(
            in_features=self.config.embedding_size * self.config.n_heads,
            out_features=self.config.embedding_size,
        )
        self.bn1 = BatchNorm1d(num_features=self.config.embedding_size)

        # Other layers
        for layer in range(self.config.n_layers):
            conv = TransformerConv(
                in_channels=self.config.embedding_size,
                out_channels=self.config.embedding_size,
                heads=self.config.n_heads,
                dropout=self.config.dropout_rate,
                edge_dim=edge_size,
                beta=True,
            )
            self.conv_layers.append(conv)

            transform = Linear(
                in_features=self.config.embedding_size * self.config.n_heads,
                out_features=self.config.embedding_size,
            )
            self.transform_layers.append(transform)

            bn = BatchNorm1d(num_features=self.config.embedding_size)
            self.bn_layers.append(bn)

            if layer % self.config.top_k_every_n == 0:
                pooling = TopKPooling(
                    in_channels=self.config.embedding_size,
                    ratio=self.config.top_k_ratio,
                )
                self.pooling_layers.append(pooling)

        # Linear output layers
        self.linear1 = Linear(self.config.embedding_size * 2, self.config.dense_neurons)
        self.linear2 = Linear(self.config.dense_neurons, int(self.config.dense_neurons / 2))
        self.linear3 = Linear(int(self.config.dense_neurons / 2), 1)

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
        for layer in range(self.config.n_layers):
            x = self.conv_layers[layer](x, edge_index, edge_attr)
            x = torch.relu(self.transform_layers[layer](x))
            x = self.bn_layers[layer](x)

            # Always aggregate last layer
            if layer % self.config.top_k_every_n == 0 or layer == self.config.n_layers - 1:
                pooling = self.pooling_layers[int(layer / self.config.top_k_every_n)]
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
            x = F.dropout(x, p=0.8, training=self.training)
            x = torch.relu(self.linear2(x))
            x = F.dropout(x, p=0.8, training=self.training)
            x = self.linear3(x)

            return x
