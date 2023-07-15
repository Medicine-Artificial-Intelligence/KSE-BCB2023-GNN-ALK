import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp,global_add_pool as gadp
torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN, self).__init__()
        self.embedding_size = model_params["model_embedding_size"]
        n_heads = 3
        self.n_layers = model_params["model_layers"]
        self.dropout_rate = model_params["model_dropout_rate"]
        self.top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = 1
        self.dense_neurons = model_params["model_dense_neurons"]
        edge_dim = 11
        #batch = model_params["batch_size"]
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size, 
                                    int(self.embedding_size), 
                                    heads=n_heads, 
                                    dropout=self.dropout_rate,
                                    edge_dim=edge_dim,
                                    beta=True) 

        self.transf1 = Linear(int(self.embedding_size)*n_heads, int(self.embedding_size))
        self.bn1 = BatchNorm1d(int(self.embedding_size))

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(int(self.embedding_size), 
                                                    int(self.embedding_size), 
                                                    heads=n_heads, 
                                                    dropout=self.dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(int(self.embedding_size)*n_heads, int(self.embedding_size)))
            self.bn_layers.append(BatchNorm1d(int(self.embedding_size)))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(int(self.embedding_size), ratio=self.top_k_ratio))
            

        # Linear layers
        self.linear1 = Linear(int(self.embedding_size)*3, int(self.dense_neurons))
        self.linear2 = Linear(int(self.dense_neurons), int(self.dense_neurons/2))  
        self.linear3 = Linear(int(self.dense_neurons/2), 1)  

    def forward(self, x, edge_attr, edge_index, batch_index,model_params):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []
        rate = model_params["mlp_dropout"]
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                    )
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index),gadp(x,batch_index)], dim=1))
    
        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=rate, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=rate, training=self.training)
        x = self.linear3(x)

        return x

