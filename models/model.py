import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, layers):
        super(NCF, self).__init__()
        
        # User and Item embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

        # MLP layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i+1]))
            self.fc_layers.append(nn.ReLU())

        
        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        
        x = torch.cat([user_embedded, item_embedded], dim=-1)
        
        for layer in self.fc_layers:
            x = layer(x)
        
        output = self.output_layer(x)
        prediction = self.sigmoid(output)
        
        return prediction.squeeze()

    