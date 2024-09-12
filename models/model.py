import torch.nn as nn
import torch


class NCF(nn.Module):
    def __init__(self, num_decks, num_cards, embedding_size, mlp_layers, dropout=0.2):
        super(NCF, self).__init__()
        
        # GMF part
        self.deck_embedding_gmf = nn.Embedding(num_decks, embedding_size)
        self.card_embedding_gmf = nn.Embedding(num_cards, embedding_size)
        
        # MLP part
        self.deck_embedding_mlp = nn.Embedding(num_decks, embedding_size)
        self.card_embedding_mlp = nn.Embedding(num_cards, embedding_size)
        
        self.mlp_layers = nn.ModuleList()
        input_size = embedding_size * 2
        for output_size in mlp_layers:
            self.mlp_layers.append(nn.Linear(input_size, output_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = output_size


        self.output_layer = nn.Linear(embedding_size + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, deck_indices, card_indices):
        # GMF part
        deck_embedded_gmf = self.deck_embedding_gmf(deck_indices)
        card_embedded_gmf = self.card_embedding_gmf(card_indices)
        gmf_output = deck_embedded_gmf * card_embedded_gmf

        deck_embedded_mlp = self.deck_embedding_mlp(deck_indices)
        card_embedded_mlp = self.card_embedding_mlp(card_indices)
        mlp_input = torch.cat([deck_embedded_mlp, card_embedded_mlp], dim=1)
        
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)

        concat_output = torch.cat([gmf_output, mlp_input], dim=1)
        
        logits = self.output_layer(concat_output)
        probabilities = logits
        return probabilities.squeeze()
