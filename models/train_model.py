import torch
import torch.nn as nn
from data.load_data import load_data
from data.preprocessing import create_id_mappings
import torch.optim as optim
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get mappings and counts
deck_id_map, card_id_map, num_cards, num_decks, index_to_deck_id = create_id_mappings()

dataloader = load_data(deck_id_map,card_id_map)


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

        # Output layer
        self.output_layer = nn.Linear(embedding_size + mlp_layers[-1], 4)  # 4 classes for 0, 1, 2, 3 cards
        self.softmax = nn.Softmax(dim=1)

    def forward(self, deck_indices, card_indices):
        # GMF part
        deck_embedded_gmf = self.deck_embedding_gmf(deck_indices)
        card_embedded_gmf = self.card_embedding_gmf(card_indices)
        gmf_output = deck_embedded_gmf * card_embedded_gmf

        # MLP part
        deck_embedded_mlp = self.deck_embedding_mlp(deck_indices)
        card_embedded_mlp = self.card_embedding_mlp(card_indices)
        mlp_input = torch.cat([deck_embedded_mlp, card_embedded_mlp], dim=1)
        
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)

        # Concatenate GMF and MLP outputs
        
        concat_output = torch.cat([gmf_output, mlp_input], dim=1)
        # Final prediction
        logits = self.output_layer(concat_output)
        return self.softmax(logits)

# Training setup
num_decks = len(deck_id_map)
num_cards = len(card_id_map)
embedding_size = 128
mlp_layers = [128, 64, 32] 
num_epochs = 5
batch_size = 64
learning_rate = 1e-3

model = NCF(num_decks, num_cards, embedding_size, mlp_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Training loop
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for deck_batch, card_batch, count_batch in dataloader:

        optimizer.zero_grad()
        predictions = model(deck_batch, card_batch)
        
        # Ensure count_batch is of correct shape
        if count_batch.dim() == 1:
            count_batch = count_batch.unsqueeze(1)

        loss = criterion(predictions, count_batch.squeeze())

        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
model_save_path = os.path.join(model_dir, f'ncf_model_epoch_{epoch+1}.pth')
torch.save(model.state_dict(), model_save_path)
