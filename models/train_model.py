import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from data.preprocessing import create_id_mappings
import torch.optim as optim
import os
import pandas as pd
from model import NCF
from eval import evaluate_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deck_id_map, card_id_map, num_cards, num_decks, index_to_deck_id = create_id_mappings()

def load_data_from_csv(csv_file, batch_size=64):
    full_data = pd.read_csv(csv_file)    
    deck_ids = torch.LongTensor(full_data['deck_id'].values)
    card_ids = torch.LongTensor(full_data['card_id'].values)
    card_counts = torch.LongTensor(full_data['card_count'].values)  # Binary label (1 for positive, 0 for negative)

    dataset = TensorDataset(deck_ids, card_ids, card_counts)
    return dataset

dataset = load_data_from_csv('/Users/huybro/Desktop/ncf/datasets/deck_card_dataset.csv')

# Split dataset into training and testing sets (80% training, 20% testing)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# Training setup
num_decks = len(deck_id_map)
num_cards = len(card_id_map)
embedding_size = 64 * 2
mlp_layers = [128, 64, 32] 
num_epochs = 20
batch_size = 64 
learning_rate = 0.0001

model = NCF(num_decks, num_cards, embedding_size, mlp_layers)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for deck_batch, card_batch, count_batch in train_loader:
        deck_batch, card_batch, count_batch = deck_batch.to(device), card_batch.to(device), count_batch.to(device)

        optimizer.zero_grad()
        predictions = model(deck_batch, card_batch)
        loss = criterion(predictions, count_batch.float())  
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    model_save_path = os.path.join(model_dir, f'ncf_model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)

    optimizer_save_path = os.path.join(model_dir, f'optimizer_epoch_{epoch+1}.pth')
    torch.save(optimizer.state_dict(), optimizer_save_path)

accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)

# Print the evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")