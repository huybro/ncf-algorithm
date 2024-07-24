import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from model import NCF

def train_ncf(model, train_data, epochs, batch_size, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Prepare data
    users = torch.LongTensor(train_data['user_id'].values)
    items = torch.LongTensor(train_data['item_id'].values)
    ratings = torch.FloatTensor(train_data['rating'].values)
    dataset = TensorDataset(users, items, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_users, batch_items, batch_ratings in dataloader:
            batch_users, batch_items, batch_ratings = batch_users.to(device), batch_items.to(device), batch_ratings.to(device)
            
            # Forward pass
            predictions = model(batch_users, batch_items)
            loss = criterion(predictions, batch_ratings)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

# Usage example
num_users = 1000
num_items = 500
embedding_size = 32
layers = [64, 32, 16, 8]
model = NCF(num_users, num_items, embedding_size, layers)

epochs = 10
batch_size = 64
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_model = train_ncf(model, train_data, epochs, batch_size, learning_rate, device)