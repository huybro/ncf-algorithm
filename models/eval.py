import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate model function
def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():  # Disable gradient calculation
        for deck_batch, card_batch, count_batch in test_loader:
            deck_batch, card_batch, count_batch = deck_batch.to(device), card_batch.to(device), count_batch.to(device)
            
            # Forward pass
            predictions = model(deck_batch, card_batch)
            
            # Convert predictions to binary labels (0 or 1)
            predicted_labels = (predictions > 0.5).float().cpu().numpy()  # Convert to numpy array
            all_predictions.extend(predicted_labels)
            all_labels.extend(count_batch.cpu().numpy())  # True labels in numpy array
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    return accuracy, precision, recall, f1