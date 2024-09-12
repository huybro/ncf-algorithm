import torch
from data.preprocessing import create_id_mappings
import os
from model import NCF


def recommend_cards(user_card_list, card_id_map, model, device, num_recommendations=10):

    # Encode the user's cards using the card_id_map
    user_card_indices = [card_id_map[card_id] for card_id in user_card_list if card_id in card_id_map]
    user_deck_indices = [0] * len(user_card_indices)  # Assuming the user is considered as a single "deck"
    
    # Convert to tensors
    user_deck_tensor = torch.LongTensor(user_deck_indices).to(device)
    user_card_tensor = torch.LongTensor(user_card_indices).to(device)
    
    # Get all card indices excluding those in the user's list
    all_card_indices = set(range(len(card_id_map)))
    user_card_indices_set = set(user_card_indices)
    candidate_card_indices = list(all_card_indices - user_card_indices_set)
    
    recommendations = []
    model.eval()
    with torch.no_grad():
        for card_index in candidate_card_indices:
            deck_tensor = torch.LongTensor([0]).to(device)  # Single "deck" for the user
            card_tensor = torch.LongTensor([card_index]).to(device)
            
            # Predict the relevance score
            score = model(deck_tensor, card_tensor).squeeze()
            recommendations.append((card_index, score.item()))
    
    # Sort recommendations by score in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve the top N recommendations and convert back to original card IDs
    recommended_card_ids = [list(card_id_map.keys())[idx] for idx, _ in recommendations[:num_recommendations]]
    
    return recommended_card_ids




deck_id_map, card_id_map, num_cards, num_decks, index_to_deck_id = create_id_mappings()

# Model parameters
num_decks = len(deck_id_map)
num_cards = len(card_id_map)
embedding_size = 64 * 2
mlp_layers = [128, 64, 32] 
num_epochs = 20
batch_size = 64 
learning_rate = 0.0001

# Instantiate the NCF model
model = NCF(num_decks, num_cards, embedding_size, mlp_layers)

# Example user card list (IDs)
user_card_list = [101206024, 72270339, 18144507, 2857636, 89023486, 98567237]  # deck 513617

# Load the saved model
model_dir = 'saved_models'
model_filename = 'ncf_model_epoch_20.pth' 
model_path = os.path.join(model_dir, model_filename)

# Load model state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Get recommended cards
recommended_cards = recommend_cards(user_card_list, card_id_map, model, device=device, num_recommendations=10)

# Print recommended cards
print("Recommended Cards:", recommended_cards)