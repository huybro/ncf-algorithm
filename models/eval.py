def recommend_cards(user_card_list, card_id_map, model, num_recommendations=10):
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
            deck_tensor = user_deck_tensor.new_tensor([0])  # Single "deck" for the user
            card_tensor = user_card_tensor.new_tensor([card_index])
            
            # Predict the relevance score or class
            score = model(deck_tensor, card_tensor)
            
            # Extract the score or class
            if score.dim() > 1:
                score = score.squeeze()  # Adjust based on output format
                score = score[1]  # Example: assuming positive class score for binary classification
            else:
                score = score.squeeze()
            
            recommendations.append((card_index, score.item()))
    
    # Sort recommendations by score in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve the top N recommendations and convert back to original card IDs
    recommended_card_ids = [list(card_id_map.keys())[idx] for idx, _ in recommendations[:num_recommendations]]
    
    return recommended_card_ids


# Example usage
user_card_list = [98567237,72270339,24224830,66328392,46502744]  
recommended_cards = recommend_cards(user_card_list, card_id_map, model)
print("Recommended Cards:", recommended_cards)