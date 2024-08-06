def recommend_decks(user_card_ids, model, card_id_map, deck_id_map, index_to_deck_id, top_n=5):
        # Ensure the model is in evaluation mode
    model.eval()
        # Map the input card IDs to the corresponding indices
    card_indices = [card_id_map.get(card_id, None) for card_id in user_card_ids]
        card_indices = [idx for idx in card_indices if idx is not None]  # Filter out None values

    if not card_indices:
        raise ValueError("None of the provided card IDs are found in the card ID map.")

    # Generate the recommendation for the set of cards
    aggregated_scores = torch.zeros(len(deck_id_map)).to(device)

    with torch.no_grad():
        # Using a placeholder user index (0) since we are only interested in decks
        user_tensor = torch.tensor([0], dtype=torch.long).to(device)  # Placeholder user

        for card_index in card_indices:
            card_tensor = torch.tensor([card_index], dtype=torch.long).to(device)
            scores = model(user_tensor, card_tensor).squeeze()
            aggregated_scores += scores
        # Sort the aggregated scores in descending order and select top N
    top_scores, top_indices = torch.topk(aggregated_scores, top_n)

        # Map the indices back to deck IDs
    recommended_decks = [index_to_deck_id[idx.item()] for idx in top_indices]

    return recommended_decks

input_card_ids = [32519092,43898403,93490856,24224830,10045474,14821890]  # List of card IDs provided by the user
try:
    recommended_decks = recommend_decks(input_card_ids, model, card_id_map, deck_id_map, index_to_deck_id)
    print(f"Top recommended decks based on the given cards: {recommended_decks}")
except ValueError as e:
    print(e)