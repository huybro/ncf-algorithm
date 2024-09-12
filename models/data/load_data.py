import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sqlalchemy import create_engine, text
import os
from preprocessing import create_id_mappings

# Database connection setup
engine = create_engine('postgresql://postgres:cghuy2004@localhost:5432/go_rec_sys')

# Fetch all unique deck and card IDs
def get_all_deck_and_card_ids():
    with engine.connect() as connection:
        deck_ids = pd.read_sql_query(text("SELECT DISTINCT deck_id FROM cards_in_deck"), connection)
        card_ids = pd.read_sql_query(text("SELECT DISTINCT card_id FROM cards_in_deck"), connection)
    return deck_ids['deck_id'].values, card_ids['card_id'].values

# Generate the dataset including positive and negative pairs
def generate_dataset(deck_ids, card_ids, num_negative_samples=1):
    with engine.connect() as connection:
        # Fetch existing card-deck pairs
        query = text("""
            SELECT card_id, deck_id, card_count
            FROM cards_in_deck
        """)
        existing_data = pd.read_sql_query(query, connection)

    # Create positive samples DataFrame
    positive_samples = existing_data.copy()
    positive_samples['card_count'] = (positive_samples['card_count'] > 0).astype(int)

    # Create a DataFrame of all unique deck and card pairs
    all_pairs = pd.DataFrame([(deck, card) for deck in deck_ids for card in card_ids], columns=['deck_id', 'card_id'])

    # Identify missing pairs by merging with positive samples
    merged_data = pd.merge(all_pairs, positive_samples, on=['deck_id', 'card_id'], how='left')
    merged_data['card_count'] = merged_data['card_count'].fillna(0).astype(int)  # Fill missing counts with 0

    # Create negative samples
    negative_samples = merged_data[merged_data['card_count'] == 0]
    negative_samples = negative_samples.sample(n=num_negative_samples * len(positive_samples), replace=True)
    negative_samples['card_count'] = 0

    # Combine positive and negative samples
    full_data = pd.concat([positive_samples, negative_samples], ignore_index=True)
    
    return full_data

# Save dataset as CSV file
def save_dataset_as_csv(full_data, filename='deck_card_dataset.csv'):
    output_dir = './datasets'
    os.makedirs(output_dir, exist_ok=True)
    full_data.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Dataset saved to {os.path.join(output_dir, filename)}")

# Main function to save dataset
def save_and_load_data(deck_id_map, card_id_map, num_negative_samples=1):
    deck_ids, card_ids = get_all_deck_and_card_ids()

    # Generate the full dataset including positive and negative pairs
    full_data = generate_dataset(deck_ids, card_ids, num_negative_samples)

    # Map the IDs to their new continuous indices
    full_data['deck_id'] = full_data['deck_id'].map(deck_id_map)
    full_data['card_id'] = full_data['card_id'].map(card_id_map)

    # Save dataset as CSV
    save_dataset_as_csv(full_data)

    return full_data  

deck_id_map, card_id_map, num_cards, num_decks, index_to_deck_id = create_id_mappings()
save_and_load_data(deck_id_map, card_id_map)