from sqlalchemy import create_engine, text
import pandas as pd
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

# Database connection setup
engine = create_engine('postgresql://postgres:cghuy2004@localhost:5432/go_rec_sys')

# Fetch data in chunks from the database
def get_data_iterator(chunk_size=10000):
    offset = 0
    with engine.connect() as connection:
        while True:
            query = text("""
                SELECT card_id, deck_id, card_count
                FROM cards_in_deck
                ORDER BY card_id
                LIMIT :chunk_size OFFSET :offset
            """)
            result = connection.execute(query, {"chunk_size": chunk_size, "offset": offset})
            chunk = pd.DataFrame(result.fetchall(), columns=result.keys())
            if chunk.empty:
                break
            yield chunk
            offset += chunk_size

# Convert a dataframe chunk into a PyTorch TensorDataset
def process_chunk(chunk):
    cards = torch.LongTensor(chunk['card_id'].values)
    decks = torch.LongTensor(chunk['deck_id'].values)
    num_card = torch.LongTensor(chunk['card_count'].values)
    return TensorDataset(decks, cards, num_card)


# Handle missing cards by adding them to the dataset with placeholder values
"""def add_missing_cards_to_dataset(missing_card_ids, datasets):
    placeholder_deck_id = -1  # Special ID indicating no specific deck
    placeholder_count = 0     # Zero count for missing cards
    for card_id in missing_card_ids:
        card_tensor = torch.tensor([card_id], dtype=torch.long)
        deck_tensor = torch.tensor([placeholder_deck_id], dtype=torch.long)
        count_tensor = torch.tensor([placeholder_count], dtype=torch.float)
        datasets.append(TensorDataset(deck_tensor, card_tensor, count_tensor))
"""

def load_data(deck_id_map, card_id_map, batch_size=64):
    datasets = []
    for chunk in get_data_iterator():
        # Map the IDs to their new continuous indices
        chunk['deck_id'] = chunk['deck_id'].map(deck_id_map)
        chunk['card_id'] = chunk['card_id'].map(card_id_map)

        decks = torch.LongTensor(chunk['deck_id'].values)
        cards = torch.LongTensor(chunk['card_id'].values)
        counts = torch.LongTensor(chunk['card_count'].values) 
        
        dataset = TensorDataset(decks, cards, counts)
        datasets.append(dataset)

    full_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


