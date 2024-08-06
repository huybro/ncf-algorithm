# Fetch all unique card IDs from the 'cards' table and 'cards_in_deck' table
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:cghuy2004@localhost:5432/go_rec_sys')

def create_id_mappings():
    with engine.connect() as connection:
        # Get unique deck IDs
        deck_query = text("SELECT DISTINCT deck_id FROM cards_in_deck ORDER BY deck_id")
        deck_result = connection.execute(deck_query)
        unique_deck_ids = [row[0] for row in deck_result]

        # Get unique card IDs
        card_query = text("SELECT DISTINCT id FROM cards ORDER BY id")
        card_result = connection.execute(card_query)
        unique_card_ids = [row[0] for row in card_result]

    # Create mappings
    deck_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_deck_ids)}
    card_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_card_ids)}
    index_to_deck_id = {new_id: old_id for old_id, new_id in deck_id_map.items()}

    return deck_id_map, card_id_map, len(deck_id_map), len(card_id_map),index_to_deck_id