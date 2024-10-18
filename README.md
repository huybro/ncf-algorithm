
# Card Recommendation System

This project implements a recommendation system to suggest cards based on the user's existing cards in a Yu-Gi-Oh deck. The system uses a Neural Collaborative Filtering (NCF) model to predict the relevance of cards and recommends the top cards that are most likely to complement the user's current deck.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to develop a system that can recommend Yu-Gi-Oh cards to users based on the cards they already own. The system is built using a collaborative filtering approach with neural networks, leveraging both user and card embeddings to predict the relevance of cards for a given deck.

## Features

- **Collaborative Filtering**: Uses the Neural Collaborative Filtering (NCF) approach to learn interactions between decks and cards.
- **Adaptive Negative Sampling**: Ensures the model learns more effectively by using negative samples that are hard to distinguish from positive samples.
- **Content-Based Recommendations**: Incorporates card similarity based on attributes like type and archetype.
- **Flexible Configuration**: Allows users to customize the number of recommendations, embedding size, and model architecture.

## Model Architecture

The model is based on the Neural Collaborative Filtering (NCF) approach. It uses embedding layers for decks and cards, followed by a series of fully connected layers that learn interactions between decks and cards. The model predicts the relevance of each card to the deck, which is then used to rank and recommend cards.

### Key Parameters:
- Embedding size: 128
- Fully connected layers: [128, 64, 32]
- Learning rate: 0.0001
- Batch size: 64
- Epochs: 20

## Dataset

The dataset contains user-deck interactions, with each deck containing a set of cards. The data is processed to generate:
- **deck_id_map**: A mapping from deck IDs to indices.
- **card_id_map**: A mapping from card IDs to indices.
- **user_card_list**: A list of cards the user currently owns, used as input for the recommendation system.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- scikit-learn

### Clone the repository
```bash
git clone https://github.com/your-username/ncf-algorithm.git
cd ncf-algorithm
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess Data**: Make sure your dataset is formatted and mapped correctly using the `create_id_mappings()` function.
2. **Load Pre-Trained Model**: You can load a pre-trained model using the following code:
    ```python
    model_dir = 'saved_models'
    model_filename = 'ncf_model_epoch_20.pth' 
    model_path = os.path.join(model_dir, model_filename)
    model = NCF(num_decks, num_cards, embedding_size, mlp_layers)
    model.load_state_dict(torch.load(model_path))
    ```

3. **Get Recommendations**:
    ```python
    user_card_list = [101206024, 72270339, 18144507, 2857636, 89023486, 98567237]  # Example user cards
    recommended_cards = recommend_cards(user_card_list, card_id_map, model, num_recommendations=10)
    print("Recommended Cards:", recommended_cards)
    ```

4. **Evaluate Model**: You can evaluate the model using metrics like accuracy, precision, recall, and F1 score:
    ```python
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    ```

## Model Training

To train the model from scratch, run the following script:
```bash
python train_model.py
```

You can configure the training parameters such as embedding size, number of epochs, and learning rate in the `train_model.py` script.

## Evaluation

The system is evaluated using common metrics like accuracy, precision, recall, and F1 score, which are calculated on a test dataset.

```python
accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
```

## Results

The model achieves the following results on the test set:
- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1 Score**: XX%

These results demonstrate the model's ability to recommend relevant cards based on the user's existing deck.

## Contributing

Contributions are welcome! If you'd like to improve the system or fix any issues, feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
