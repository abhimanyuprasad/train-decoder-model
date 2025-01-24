# Transformer Project

This project implements a decoder-only transformer model using PyTorch. The model is designed to process text data and can be trained on a dataset provided in a text file. The project is structured to be run in a Google Colab environment, but it can also be adapted for local execution.

## Features

- Decoder-only transformer model
- Configurable model parameters
- Training with character-level tokenization
- Model saving and loading for continued training
- Parameter count display

## Requirements

- Python 3.6+
- PyTorch
- tqdm

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd transformer_project
   ```

2. **Install dependencies**:
   Ensure you have the required Python packages installed. You can use pip to install them:
   ```bash
   pip install torch tqdm
   ```

3. **Prepare your data**:
   Place your text data in a file named `input.txt` in the project directory.

## Running the Project

1. **In Google Colab**:
   - Upload the project files to your Google Drive.
   - Open `main.py` in a Colab notebook.
   - Run the cells to execute the script.

2. **Locally**:
   - Ensure you have the necessary environment set up.
   - Run the script using Python:
     ```bash
     python main.py
     ```

## Configuration

The model configuration can be adjusted in `config.py`. Key parameters include:
- `vocab_size`: Size of the vocabulary.
- `max_seq_len`: Maximum sequence length.
- `dim`: Embedding dimension.
- `num_layers`: Number of transformer layers.
- `num_heads`: Number of attention heads.
- `dropout`: Dropout rate.

## Model Training

The model is trained using the data from `input.txt`. The training process includes:
- Loading the dataset and creating batches.
- Training the model for a specified number of epochs.
- Saving the model state to `decoder_model.pth`.
