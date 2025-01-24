import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import Config
from model import DecoderOnlyTransformer
from tqdm import tqdm
import os

class TextDataset(Dataset):
    def __init__(self, text, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self.tokenize(text)

    def tokenize(self, text):
        # Simple character-level tokenization
        char_to_idx = {ch: i for i, ch in enumerate(sorted(set(text)))}
        idx_to_char = {i: ch for ch, i in char_to_idx.items()}
        self.vocab_size = len(char_to_idx)
        
        # Convert text to indices
        indices = [char_to_idx[ch] for ch in text if ch in char_to_idx]
        
        # Create sequences
        sequences = []
        for i in range(0, len(indices) - self.seq_len, self.seq_len):
            seq = indices[i:i + self.seq_len]
            sequences.append(seq)
        
        return torch.tensor(sequences, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")

if __name__ == '__main__':
    # Adjusted configuration for ~124M parameters
    config = Config(
        vocab_size=50257,  # This will be updated based on the dataset
        max_seq_len=128,
        dim=1024,  # Increase dimension
        num_layers=12,  # Number of layers
        num_heads=16,  # Increase number of heads
        dropout=0.1
    )
    
    # Read text data from input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset
    dataset = TextDataset(text, config.max_seq_len, config.vocab_size)
    config.vocab_size = dataset.vocab_size  # Update vocab size based on dataset
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = DecoderOnlyTransformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    train_model(model, dataloader, optimizer, criterion, device, epochs=1) 