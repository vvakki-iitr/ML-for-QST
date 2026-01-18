import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data import QuantumDataset
from src.model import DensityMatrixReconstructionModel
import os
import time

def train_model():
    # Hyperparameters
    NUM_SAMPLES = 5000
    NUM_MEASUREMENTS = 100 # Sequence length
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Data
    dataset = QuantumDataset(num_samples=NUM_SAMPLES, num_measurements=NUM_MEASUREMENTS)
    train_size = int(0.8 * NUM_SAMPLES)
    val_size = NUM_SAMPLES - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = DensityMatrixReconstructionModel(max_len=NUM_MEASUREMENTS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Loss Function: MSE on real and imaginary parts
    # Fidelity is hard to optimize directly for mixed states due to sqrtm
    def matrix_mse_loss(rho_pred, rho_true):
        loss_real = nn.MSELoss()(rho_pred.real, rho_true.real)
        loss_imag = nn.MSELoss()(rho_pred.imag, rho_true.imag)
        return loss_real + loss_imag
        
    # Training Loop
    print("Starting training...")
    os.makedirs("outputs", exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            measurements = batch['measurements'].to(DEVICE) # (B, T, 2)
            rho_target = batch['density_matrix'].to(DEVICE) # (B, 2, 2)
            
            optimizer.zero_grad()
            rho_pred = model(measurements)
            
            loss = matrix_mse_loss(rho_pred, rho_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                measurements = batch['measurements'].to(DEVICE)
                rho_target = batch['density_matrix'].to(DEVICE)
                
                rho_pred = model(measurements)
                loss = matrix_mse_loss(rho_pred, rho_target)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
    # Save Model
    torch.save(model.state_dict(), "outputs/model.pth")
    print("Model saved to outputs/model.pth")

if __name__ == "__main__":
    train_model()
