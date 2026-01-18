import torch
import torch.nn as nn
import numpy as np

class DensityMatrixReconstructionModel(nn.Module):
    def __init__(self, embed_dim=32, num_heads=2, num_layers=2, max_len=512):
        super().__init__()
        # Input: Basis (3 types) x Outcome (2 types) = 6 unique tokens
        # 0: X-, 1: X+, 2: Y-, 3: Y+, 4: Z-, 5: Z+
        self.embedding = nn.Embedding(6, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: We need 4 real parameters for the Cholesky factor L of a 2x2 matrix
        self.fc = nn.Linear(embed_dim, 4)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 2) where 2 is (basis, outcome)
        # Convert (basis, outcome) to single token index
        # basis: 0,1,2. outcome: 0,1. 
        # Token = basis * 2 + outcome
        basis = x[:, :, 0]
        outcome = x[:, :, 1]
        tokens = basis * 2 + outcome
        
        # Embedding
        x = self.embedding(tokens) # (B, T, D)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        
        # Pooling (Global Average Pooling)
        x = x.mean(dim=1) # (B, D)
        
        # Predict Cholesky params
        params = self.fc(x) # (B, 4)
        
        # Construct Density Matrix
        rho = self.params_to_rho(params)
        return rho
    
    def params_to_rho(self, params):
        """
        The magic happens here! 
        We take the 4 real numbers predicted by the neural net and turn them 
        into a valid physical density matrix using Cholesky Decomposition.
        
        Formula: rho = (L * L^H) / Tr(L * L^H)
        where L is a lower triangular matrix.
        
        This guarantees:
        1. Hermitian (by construction L * L^H)
        2. Positive Semi-Definite (by construction)
        3. Unit Trace (by normalization)
        """
        # params: (B, 4) -> mapping to lower triangular L
        # l00 and l11 are the real diagonals
        # l10 is the complex off-diagonal (represented by real + imag parts)
        l00 = params[:, 0]
        l11 = params[:, 1]
        re_l10 = params[:, 2]
        im_l10 = params[:, 3]
        
        batch_size = params.size(0)
        
        # Construct the Lower Triangular Matrix L
        L = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=params.device)
        
        # Fill in the values
        L[:, 0, 0] = l00.to(torch.complex64)
        L[:, 1, 1] = l11.to(torch.complex64)
        L[:, 1, 0] = (re_l10 + 1j * im_l10).to(torch.complex64)
        
        # Calculate unnormalized rho = L @ L_dagger
        L_dagger = torch.conj(L).transpose(1, 2)
        rho_unnorm = torch.bmm(L, L_dagger)
        
        # Normalize by the trace so sum of diagonals is 1
        trace = torch.real(torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(dim=-1))
        
        # Small epsilon to avoid division by zero (safety first!)
        trace = trace.view(-1, 1, 1) + 1e-8
        
        rho = rho_unnorm / trace
        return rho

if __name__ == "__main__":
    # Smoke test
    model = DensityMatrixReconstructionModel()
    dummy_input = torch.zeros(2, 10, 2, dtype=torch.long) # B=2, T=10
    rho = model(dummy_input)
    print("Output shape:", rho.shape)
    print("Trace:", torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)))
