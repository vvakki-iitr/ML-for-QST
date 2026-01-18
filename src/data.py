import numpy as np
import torch
from torch.utils.data import Dataset

def random_density_matrix(dim=2):
    """
    Generates a random single-qubit density matrix (2x2) using the Haar measure.
    Scale matters! We use the Ginibre ensemble method here.
    Basically, we create a random complex matrix G and then do G * G^H.
    This guarantees the matrix is Positive Semi-Definite (PSD).
    Then we normalize by the trace so it sums to 1 (valid probability).
    """
    # Ginibre ensemble: G is a complex matrix with N(0,1) entries
    G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = G @ G.conj().T  # This makes it Hermitian and PSD
    rho /= np.trace(rho)  # Normalize to ensure Trace = 1
    return rho

def get_pauli_matrices():
    """
    Returns the standard Pauli matrices X, Y, Z.
    These are the observables we measure.
    """
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {'X': X, 'Y': Y, 'Z': Z}

def simulate_measurement(rho, num_measurements):
    """
    Simulates random Pauli measurements on the density matrix rho.
    Returns a sequence of (basis_index, outcome).
    Basis: 0->X, 1->Y, 2->Z
    Outcome: +1 or -1 (mapped to 0 and 1 for embedding)
    """
    paulis = get_pauli_matrices()
    basis_map = {0: 'X', 1: 'Y', 2: 'Z'}
    
    measurements = []
    
    for _ in range(num_measurements):
        # 1. Select random basis
        basis_idx = np.random.choice([0, 1, 2])
        basis_str = basis_map[basis_idx]
        M = paulis[basis_str]
        
        # 2. Born Rule: P(outcome) = Tr(Pi * rho)
        # Projectors for +1 and -1 eigenvalues
        # For single qubit Pauli, eigenvalues are +1, -1.
        # Projector P+ = (I + M)/2, P- = (I - M)/2
        eigvals, eigvecs = np.linalg.eigh(M)
        # eigvals are sorted ascending usually, so [-1, 1]
        
        # Calculate probabilities
        probs = []
        outcomes = []
        for val, vec in zip(eigvals, eigvecs.T): # eigh returns cols as vecs
            # Projector = |v><v|
            vec = vec.reshape(-1, 1)
            P = vec @ vec.conj().T
            prob = np.real(np.trace(rho @ P))
            probs.append(max(0.0, min(1.0, prob))) # Clip for numerical stability
            outcomes.append(val)
            
        # Normalize probs just in case
        probs = np.array(probs)
        probs /= probs.sum()
        
        # 3. Sample outcome
        outcome_val = np.random.choice(outcomes, p=probs)
        
        # Map outcome: -1 -> 0, +1 -> 1 (for neural net embedding)
        outcome_idx = 0 if outcome_val < 0 else 1
        
        measurements.append((basis_idx, outcome_idx))
        
    return np.array(measurements)

class QuantumDataset(Dataset):
    def __init__(self, num_samples=1000, num_measurements=100):
        self.num_samples = num_samples
        self.num_measurements = num_measurements
        self.data = []
        
        print(f"Generating dataset with {num_samples} samples...")
        for _ in range(num_samples):
            rho = random_density_matrix()
            meas = simulate_measurement(rho, num_measurements)
            # Flatten density matrix to real parts for regression target
            # rho is 2x2 complex. 
            # We can represent it as 4 real numbers (Re(rho00), Re(rho01), Im(rho01), Re(rho11))
            # Or just store the full complex matrix and handle it in loss
            self.data.append({
                'measurements': torch.tensor(meas, dtype=torch.long), # (N, 2)
                'density_matrix': torch.tensor(rho, dtype=torch.complex64)
            })
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # Test generation
    ds = QuantumDataset(num_samples=5, num_measurements=10)
    print("Sample 0 measurements shape:", ds[0]['measurements'].shape)
    print("Sample 0 rho shape:", ds[0]['density_matrix'].shape)
    print("Sample 0 rho trace:", torch.trace(ds[0]['density_matrix']))
