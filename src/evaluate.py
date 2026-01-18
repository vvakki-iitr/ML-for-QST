import torch
import numpy as np
from scipy.linalg import sqrtm
from src.data import QuantumDataset
from src.model import DensityMatrixReconstructionModel
import time

def compute_fidelity(rho_pred, rho_true):
    """
    Computes Fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
    """
    # Convert to numpy for scipy.linalg
    rho = rho_pred.detach().cpu().numpy()
    sigma = rho_true.detach().cpu().numpy()
    
    fidelities = []
    for r, s in zip(rho, sigma):
        # sqrtm can return complex result even if matrix is real, so we keep it complex
        # But for density matrices it should be fine.
        sqrt_r = sqrtm(r)
        
        # Product sqrt(rho) * sigma * sqrt(rho)
        temp = sqrt_r @ s @ sqrt_r
        
        # sqrt of that
        temp_sqrt = sqrtm(temp)
        
        # Trace
        tr = np.trace(temp_sqrt)
        
        # Real part squared
        fid = np.real(tr) ** 2
        fidelities.append(np.clip(fid, 0.0, 1.0))
        
    return np.mean(fidelities)

def compute_trace_distance(rho_pred, rho_true):
    """
    Computes Trace Distance T(rho, sigma) = 0.5 * Tr(|rho - sigma|)
    """
    rho = rho_pred.detach().cpu().numpy()
    sigma = rho_true.detach().cpu().numpy()
    
    distances = []
    for r, s in zip(rho, sigma):
        diff = r - s
        # Eigenvalues of difference
        eigvals = np.linalg.eigvals(diff)
        # Sum of absolute eigenvalues
        dist = 0.5 * np.sum(np.abs(eigvals))
        distances.append(dist)
        
    return np.mean(distances)

def evaluate_model(model_path="outputs/model.pth", num_samples=1000):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = DensityMatrixReconstructionModel(max_len=100).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print("Model file not found! Please train first.")
        return

    model.eval()
    
    # Data
    dataset = QuantumDataset(num_samples=num_samples, num_measurements=100)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_fidelities = []
    all_distances = []
    latencies = []
    
    print(f"Evaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for batch in loader:
            measurements = batch['measurements'].to(DEVICE)
            rho_target = batch['density_matrix'].to(DEVICE)
            
            # Latency Measurement
            start_time = time.time()
            rho_pred = model(measurements)
            end_time = time.time()
            
            # Per sample latency (approximate for batch)
            batch_latency = (end_time - start_time) / measurements.size(0)
            latencies.append(batch_latency)
            
            # Metrics
            fid = compute_fidelity(rho_pred, rho_target)
            dist = compute_trace_distance(rho_pred, rho_target)
            
            all_fidelities.append(fid)
            all_distances.append(dist)
            
    mean_fidelity = np.mean(all_fidelities)
    mean_trace_distance = np.mean(all_distances)
    avg_latency_ms = np.mean(latencies) * 1000
    
    print("\n=== Evaluation Results ===")
    print(f"Mean Fidelity: {mean_fidelity:.4f}")
    print(f"Mean Trace Distance: {mean_trace_distance:.4f}")
    print(f"Inference Latency: {avg_latency_ms:.4f} ms/sample")
    
    # Save results to a file for report
    with open("outputs/evaluation_metrics.txt", "w") as f:
        f.write(f"Mean Fidelity: {mean_fidelity:.4f}\n")
        f.write(f"Mean Trace Distance: {mean_trace_distance:.4f}\n")
        f.write(f"Inference Latency: {avg_latency_ms:.4f} ms/sample\n")

import os
if __name__ == "__main__":
    evaluate_model()
