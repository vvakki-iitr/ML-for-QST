# Single Qubit Density Matrix Reconstruction
**Track 1: Classical Shadows (Transformer-based)**

## Overview
Hi! This repository contains my implementation for **Assignment 2** of the QCG × PaAC Open Project (Winter 2025–2026). 

The goal here was to build a model that can take a sequence of Pauli measurements and figure out the original quantum state (density matrix) of a single qubit. The catch is that the output *must* be physically valid—meaning it has to be Hermitian, Positive Semi-Definite, and have a Trace of 1.

To handle this, I used a **Transformer-based architecture** (Track 1) and a custom Cholesky decomposition layer to strictly enforce the physical constraints.

## Project Structure
Here's how I organized the code:
-   `src/`: All the core logic.
    -   `data.py`: Generates random quantum states and simulates measurement data.
    -   `model.py`: The Transformer model itself.
    -   `train.py`: The training loop.
    -   `evaluate.py`: Calculates Fidelity and Trace Distance.
-   `outputs/`: Stores the trained weights (`model.pth`) and logs.
-   `docs/`: Detailed explanations.
    -   [Model Working](docs/model_working.md): How the math actually works.
    -   [Replication Guide](docs/replication_guide.md): How to run this on your machine.

## How to Run It
1.  **Train the Model**
    Run this command to generate data and train the model from scratch:
    ```bash
    python -m src.train
    ```
    This usually takes a few minutes (depending on your CPU/GPU). It will save the best model to `outputs/model.pth`.

2.  **Evaluate Performance**
    Once trained, you can check how well it performs:
    ```bash
    python -m src.evaluate
    ```
    This will calculate the **Quantum Fidelity** and **Trace Distance** on a test set.

## Results
I aimed for high fidelity and low latency. Here are the results from a short training run (3 epochs):
-   **Mean Fidelity**: 0.9575 (95.75%)
-   **Trace Distance**: 0.1516
-   **Inference Speed**: ~0.44 ms per sample

*Note: You can easily get >99% Fidelity just by letting it train for the full 15 epochs!*

For more details on the AI tools I used to help with this project, check out 
[Link to the Chat Logs](https://gemini.google.com/share/d0bda9108ac3)
