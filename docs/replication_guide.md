# Replication Guide

Here is how you can run my code and verify the results yourself.

## Setup
You'll need Python installed. I used:
-   `numpy` (for math)
-   `torch` (for the neural net)
-   `scipy` (for calculating fidelity)

You can install them with:
```bash
pip install torch numpy scipy
```

## Running the Code
I've set it up so you can run almost everything with one-liners.

### 1. Train the Model
To train the model from scratch on my generated dataset:
```bash
python -m src.train
```
This script will:
1.  Generate random quantum states.
2.  Simulate measurements.
3.  Train the Transformer.
4.  Save the model to `outputs/model.pth`.

### 2. Evaluate
To see the Fidelity and Trace Distance scores:
```bash
python -m src.evaluate
```
This will load the model you just trained and test it on new, unseen data. It prints the results right in the terminal.

## What to Expect
If everything goes well, you should see:
-   **Fidelity**: > 99% (High accuracy)
-   **Trace Distance**: < 0.05 (Low error)
-   **Inference Time**: Fast! (< 1ms per state)
