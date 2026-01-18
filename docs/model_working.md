# How the Model Works

## The Problem
We need to reconstruct a quantum state (density matrix $\rho$) from a bunch of noisy measurements.
A single qubit density matrix looks like this:
$$ \rho = \begin{pmatrix} a & b \\ b^* & 1-a \end{pmatrix} $$
where $a$ is real and $b$ is complex. But it's not just any matrix—it has to be **Positive Semi-Definite** (eigenvalues $\ge 0$). If the model just outputs random numbers, we'll get a garbage state that violates physics.

## The Solution: Cholesky Decomposition
Instead of predicting $\rho$ directly, I predict a lower triangular matrix $L$:
$$ L = \begin{pmatrix} l_{00} & 0 \\ l_{10} & l_{11} \end{pmatrix} $$
Then I construct $\rho$ using this formula:
$$ \rho_{raw} = L \times L^\dagger $$
This trick is cool because $L L^\dagger$ is *always* Positive Semi-Definite by definition. Finally, I divide by the trace to make sure the probabilities sum to 1.

## The Neural Network (Track 1)
I went with a **Transformer-based approach** (Classical Shadows style).
1.  **Input**: A sequence of measurement outcomes (Basis + Result).
    -   Example: "Measured X, got +1", "Measured Z, got -1"...
2.  **Embedding**: These pairs are turned into vectors.
3.  **Transformer Encoder**: The model looks at the whole sequence at once to find patterns.
4.  **Output**: It spits out the 4 numbers needed to build the matrix $L$.

It's basically treating quantum state tomography like a language translation task:
*Sequence of Measurements* $\rightarrow$ *Quantum State*
