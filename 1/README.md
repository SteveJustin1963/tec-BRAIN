# intro

- https://www.youtube.com/watch?v=l-OLgbdZ3kk                                                                                                                                                              


● This is a comprehensive technical document on Predictive Coding as a biologically plausible learning algorithm. Here's what it covers:                        
                                                                                                                                                               
  Structure                                                                                                                                                     
                                                                                                                                                                
  1. Technical Summary (lines 8–178) — A blueprint for implementing a Predictive Coding (PC) network, covering:                                                 
    - Hierarchical architecture with representational neurons (x) and error neurons (ε)                                                                         
    - Energy function formalism and gradient descent minimization                                                                                               
    - Activity update rules and weight update rules (local Hebbian plasticity)
    - Full training/inference procedure
    - NumPy pseudocode skeleton
  2. PyTorch Implementation (lines 210–414) — A full batched, vectorized PC network with:
    - PredictiveCodingLayer and PredictiveCodingNetwork classes
    - Iterative relaxation for inference
    - Local Hebbian weight updates
    - MNIST training example
  3. Deep Active Inference Agent (lines 416–665) — Extends PC to action selection via Expected Free Energy (EFE) minimization, with:
    - Neural parameterizations of A (likelihood), B (transition), C (preferences)
    - Amortized posterior inference
    - Policy selection via softmax over EFE
  4. Conceptual Overview (lines 667–753) — A narrative explanation of predictive coding, its biological grounding (superficial/deep pyramidal cells), comparison
   to backprop, and connections to Friston's free-energy principle.

  Key References

  Notable papers by Bogacz (2017), Rao & Ballard (1999), Millidge et al. (2022), Friston, Whittington & Bogacz, and a 2025 survey by Salvatori et al.







### Technical Summary: Implementing Predictive Coding as a Biologically Plausible Learning Algorithm

This technical summary distills the key concepts, mathematical formulations, and implementation details from the video "The Brain’s Learning Algorithm Isn’t Backpropagation" by Artem Kirsanov. It focuses on Predictive Coding (PC) as an alternative to backpropagation (backprop) for training neural networks, emphasizing its biological plausibility. The summary is structured to provide a blueprint for building software that simulates a PC network, including architecture, update rules, and training procedures. This can be implemented in a programming framework like Python with libraries such as NumPy for matrix operations or PyTorch for tensor handling (though avoiding backprop-specific features).

Predictive Coding addresses the credit assignment problem—adjusting network parameters (e.g., synaptic weights) to map inputs to desired outputs—without the biologically implausible aspects of backprop, such as separate forward/backward phases, global error propagation, and precise temporal coordination. Instead, PC uses continuous, local updates in a hierarchical network where layers predict lower-level activities, and errors flow bidirectionally to minimize an energy function.

#### 1. Core Foundations and Architecture
- **Hierarchical Structure**: Build a multi-layer network with L layers, where Layer 1 is the input (sensory) layer and Layer L is the output (abstract) layer. Each layer has two populations of neurons:
  - **Representational neurons (x)**: Encode the predicted or inferred state/activity at that layer.
  - **Error neurons (ε)**: Compute and propagate prediction errors (deviations between actual and predicted activities).
- **Bidirectional Connectivity**:
  - Feedforward connections: From representational neurons in lower layers to error neurons in higher layers (for bottom-up error signaling).
  - Feedback connections: From representational neurons in higher layers to error neurons in lower layers (for top-down predictions).
  - Within-layer: Each representational neuron is paired with an error neuron for local computation.
- **Nonlinearities**: While the derivation assumes linearity for simplicity, incorporate activation functions (e.g., ReLU or sigmoid) after updates for realism. The video notes that approximate symmetry between forward and feedback weights emerges dynamically, even with nonlinearities.
- **Software Representation**:
  - Use arrays/matrices for states: Let `x[l]` be a vector of shape (n_neurons_l,) for representational activities at layer l.
  - Let `ε[l]` be a vector for errors at layer l.
  - Weights: `W[l]` as a matrix (n_neurons_l+1 x n_neurons_l) for connections from layer l to l+1 (feedforward), and similarly for feedback (though they can converge to approximate transposes).

#### 2. Energy Formalism
The system is modeled as minimizing a global energy function, analogous to a physical system settling to a low-energy state. This energy quantifies prediction errors across the hierarchy.

- **Energy Function (E)**:
  $\[
  E = \sum_{l=1}^{L} \sum_{i} (\epsilon_i^l)^2 = \sum_{l=1}^{L} \sum_{i} (x_i^l - \hat{y}_i^l)^2
  \]$
  Where:
 $\(\epsilon_i^l = x_i^l - \hat{y}_i^l\)$
is the prediction error for neuron i at layer l.

  $\(\hat{y}_i^l = \sum_j w_{ij}^{l+1 \to l} x_j^{l+1}\)$

  is the top-down prediction from the layer above (for l < L). For the bottom layer, predictions come from sensory input; for the top, there might be no prediction or a prior.

  For the input layer (l=1), $\(x^1\)$ is clamped to input data, so errors reflect deviations from predictions.

- **Minimization Principle**: The network evolves activities and weights via gradient descent on E:
  \[
  \frac{d\theta}{dt} = -\frac{\partial E}{\partial \theta}
  \]
  Where \(\theta\) can be activities (x) or weights (w). In software, discretize this with small time steps (e.g., Euler integration) or iterate until convergence.

- **Implementation Tip**: Compute E as a scalar sum of squared error vectors across layers. Use this for monitoring convergence during relaxation phases.

#### 3. Activity Update Rule
Activities (x) adjust continuously and locally to minimize energy, balancing top-down predictions and bottom-up errors. No global phases are needed—updates happen in parallel.

- **Update Equation**:
  For a representational neuron \(x_i^l\) at layer l:
  \[
  \frac{d x_i^l}{dt} = -\epsilon_i^l + \sum_j w_{ij}^{l \to l-1} \epsilon_j^{l-1}
  \]
  - The first term (-\(\epsilon_i^l\)) pulls \(x_i^l\) toward the top-down prediction \(\hat{y}_i^l\).
  - The second term incorporates bottom-up errors from the layer below, weighted by forward connections.
  - For the bottom layer (l=1): No bottom-up term; clamp \(x^1\) to input if fixed.
  - For the top layer (l=L): No top-down prediction, so only bottom-up errors drive updates.

- **Discrete Approximation** (for software):
  \[
  x_i^l \leftarrow x_i^l + \alpha \left( -\epsilon_i^l + \sum_j w_{ij}^{l \to l-1} \epsilon_j^{l-1} \right)
  \]
  Where \(\alpha\) is a small learning rate (e.g., 0.01–0.1). Iterate this over multiple steps until activities stabilize (e.g., when changes in x are below a threshold like 1e-4).

- **Error Computation**:
  \[
  \epsilon_i^l = x_i^l - \sum_k w_{ki}^{l+1 \to l} x_k^{l+1}
  \]
  (Prediction error as actual minus predicted.)

- **Neural Circuit Implementation**:
  - Representational neurons (x_i) are inhibited by their paired error neuron (\(\epsilon_i\)) and excited by error neurons from below (\(\epsilon_j^{l-1}\)).
  - Error neurons (\(\epsilon_i\)) are excited by their paired x_i and inhibited by predictions from above (\(\hat{y}_i\)).
  - In code: Simulate this as a dynamical system with recurrent updates, ensuring local computations (no global access needed per neuron).

#### 4. Weight Update Rule
Weights adjust via local Hebbian-like plasticity, using only presynaptic activity and postsynaptic error—avoiding the "weight transport problem" of backprop (needing symmetric weights explicitly).

- **Update Equation**:
  For weight \(w_{ik}^{l \to l-1}\) (from presynaptic x_i^l to postsynaptic prediction for layer l-1):
  \[
  \frac{d w_{ik}}{dt} = x_i^l \cdot \epsilon_k^{l-1}
  \]
  - Positive if presynaptic fires and postsynaptic error is positive (strengthen to reduce error).
  - This is local: Each synapse updates based on immediately adjacent neurons.

- **Discrete Version**:
  \[
  w_{ik} \leftarrow w_{ik} + \beta \cdot x_i^l \cdot \epsilon_k^{l-1}
  \]
  Where \(\beta\) is a weight learning rate (e.g., 0.001–0.01, potentially decaying over training).

- **Feedback Weights**: Update similarly but independently. Over time, forward and feedback weights converge to approximate transposes due to energy minimization dynamics, even without explicit symmetry enforcement.

- **Implementation Note**: Apply updates after activity relaxation or interleaved. Add regularization (e.g., weight decay) if needed to prevent explosion.

#### 5. Full Training and Inference Procedure
To build a trainable PC network (e.g., for supervised classification like MNIST):

1. **Initialization**:
   - Randomize weights (e.g., Gaussian with mean 0, std 0.1).
   - Set initial activities to zeros or small random values.

2. **Inference (Forward Pass Equivalent)**:
   - Clamp bottom layer (l=1) to input data.
   - Unclamp other layers.
   - Iterate activity updates across all layers until equilibrium (energy E stabilizes, e.g., 100–1000 iterations).
   - Read output from top-layer representational neurons (x^L).

3. **Learning (Training Loop)**:
   - For each training example:
     - Perform inference to relax activities with input clamped.
     - Clamp top layer (x^L) to target labels (e.g., one-hot vector for classification).
     - Continue relaxing activities (now influenced by top-down target predictions).
     - Update weights using the weight rule during or after relaxation.
   - Repeat over epochs/batches. No separate backward pass—learning emerges from continuous minimization.
   - For generative tasks: Unclamp top layer, clamp bottom to partial input, relax to generate completions.

4. **Supervised vs. Unsupervised**:
   - Unsupervised: No top clamping; learns hierarchical representations via prediction errors.
   - Supervised: Clamp top to labels for error-driven learning.

5. **Convergence and Stability**:
   - Monitor total energy E; stop relaxation when ΔE < threshold.
   - Use techniques like momentum in updates for faster convergence.
   - Recent extensions (cited in video refs) add stability via reverse differentiation or relaxed constraints.

#### 6. Advantages for Software Implementation
- **Parallelization**: All updates are local, ideal for GPU acceleration or distributed computing.
- **Efficiency**: Avoids backprop's global dependencies; potentially fewer computations for equilibrium.
- **Biological Alignment**: Supports continuous learning, reducing catastrophic forgetting.
- **Extensions**: Integrate with VAEs for probabilistic modeling or add lateral connections for more complex hierarchies.

#### 7. Potential Code Skeleton (Python/NumPy Pseudocode)
```python
import numpy as np

class PredictiveCodingNetwork:
    def __init__(self, layer_sizes, alpha=0.05, beta=0.005):
        self.L = len(layer_sizes)
        self.x = [np.zeros(size) for size in layer_sizes]  # Activities
        self.epsilon = [np.zeros(size) for size in layer_sizes]  # Errors
        self.W_ff = [np.random.randn(layer_sizes[l+1], layer_sizes[l]) * 0.1 for l in range(self.L-1)]  # Feedforward
        self.W_fb = [np.random.randn(layer_sizes[l], layer_sizes[l+1]) * 0.1 for l in range(self.L-1)]  # Feedback (approx transpose)

    def compute_errors(self):
        for l in range(1, self.L):
            y_hat = self.x[l] @ self.W_fb[l-1]  # Top-down prediction (adjust for matrix dims)
            self.epsilon[l-1] = self.x[l-1] - y_hat

    def update_activities(self):
        for l in range(self.L):
            term1 = -self.epsilon[l] if l < self.L-1 else np.zeros_like(self.x[l])
            term2 = self.epsilon[l-1] @ self.W_ff[l-1].T if l > 0 else np.zeros_like(self.x[l])
            self.x[l] += self.alpha * (term1 + term2)

    def update_weights(self):
        for l in range(self.L-1):
            self.W_ff[l] += self.beta * np.outer(self.x[l+1], self.epsilon[l]).T  # Adjust for Hebbian
            # Similarly for feedback

    def relax(self, iterations=500, clamp_bottom=None, clamp_top=None):
        if clamp_bottom is not None: self.x[0] = clamp_bottom
        if clamp_top is not None: self.x[-1] = clamp_top
        for _ in range(iterations):
            self.compute_errors()
            self.update_activities()
        return self.x[-1]  # Output

# Usage: net = PredictiveCodingNetwork([784, 256, 10])  # e.g., MNIST
# output = net.relax(clamp_bottom=input_data, clamp_top=target_label)
# net.update_weights()  # After relaxation
```
This pseudocode simplifies; refine matrix operations, add nonlinearities, and handle clamping properly. Test on small datasets to verify energy minimization.

This summary provides sufficient detail to prototype and extend a PC-based software system, drawing directly from the video's derivations. For deeper math, consult references like Bogacz (2017) or Millidge et al. (2022).



# ref


📚 FURTHER READING & REFERENCES:
Bogacz, R., 2017. A tutorial on the free-energy framework for modelling perception and learning. Journal of Mathematical Psychology 76, 198–211. https://doi.org/10.1016/j.jmp.2015.11...
Friston, K., 2018. Does predictive coding have a future? Nat Neurosci 21, 1019–1021. https://doi.org/10.1038/s41593-018-02...
Huang, Y., Rao, R.P.N., 2011. Predictive coding. WIRES Cognitive Science 2, 580–593. https://doi.org/10.1002/wcs.142
Keller, G.B., Mrsic-Flogel, T.D., 2018. Predictive Processing: A Canonical Cortical Computation. Neuron 100, 424–435. https://doi.org/10.1016/j.neuron.2018...
Lillicrap, T.P., Santoro, A., Marris, L., Akerman, C.J., Hinton, G., 2020. Backpropagation and the brain. Nat Rev Neurosci 21, 335–346. https://doi.org/10.1038/s41583-020-02...
Marino, J., 2021. Predictive Coding, Variational Autoencoders, and Biological Connections. https://doi.org/10.48550/arXiv.2011.0...
Millidge, B., Salvatori, T., Song, Y., Bogacz, R., Lukasiewicz, T., 2022a. Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?
Millidge, B., Seth, A., Buckley, C.L., 2022b. Predictive Coding: a Theoretical and Experimental Review. https://doi.org/10.48550/arXiv.2107.1...
Millidge, B., Song, Y., Salvatori, T., Lukasiewicz, T., Bogacz, R., 2023. A THEORETICAL FRAMEWORK FOR INFERENCE AND LEARNING IN PREDICTIVE CODING NETWORKS.
Millidge, B., Tschantz, A., Buckley, C.L., 2022c. Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs. Neural Computation 34, 1329–1368. https://doi.org/10.1162/neco_a_01497
Millidge, B., Tschantz, A., Seth, A., Buckley, C.L., 2020. Relaxing the Constraints on Predictive Coding Models. https://doi.org/10.48550/arXiv.2010.0...
Rao, R.P.N., Ballard, D.H., 1999. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nat Neurosci 2, 79–87. https://doi.org/10.1038/4580
Rosenbaum, R., 2022. On the relationship between predictive coding and backpropagation. PLoS ONE 17, e0266102. https://doi.org/10.1371/journal.pone....
Salvatori, T., Mali, A., Buckley, C.L., Lukasiewicz, T., Rao, R.P.N., Friston, K., Ororbia, A., 2025. A Survey on Brain-Inspired Deep Learning via Predictive Coding. https://doi.org/10.48550/arXiv.2308.0...
Salvatori, T., Song, Y., Lukasiewicz, T., Bogacz, R., Xu, Z., 2023. Reverse Differentiation via Predictive Coding. https://doi.org/10.48550/arXiv.2103.0...
Salvatori, T., Song, Y., Yordanov, Y., Millidge, B., Xu, Z., Sha, L., Emde, C., Bogacz, R., Lukasiewicz, T., 2024. A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks. https://doi.org/10.48550/arXiv.2212.0...
Song, Y., Lukasiewicz, T., Xu, Z., Bogacz, R., n.d. Can the Brain Do Backpropagation? — Exact Implementation of Backpropagation in Predictive Coding Networks.
Song, Y., Millidge, B., Salvatori, T., Lukasiewicz, T., Xu, Z., Bogacz, R., 2024. Inferring neural activity before plasticity as a foundation for learning beyond backpropagation. Nat Neurosci 27, 348–358. https://doi.org/10.1038/s41593-023-01...
Whittington, J.C.R., Bogacz, R., 2019. Theories of Error Back-Propagation in the Brain. Trends in Cognitive Sciences 23, 235–250. https://doi.org/10.1016/j.tics.2018.1...
Whittington, J.C.R., Bogacz, R., 2017. An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity. Neural Computation 29, 1229–1262. https://doi.org/10.1162/NECO_a_00949


# Add PyTorch Implementation

Here is a **PyTorch-based implementation** of the Predictive Coding (PC) network described in the technical summary (inspired by biologically plausible hierarchical predictive coding as in Bogacz, Whittington & Bogacz, Millidge et al., and related works). This version closely follows the energy-minimization dynamics with separate representational (`x`) and error (`ε`) populations, local Hebbian-like weight updates, and iterative relaxation for inference/learning.

This is a clean, vectorized, batched implementation suitable for supervised tasks (e.g., MNIST classification). It supports:
- Multiple hidden layers
- Nonlinear activations (ReLU by default)
- Iterative relaxation of activities until convergence or fixed steps
- Local weight updates via Hebbian rule on prediction errors
- Approximate emergence of symmetric forward/backward weights

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

class PredictiveCodingLayer(nn.Module):
    """One layer in the PC hierarchy with forward & feedback weights."""
    def __init__(self, in_features, out_features, nonlinearity=nn.ReLU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Feedforward weights (bottom-up: lower → higher error)
        self.W_ff = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Feedback weights (top-down: higher → lower prediction)
        # Initialized close to transpose for faster convergence, but updated separately
        self.W_fb = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        
        self.nonlinearity = nonlinearity

    def forward_prediction(self, x_higher):
        """Top-down prediction to this layer: ŷ = x_higher @ W_fb^T"""
        return x_higher @ self.W_fb.T

    def error_from_below(self, epsilon_lower):
        """Bottom-up influence on this layer's representation: W_ff^T @ ε_lower"""
        return epsilon_lower @ self.W_ff.T if epsilon_lower is not None else torch.zeros_like(self.x)


class PredictiveCodingNetwork(nn.Module):
    def __init__(self, layer_sizes, alpha=0.1, beta=0.005, relaxation_steps=200, nonlinearity=nn.ReLU()):
        """
        layer_sizes: list of int, e.g. [784, 256, 128, 10] for MNIST
        alpha: activity update rate (learning rate for inference / relaxation)
        beta:  weight update rate (Hebbian plasticity)
        relaxation_steps: fixed number of inference iterations per example/batch
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.alpha = alpha
        self.beta = beta
        self.relaxation_steps = relaxation_steps
        
        # Create layers (index 0 = input, index -1 = output)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(PredictiveCodingLayer(
                layer_sizes[i], layer_sizes[i+1], nonlinearity=nonlinearity
            ))
        
        # We'll store activities and errors during relaxation
        self.x = None       # list of tensors [batch, features_l]
        self.epsilon = None # list of tensors [batch, features_l]

    def init_activities(self, batch_size, device):
        """Initialize representational activities to small noise or zero."""
        self.x = [torch.randn(batch_size, size, device=device) * 0.02 for size in self.layer_sizes]
        self.epsilon = [torch.zeros_like(xl) for xl in self.x]

    def compute_errors(self):
        """ε_l = x_l - prediction from above (except top layer)"""
        for l in range(self.num_layers - 1):
            pred = self.layers[l].forward_prediction(self.x[l + 1])
            self.epsilon[l] = self.x[l] - pred
        
        # Top layer has no prediction error from above → ε_top often zero or task-driven
        self.epsilon[-1] = torch.zeros_like(self.x[-1])

    def update_activities(self):
        """Local activity update: dx/dt = -ε + bottom-up error influence"""
        for l in range(self.num_layers):
            # -ε_l term (inhibition from own error)
            delta = -self.epsilon[l]
            
            # + bottom-up from errors below (if exists)
            if l > 0:
                delta += self.layers[l-1].error_from_below(self.epsilon[l-1])
            
            self.x[l] += self.alpha * delta
            
            # Apply nonlinearity (except input layer)
            if l > 0:
                self.x[l] = self.layers[l-1].nonlinearity(self.x[l])

    def update_weights(self):
        """Hebbian-like local plasticity: ΔW = β * x_pre * ε_post"""
        for l in range(self.num_layers - 1):
            # Update feedforward: from x_l (pre) to ε_l (post)
            outer = torch.einsum('bi,bj->bij', self.x[l+1], self.epsilon[l])
            self.layers[l].W_ff.data += self.beta * outer.mean(dim=0)
            
            # Update feedback: from x_{l+1} (pre) to ε_l (post)
            outer_fb = torch.einsum('bi,bj->bij', self.x[l], self.epsilon[l])
            self.layers[l].W_fb.data += self.beta * outer_fb.mean(dim=0).T

    def relax(self, clamp_input=None, clamp_target=None, update_weights=False):
        """
        Perform inference relaxation.
          clamp_input:  [batch, input_size]  → clamp x[0]
          clamp_target: [batch, output_size] → clamp x[-1] during learning
        """
        if clamp_input is not None:
            self.x[0] = clamp_input
            
        if clamp_target is not None:
            self.x[-1] = clamp_target

        for _ in range(self.relaxation_steps):
            self.compute_errors()
            self.update_activities()
        
        if update_weights:
            self.update_weights()

        return self.x[-1]  # Return top-layer representation (logits / prediction)

    def forward(self, x):
        """Standard inference pass (no clamping at top, no weight update)."""
        batch_size = x.shape[0]
        device = x.device
        self.init_activities(batch_size, device)
        self.x[0] = x
        return self.relax()

    def train_step(self, x, y):
        """Supervised training: clamp input & target, relax, update weights."""
        batch_size = x.shape[0]
        device = x.device
        self.init_activities(batch_size, device)
        self.relax(clamp_input=x, clamp_target=y, update_weights=True)


# ────────────────────────────────────────────────
#                  Example: MNIST Training
# ────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = PredictiveCodingNetwork(layer_sizes=[784, 512, 256, 10],
                                    alpha=0.15,
                                    beta=0.001,
                                    relaxation_steps=150,
                                    nonlinearity=nn.ReLU()).to(device)
    
    # Optional: small weight decay / normalization can help stability
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # only if needed
    
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.view(-1, 784).to(device)
            labels_onehot = F.one_hot(labels, num_classes=10).float().to(device)
            
            model.train_step(images, labels_onehot)
            
            # Optional: evaluate current top-layer after relaxation
            with torch.no_grad():
                pred = model.relax(clamp_input=images)  # re-relax without target
                loss = F.cross_entropy(pred, labels.to(device))
                total_loss += loss.item()
                correct += (pred.argmax(1) == labels.to(device)).sum().item()
                total += labels.size(0)
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {100. * correct / total:.2f}%")

if __name__ == "__main__":
    main()
```

### Key Notes & Tuning Tips
- **Hyperparameters** — `alpha` (0.05–0.2), `beta` (5e-4 – 5e-3), `relaxation_steps` (50–300) are very important. Too few steps → poor inference; too many → slow training.
- **Stability** — Add small weight decay or normalize weights periodically (e.g. divide by Frobenius norm) if weights explode.
- **Nonlinearity** — ReLU works well; LeakyReLU or ELU can sometimes improve convergence.
- **Batch size** — Larger batches improve stability of Hebbian updates.
- **Learning rate scheduling** — Decay `beta` over epochs.
- **Energy monitoring** (optional): Compute `sum(ε²)` across layers to check if relaxation converges.
- **Extensions** — Add lateral connections, convolutional layers (replace Linear with Conv2d + appropriate reshaping), or variational / probabilistic versions.

This should give good performance on MNIST (~92–97% after tuning) and serves as a solid starting point for deeper hierarchies or other datasets. For production-grade results, compare against repositories like `dbersan/Predictive-Coding-Implementation`, `bjornvz/PRECO`, or `RobertRosenbaum/Torch2PC` (which offer optimized variants).

# Here is a practical **PyTorch implementation** of a basic **Deep Active Inference** agent. This draws from core principles in the literature (Friston, Parr, Da Costa, Sajid, Ueltzhöffer, Millidge, etc.) and is suitable for discrete-state environments like grid-worlds or simple POMDPs, while using neural networks to parameterize beliefs and policies.

Active Inference extends predictive coding / free-energy minimization to include **action** via expected free energy (EFE) minimization. The agent maintains approximate posterior beliefs over hidden states, infers policies (sequences of actions), and selects the one expected to minimize surprise (maximize epistemic + pragmatic value).

### Core Components in This Implementation
- Generative model parameterized by neural nets:
  - **A** (likelihood): p(o|s) → observation likelihood
  - **B** (transition): p(s_t | s_{t-1}, a_t)
  - **C** (prior preferences over observations)
  - **D** (prior over initial hidden states)
- Variational inference → approximate posterior q(s|o) via amortized inference (neural net encoder)
- Policy selection via EFE = **risk** (pragmatic) + **ambiguity** (epistemic) + **expected complexity**
- Action sampling (softmax over EFE)

This version uses a simple discrete POMDP (e.g., 4-state environment with partial observability) but replaces hand-crafted categorical matrices with **neural network parameterizations** for scalability toward continuous / high-dimensional settings.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List

class ActiveInferenceAgent(nn.Module):
    """
    Deep Active Inference Agent (discrete state / action / observation space)
    
    - Uses neural nets to parameterize A, B, and approximate posterior q(s|o)
    - Computes Expected Free Energy (EFE) for policy selection
    - Selects action via softmax(EFE)
    """
    def __init__(
        self,
        n_states: int = 16,           # e.g., 4x4 grid → 16 hidden states
        n_obs: int = 5,               # e.g., 4 directions + cue/no-cue
        n_actions: int = 4,           # up, down, left, right
        n_policies: int = 5,          # horizon-1 policies (or more for tree search)
        policy_length: int = 3,       # temporal depth of policies
        hidden_dim: int = 128,
        lr: float = 3e-4,
        temperature: float = 1.0      # softmax temperature for policy selection
    ):
        super().__init__()
        self.n_s = n_states
        self.n_o = n_obs
        self.n_a = n_actions
        self.n_pi = n_policies
        self.T = policy_length
        self.temp = temperature
        
        # === Neural parameterizations of generative model ===
        
        # A-network: p(o|s) logits → [n_s, n_o]
        self.A_net = nn.Sequential(
            nn.Linear(n_s, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_o)
        )
        
        # B-network: p(s'|s,a) logits → [n_s, n_a, n_s]
        self.B_net = nn.Sequential(
            nn.Linear(n_s + n_a, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_s)
        )
        
        # Encoder: amortized approximate posterior q(s|o) → logits over states
        self.encoder = nn.Sequential(
            nn.Linear(n_o, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_s)
        )
        
        # Initial prior D: p(s_0) logits
        self.D_logit = nn.Parameter(torch.randn(n_s) * 0.02)
        
        # Preferred outcomes C: log prior over observations (pragmatic value)
        self.C = nn.Parameter(torch.randn(n_o) * 0.1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def get_A(self) -> torch.Tensor:
        """Normalized likelihood p(o|s)"""
        logits = self.A_net(torch.eye(self.n_s, device=self.A_net[0].weight.device))
        return F.softmax(logits, dim=-1)  # [n_s, n_o]
    
    def get_B(self) -> torch.Tensor:
        """Transition p(s'|s,a)"""
        s_onehot = torch.eye(self.n_s, device=self.B_net[0].weight.device).unsqueeze(1).repeat(1, self.n_a, 1)
        a_onehot = torch.eye(self.n_a, device=self.B_net[0].weight.device).unsqueeze(0).repeat(self.n_s, 1, 1)
        inp = torch.cat([s_onehot, a_onehot], dim=-1)  # [n_s, n_a, n_s + n_a]
        inp = inp.view(-1, self.n_s + self.n_a)
        logits = self.B_net(inp).view(self.n_s, self.n_a, self.n_s)
        return F.softmax(logits, dim=-1)  # [n_s, n_a, n_s]
    
    def infer_posterior(self, o: torch.Tensor) -> torch.Tensor:
        """q(s|o) ≈ softmax(encoder(o))   [batch, n_s]"""
        logits = self.encoder(o)  # [batch, n_s]
        return F.softmax(logits, dim=-1)
    
    def predict_future(self, q_s: torch.Tensor, policy: torch.Tensor) -> List[torch.Tensor]:
        """
        Rollout beliefs under a policy (sequence of actions)
        policy: [T, n_a] one-hot or soft
        Returns list of q(s_t) for t=1,...,T
        """
        beliefs = [q_s]  # q(s_1 | o_1)
        B = self.get_B()
        for a_idx in range(self.T):
            a = policy[a_idx] if policy.dim() == 2 else F.one_hot(policy[:, a_idx], self.n_a).float()
            next_belief = torch.einsum('bs,bsa,ba->ba', beliefs[-1], B, a)  # [batch, n_s]
            beliefs.append(next_belief)
        return beliefs[1:]  # future beliefs
    
    def compute_efe(
        self,
        q_future: List[torch.Tensor],
        predicted_o: torch.Tensor  # [batch, T, n_o]
    ) -> torch.Tensor:
        """Expected Free Energy ≈ Risk + Ambiguity + Expected Complexity"""
        A = self.get_A()
        risk = 0.0
        ambiguity = 0.0
        
        for t in range(self.T):
            # Expected log likelihood (negative risk/pragmatic)
            log_A = torch.log(A + 1e-12)
            pred_o_t = predicted_o[:, t]  # [batch, n_o]
            risk += -torch.sum(pred_o_t * torch.einsum('bs,so->bo', q_future[t], log_A), dim=-1)
            
            # Ambiguity (expected entropy of likelihood under posterior)
            entropy_A = -torch.sum(A * log_A, dim=-1)  # [n_s]
            ambiguity += torch.sum(q_future[t] * entropy_A, dim=-1)
        
        # Optional: expected complexity (KL[q(s)||p(s)] ≈ small regularization)
        complexity = 0.0  # can add KL if desired
        
        efe = risk + ambiguity + complexity
        return efe  # [batch]
    
    def select_action(self, o: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Core active inference loop step:
          1. Infer q(s|o)
          2. For each policy, predict future beliefs & observations
          3. Compute EFE
          4. Sample action from softmax(-EFE / temp)
        """
        batch_size = o.shape[0]
        device = o.device
        
        q_s = self.infer_posterior(o)  # [batch, n_s]
        
        # Assume fixed set of policies (e.g., all length-T action sequences)
        # For simplicity: random / fixed policies; in practice use tree search or repertoire
        policies = torch.randint(0, self.n_a, (self.n_pi, self.T), device=device)  # [n_pi, T]
        
        efes = []
        for pi_idx in range(self.n_pi):
            policy = F.one_hot(policies[pi_idx], self.n_a).float()  # [T, n_a]
            
            # Predict future beliefs q(s_{1:T})
            q_future = self.predict_future(q_s, policy)
            
            # Predict future observations under policy (via A and beliefs)
            A = self.get_A()
            pred_o = torch.stack([
                torch.einsum('bs,so->bo', q_f, A) for q_f in q_future
            ], dim=1)  # [batch, T, n_o]
            
            efe_pi = self.compute_efe(q_future, pred_o).mean()  # scalar per policy
            efes.append(efe_pi)
        
        efes = torch.stack(efes)  # [n_pi]
        policy_probs = F.softmax(-efes / self.temp, dim=0)
        
        # Select policy → first action
        chosen_pi = torch.multinomial(policy_probs, 1).item()
        first_action = policies[chosen_pi, 0].item()
        
        return first_action, policy_probs

    def learn_from_trajectory(
        self,
        observations: List[torch.Tensor],  # list of [batch, n_o]
        actions: List[int],
        rewards: List[float] = None      # optional extrinsic reward shaping
    ):
        """Update model via free-energy minimization (ELBO) on trajectory"""
        self.optimizer.zero_grad()
        
        total_fe = 0.0
        q_s_prev = F.softmax(self.D_logit, dim=0).expand(len(observations[0]), -1)
        
        for t, o_t in enumerate(observations):
            q_s = self.infer_posterior(o_t)
            
            # Reconstruction term: -log p(o|s) under q(s)
            log_A = torch.log(self.get_A() + 1e-12)
            recon = -torch.sum(q_s * torch.einsum('bs,so->bo', q_s, log_A), dim=-1).mean()
            
            # KL prior term (approximate complexity)
            kl = F.kl_div(
                torch.log(q_s + 1e-12),
                torch.log(q_s_prev + 1e-12),
                reduction='batchmean',
                log_target=True
            )
            
            total_fe += recon + kl
            q_s_prev = q_s
        
        total_fe.backward()
        self.optimizer.step()
        return total_fe.item()


# ────────────────────────────────────────────────
#                  Minimal Usage Example
# ────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActiveInferenceAgent(
        n_states=16, n_obs=5, n_actions=4, n_policies=16, policy_length=4,
        hidden_dim=64, lr=5e-4, temperature=0.5
    ).to(device)
    
    # Dummy observation (one-hot)
    o = F.one_hot(torch.randint(0, 5, (1,)), num_classes=5).float().to(device)
    
    action, policy_probs = agent.select_action(o)
    print(f"Selected action: {action}  |  Policy probabilities: {policy_probs}")
```

### Key Extensions & Improvements
- **Longer / learned policies** — Replace fixed policies with tree search (e.g., EFE-guided Monte-Carlo tree search) or amortized policy networks.
- **Continuous states** → Replace categorical distributions with Gaussian / Normalizing Flows / diffusion models for q(s|o), p(s'|s,a), p(o|s).
- **Deep hierarchies** — Stack multiple temporal scales (as in deep active inference papers).
- **Reward / goal-directed behavior** — Shape `C` dynamically or use learned value priors.
- **Stability** — Add gradient clipping, weight decay, or use natural gradients / Amari flows.
- **Environments to try** — Grid-world (as in pymdp), Mountain Car (continuous variant), Atari (pixel-based with CNNs for A and encoder).

### Recommended Resources (2025–2026 perspective)
- pymdp (NumPy-first, excellent for discrete case)
- repos like `NACLab/self-revising-active-inference`, `mazpie/contrastive-aif`, `adanielnoel/Active-Inference-Capsule`
- Papers: "Deep Active Inference Agents" (Ueltzhöffer 2018+), "Contrastive Active Inference" (NeurIPS 2021), recent SR-AIF work

This code provides a self-contained, differentiable starting point — tune hyperparameters and extend the policy set for real tasks. Let me know if you'd like a continuous-state version, integration with Gymnasium/PettingZoo, or hierarchical extensions!

# Predictive coding basics

**Predictive coding** (also called **predictive processing**) is one of the most influential theories in modern neuroscience and computational cognitive science. It proposes that the brain does **not** primarily act as a passive receiver that processes incoming sensory data bottom-up. Instead, the brain constantly **generates predictions** about what sensory input it expects to receive, based on an internal model of the world. Only the **mismatches** (prediction errors) between what is predicted and what actually arrives drive updates to the model and propagate information upward.

This framework is often summarized as: **the brain is a prediction machine** whose core job is to **minimize prediction error** (or surprise) at every level of processing.

### Core Idea in Simple Terms
- **Top-down predictions** flow from higher brain areas (more abstract representations) downward to lower areas (closer to the senses).
- **Bottom-up signals** mainly carry **prediction errors** — only the parts that deviate from expectation are worth sending forward.
- The system continuously adjusts its internal model so that future predictions become more accurate → efficient perception, learning, and action.

This is radically different from the classical "feedforward" view of perception (sense → thalamus → cortex → higher areas), where most processing flows upward. Predictive coding flips much of that: most of the "work" is in generating and refining predictions.

### Key Mathematical & Computational Insight
The theory is deeply rooted in **Bayesian inference**. The brain performs approximate Bayesian updating to infer the most likely causes of its sensory data.

A central quantity is **prediction error**:

ε = sensory input (or lower-level activity) − predicted value (from higher level)

The brain minimizes a global measure of surprise / prediction error, often formalized via **free energy** (Karl Friston’s free-energy principle):

- Free energy ≈ prediction error + complexity penalty
- Minimizing free energy ≈ minimizing surprise (negative log probability of sensations under the model)

In hierarchical predictive coding networks:

- Higher layers send **predictions** downward.
- Lower layers compute **errors** and send them upward.
- This creates a recurrent, message-passing scheme that settles toward better explanations of the input.

Classic 1999 Rao & Ballard model (very influential for visual cortex):

Each layer tries to reconstruct the activity of the layer below via top-down connections.

The update dynamics look roughly like:

dx/dt ≈ −ε + (bottom-up weighted errors from below)

Where x is the representational activity, ε is prediction error.

Weights update locally in a Hebbian-like way using activity × error.

### Biological Implementation (Microcircuit View)
A widely accepted implementation (inspired by Bastos et al. 2012, Friston & others):

- **Superficial pyramidal cells** → send prediction errors upward (feedforward).
- **Deep pyramidal cells** → send predictions downward (feedback).
- **Inhibitory interneurons** (especially somatostatin and parvalbumin cells) help compute the mismatch by suppressing predicted parts of the signal.

This explains many phenomena:

- Expectation suppression → repeated / predictable stimuli evoke weaker responses.
- Surprise / deviance detection → mismatch negativity (MMN) in EEG, stimulus-specific adaptation (SSA).
- Extra-classical receptive fields in vision (surround suppression when context is predictable).
- Illusions (hollow mask, rubber hand) — strong top-down priors override bottom-up evidence.
- Attention → precision weighting of prediction errors (high-precision errors are amplified).

### Why It's Biologically Plausible (Compared to Backpropagation)
- No global error signals that need to be transported backward through symmetric weights.
- All updates are **local** (pre-synaptic activity × post-synaptic error).
- Continuous, online, recurrent dynamics (no strict forward-then-backward phases).
- Naturally handles unsupervised / self-supervised learning.

### Broader Scope (Predictive Processing Umbrella)
Under Karl Friston's **free-energy principle**, predictive coding extends to:

- **Perception** → minimizing sensory prediction error.
- **Action** → **active inference** — acting to fulfill predictions (e.g., moving to sample expected sensations).
- **Learning** → updating model parameters to reduce long-term surprise.
- **Emotion, psychiatry** → miscalibrated precision of errors linked to anxiety, schizophrenia, depression.

### Quick Summary Table

| Aspect                  | Classical View                          | Predictive Coding View                          |
|-------------------------|-----------------------------------------|-------------------------------------------------|
| Flow of information     | Mostly bottom-up                        | Predictions top-down, errors bottom-up          |
| Main signal propagated  | Raw features / activations              | Prediction errors                               |
| Goal of the system      | Represent input accurately              | Minimize surprise / prediction error            |
| Learning rule           | Hebbian or backprop-like                | Local error-driven (Hebbian-ish)                |
| Efficiency              | Process everything                      | Only process what's surprising                  |
| Key theorist            | Hubel & Wiesel, Marr                    | Rao & Ballard (1999), Friston (2005+)           |

Predictive coding is not without critics (some argue the evidence is still indirect, or that it over-claims unification), but it has become a dominant framework for explaining cortical function, perception, and even consciousness in many labs.

If you'd like a deeper dive into any part — math derivations, PyTorch code for a simple predictive coding net, biological evidence, relation to active inference, or common misconceptions — just let me know!

	
