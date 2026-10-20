# CS345 Backpropagation Neural Network
### With Windows Forms GUI, Cross-Entropy Loss & L2 Weight Decay

A foundational, educational **Backpropagation Artificial Neural Network** implemented in pure **C# (.NET 8.0)** without external machine learning packages, mirroring and extending the **CS345 IS 1 specification** by Dr. James McCaffrey.

---

## 1. Key Features

### A. Core CS345 Architecture
- **4 Inputs → 5 Hidden Units → 3 Output Units (4-5-3)**: Matches the exact 43-parameter topology from the CS345 reference.
- **1,000 Synthetic Data Items**: Seeded artificial data generation with 1-of-N one-hot target encoding and 80/20 train/test partitioning.
- **Tanh Activation (Hidden) & Softmax (Output)**: Classic zero-centered hidden activations combined with normalized probability distribution outputs.
- **Online SGD with Momentum**: Stochastic gradient descent with previous delta retention ($\alpha = 0.01$).

### B. Extensions from 'WHAT TO DO' (Page 8 of Assignment)
1. **Cross-Entropy Error**:
   $$E = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} t_{ik} \ln(y_{ik})$$
   When combined with Softmax output activation, the error gradient simplifies to:
   $$\delta_k = t_k - y_k$$
   *(The derivative factor $y_k(1 - y_k)$ is canceled out analytically, eliminating gradient saturation.)*

2. **Weight Decay (L2 Regularization)**:
   Penalizes large weight magnitudes to prevent overfitting:
   $$E_{\text{total}} = E + \frac{1}{2} \lambda \sum w^2$$
   $$\Delta w = \eta \cdot (\text{grad} - \lambda \cdot w) + \alpha \cdot \Delta w_{\text{prev}}$$
   *(Keeps weight magnitudes compact and ensures better generalization on unseen test data.)*

### C. Dual Interface (GUI + CLI)
- **Windows Forms GUI (`MainForm.cs`)**: Replicates the interactive desktop experience with buttons for data generation, network creation, real-time training progress, manual prediction on custom inputs, and Markdown report export.
- **Self-Documenting Terminal Interface**: Runs on Linux and CLI environments with interactive menus, ASCII network visualizer, and single-sample mathematical trace.

---

## 2. Mathematical Formulations

### 2.1 Forward Pass
For hidden node $j \in \{0, \dots, 4\}$:
$$z_j^{(h)} = \sum_{i=0}^{3} x_i \cdot w_{ij}^{(ih)} + b_j^{(h)}, \quad h_j = \tanh(z_j^{(h)}) = \frac{e^{z_j} - e^{-z_j}}{e^{z_j} + e^{-z_j}}$$

For output node $k \in \{0, 1, 2\}$:
$$z_k^{(o)} = \sum_{j=0}^{4} h_j \cdot w_{jk}^{(ho)} + b_k^{(o)}, \quad y_k = \text{Softmax}(z_k^{(o)}) = \frac{e^{z_k^{(o)}}}{\sum_{m=0}^{2} e^{z_m^{(o)}}}$$

### 2.2 Error Gradient Backpropagation
- **Under Mean Squared Error (MSE)**:
  $$\delta_k = (t_k - y_k) \cdot y_k \cdot (1 - y_k)$$
- **Under Cross-Entropy Error**:
  $$\delta_k = t_k - y_k$$
- **Hidden Node Signal ($\delta_j$)**:
  $$\delta_j = (1 - h_j^2) \cdot \sum_{k=0}^{2} \delta_k \cdot w_{jk}^{(ho)}$$

### 2.3 Weight & Bias Updates
- **Hidden-to-Output Weights**:
  $$\Delta w_{jk}^{(ho)} = \eta \cdot (\delta_k \cdot h_j - \lambda \cdot w_{jk}^{(ho)}) + \alpha \cdot \Delta w_{jk}^{(ho, \text{prev})}$$
- **Input-to-Hidden Weights**:
  $$\Delta w_{ij}^{(ih)} = \eta \cdot (\delta_j \cdot x_i - \lambda \cdot w_{ij}^{(ih)}) + \alpha \cdot \Delta w_{ij}^{(ih, \text{prev})}$$
- **Biases (Unregularized)**:
  $$\Delta b_k^{(o)} = \eta \cdot \delta_k + \alpha \cdot \Delta b_k^{(o, \text{prev})}, \quad \Delta b_j^{(h)} = \eta \cdot \delta_j + \alpha \cdot \Delta b_j^{(h, \text{prev})}$$

---

## 3. Project Structure

```
Back-Propagation/
├── BackPropagation.sln                 # Visual Studio .NET Solution
├── README.md                           # Documentation and math guide
├── materials/                          # [GITIGNORED] Original PDF & C# files
├── references/                         # [GITIGNORED] Cloned Xylphy reference repo
├── reports/                            # Auto-generated Markdown training reports
│
└── src/
    ├── BackPropagation.csproj          # Multi-targeted (.NET 8.0 & .NET 8.0 Windows)
    │
    ├── Core/                           # Primitive Engine Components
    │   ├── Activations.cs              # Sigmoid, Tanh, Softmax + analytical derivatives
    │   ├── MatrixNeuralNet.cs          # CS345 array engine with Cross-Entropy & Weight Decay
    │   ├── NeuralNet.cs                # Object-oriented network coordinator
    │   ├── INeuron.cs                  # Input neuron entity
    │   ├── HNeuron.cs                  # Hidden neuron with backprop error calculation
    │   └── ONeuron.cs                  # Output neuron entity
    │
    ├── Data/                           # Dataset Generators
    │   ├── SyntheticDataset.cs         # 1000-sample 4-5-3 synthetic dataset generator
    │   └── LogicGateDataset.cs         # 16-row 4-input boolean truth tables (AND, XOR)
    │
    ├── GUI/                            # Desktop Interface
    │   └── MainForm.cs                 # Windows Forms graphical dashboard
    │
    ├── SelfDoc/                        # Autonomous Self-Documentation
    │   ├── MathTracer.cs               # Single-sample step-by-step arithmetic trace
    │   ├── AsciiVisualizer.cs          # Console ASCII topology renderer
    │   └── MarkdownReporter.cs         # Automated Markdown report generator
    │
    ├── Experiments/                    # Core Benchmarks & Tasks
    │   ├── SyntheticBenchmark.cs       # CS345 demo (MSE, Cross-Entropy, Weight Decay)
    │   ├── LogicGateExperiment.cs      # 4-input logic gate (train-until-fit)
    │   └── HiddenOptimizer.cs          # Hidden neuron sweep benchmark
    │
    └── Program.cs                      # Entry point (auto-launches GUI on Windows, CLI on Linux)
```

---

## 4. How to Run

### 4.1 On Windows (Graphical Interface)
Simply open `BackPropagation.sln` in **Visual Studio** and press **F5** (or run `dotnet run` in the terminal). The **Windows Forms GUI** will launch automatically.

### 4.2 In Terminal / Linux (Interactive Menu)
```bash
dotnet run -f net8.0 --project src/BackPropagation.csproj
```

### 4.3 Command-Line Flags
You can run any benchmark directly with dedicated flags:

```bash
# 1. Standard CS345 Benchmark (Mean Squared Error)
dotnet run -f net8.0 --project src/BackPropagation.csproj -- --synthetic

# 2. CS345 Extension: Cross-Entropy Loss (from 'WHAT TO DO')
dotnet run -f net8.0 --project src/BackPropagation.csproj -- --cross-entropy

# 3. CS345 Extension: Weight Decay / L2 Regularization (from 'WHAT TO DO')
dotnet run -f net8.0 --project src/BackPropagation.csproj -- --weight-decay

# 4. Step-by-Step Mathematical Trace
dotnet run -f net8.0 --project src/BackPropagation.csproj -- --trace

# 5. ASCII Network Topology
dotnet run -f net8.0 --project src/BackPropagation.csproj -- --visualize

# 6. 4-Input Logic Gate Experiment
dotnet run -f net8.0 --project src/BackPropagation.csproj -- --logic
```
