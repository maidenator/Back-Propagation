# Primitive Self-Documenting Backpropagation Neural Network

An educational, foundational **Backpropagation Artificial Neural Network** implemented in pure, idiomatic **C# (.NET 8.0)** without external machine learning dependencies.

---

## 1. Core Philosophy

### A. Absolute Primitiveness
- **Zero Third-Party ML Packages**: No ML.NET, no TorchSharp, no external linear algebra libraries.
- **Elementary Language Primitives**: Constructed entirely with standard C# primitives (`double[]`, `double[][]`), explicit nested loops, and `System.Math`.
- **Complete Mathematical Transparency**: Every dot product, activation calculation, error derivative, gradient signal, and momentum update is fully written out and exposed—nothing is obscured inside black-box tensor libraries.

### B. Autonomous Self-Documentation
- **Step-by-Step Mathematical Tracer (`--trace`)**: Traces any single sample forward and backward, printing intermediate arithmetic with actual numerical values.
- **Automated Markdown Run Reports (`--report`)**: Writes persistent, structured Markdown reports into `reports/` upon training completion, detailing convergence metrics, truth table verification tables, and weight matrices.
- **Console ASCII Topology Visualizer (`--visualize`)**: Draws dynamic ASCII diagrams of network architecture and connection parameters directly in the terminal.

---

## 2. Mathematical Foundations

### 2.1 Forward Propagation

For hidden node $j \in \{0, \dots, H-1\}$:
$$z_j^{(h)} = \sum_{i=0}^{M-1} x_i \cdot w_{ij}^{(ih)} + b_j^{(h)}$$
$$h_j = \sigma(z_j^{(h)}) = \frac{1}{1 + e^{-z_j^{(h)}}} \quad \text{or} \quad \tanh(z_j^{(h)})$$

For output node $k \in \{0, \dots, N-1\}$:
$$z_k^{(o)} = \sum_{j=0}^{H-1} h_j \cdot w_{jk}^{(ho)} + b_k^{(o)}$$
$$y_k = \sigma(z_k^{(o)}) \quad \text{or} \quad \text{Softmax}(z_k^{(o)}) = \frac{e^{z_k^{(o)}}}{\sum_{m} e^{z_m^{(o)}}}$$

---

### 2.2 Error Gradient Backpropagation

#### Output Layer Error Signal ($\delta_k$)
Using squared error $E = \frac{1}{2} \sum_k (t_k - y_k)^2$ with logistic sigmoid activation:
$$\delta_k = (t_k - y_k) \cdot \sigma'(y_k) = (t_k - y_k) \cdot y_k \cdot (1 - y_k)$$

#### Hidden Layer Error Signal ($\delta_j$)
Backpropagating error from the output layer to hidden node $j$:
$$\delta_j = \sigma'(h_j) \cdot \sum_{k=0}^{N-1} \delta_k \cdot w_{jk}^{(ho)}$$
For $\tanh$ activations:
$$\delta_j = (1 + h_j)(1 - h_j) \cdot \sum_{k=0}^{N-1} \delta_k \cdot w_{jk}^{(ho)}$$

---

### 2.3 Weight & Bias Update Rules (with Momentum)

$$\Delta w_{jk}^{(ho)} = \eta \cdot \delta_k \cdot h_j + \alpha \cdot \Delta w_{jk}^{(ho, \text{prev})}$$
$$w_{jk}^{(ho)} \leftarrow w_{jk}^{(ho)} + \Delta w_{jk}^{(ho)}$$

$$\Delta b_k^{(o)} = \eta \cdot \delta_k \cdot 1.0 + \alpha \cdot \Delta b_k^{(o, \text{prev})}$$
$$b_k^{(o)} \leftarrow b_k^{(o)} + \Delta b_k^{(o)}$$

*(Analogous rules apply for input-to-hidden weights $w_{ij}^{(ih)}$ and hidden biases $b_j^{(h)}$.)*

---

## 3. Project Structure

```
Back-Propagation/
├── BackPropagation.sln                 # Visual Studio / .NET Solution
├── PROJECT_PLAN.md                     # Architectural specification document
├── README.md                           # Documentation, theory, and quickstart
├── reports/                            # Auto-generated Markdown training reports
│   └── run_YYYYMMDD_HHMMSS.md
├── materials/                          # [GITIGNORED] Original CS345 PDF & C# files
├── references/                         # [GITIGNORED] Cloned Xylphy reference repo
│
└── src/
    ├── BackPropagation.csproj          # .NET 8.0 project file
    │
    ├── Core/                           # Primitive Engine Components
    │   ├── Activations.cs              # Sigmoid, Tanh, Softmax + analytical derivatives
    │   ├── INeuron.cs                  # Input neuron entity (Xylphy reference model)
    │   ├── HNeuron.cs                  # Hidden neuron with backprop gradient calculation
    │   ├── ONeuron.cs                  # Output neuron entity
    │   ├── NeuralNet.cs                # Object-oriented neural network coordinator
    │   └── MatrixNeuralNet.cs          # Primitive matrix/array engine (CS345 McCaffrey model)
    │
    ├── Data/                           # Dataset Generators
    │   ├── LogicGateDataset.cs         # 16-row 4-input boolean truth tables (AND, XOR)
    │   └── SyntheticDataset.cs         # 1000-sample 4-5-3 synthetic classifier dataset
    │
    ├── SelfDoc/                        # Autonomous Documentation Subsystem
    │   ├── MathTracer.cs               # Single-sample step-by-step arithmetic logger
    │   ├── AsciiVisualizer.cs          # Console ASCII topology & weight renderer
    │   └── MarkdownReporter.cs         # Generates structured Markdown run reports
    │
    ├── Experiments/                    # Core Benchmarks & Tasks
    │   ├── LogicGateExperiment.cs      # 4-input logic gate (train-until-fit + verification)
    │   ├── HiddenOptimizer.cs          # Hidden neuron sweep benchmark (buttons 4 & 5)
    │   └── SyntheticBenchmark.cs       # 4-5-3 multi-class classifier demo (CS345)
    │
    └── Program.cs                      # CLI entry point & interactive console menu
```

---

## 4. Quickstart & Usage

### 4.1 Prerequisites
- [.NET 8.0 SDK](https://dotnet.microsoft.com/download)

### 4.2 Build the Solution
```bash
dotnet build
```

### 4.3 Launch the Interactive Menu
```bash
dotnet run --project src/BackPropagation.csproj
```

The interactive menu provides:
```
================================================================================
          PRIMITIVE SELF-DOCUMENTING BACKPROPAGATION NEURAL NETWORK             
================================================================================
  [1] Run 4-Input Logic Gate Experiment (Train-Until-Fit & Generate Report)
  [2] Run CS345 4-5-3 Synthetic Classifier Benchmark (McCaffrey)
  [3] Run Hidden Neuron Count Optimization Benchmark
  [4] Mathematical Step-by-Step Calculation Trace (--trace)
  [5] Interactive Manual Input Testing (4 Inputs -> Output)
  [6] Display ASCII Network Topology Diagram
  [7] Exit
================================================================================
```

### 4.4 Command-Line Flags
You can run any experiment non-interactively via flags:

| Flag | Description |
| :--- | :--- |
| `--logic`, `-l` | Trains a 4-input logic gate until fit and saves a Markdown run report |
| `--synthetic`, `-s` | Executes the CS345 4-5-3 synthetic classification benchmark |
| `--optimize`, `-o` | Sweeps hidden neuron counts to determine fastest convergence |
| `--trace`, `-t` | Runs an arithmetic calculation trace for a single sample |
| `--visualize`, `-v` | Renders the network topology directly in ASCII |

**Examples:**
```bash
# Run the 4-input logic gate experiment
dotnet run --project src/BackPropagation.csproj -- --logic

# Run mathematical step-by-step trace
dotnet run --project src/BackPropagation.csproj -- --trace

# Run CS345 4-5-3 synthetic benchmark
dotnet run --project src/BackPropagation.csproj -- --synthetic
```

---

## 5. Reference Synthesis & Comparison

| Feature | CS345 / McCaffrey Reference | Xylphy WinForms Reference | This Implementation |
| :--- | :--- | :--- | :--- |
| **Data Structures** | Jagged arrays (`double[][]`) | Object hierarchy (`INeuron`, `HNeuron`) | **Both supported** (`MatrixNeuralNet` & `NeuralNet`) |
| **Problem Domain** | 4-5-3 Synthetic Multi-Class (1000 items) | 4-Input Logic Gate (16-row truth table) | **Both supported** via standalone experiments |
| **Activations** | Tanh hidden + Softmax output | Sigmoid throughout | **Configurable** (`Activations.cs`) |
| **Convergence** | Fixed epochs (1000) with momentum | Train-until-fit (`fitTest() == true`) | **Both fixed epochs & train-until-fit** |
| **Self-Documentation** | Console logs | GUI textboxes | **ASCII topology, step tracer, and auto-generated Markdown reports** |
