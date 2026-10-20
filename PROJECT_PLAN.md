# Project Plan: Primitive Self-Documenting C# Backpropagation Neural Network

## 1. Executive Summary & Core Philosophy

This project is a 100% **C#** implementation of an educational **Backpropagation Artificial Neural Network** engineered from first principles, drawing on Dr. James McCaffrey's CS345 article and reference program (kept locally under `materials/`), and on [Xylphy's reference repository](https://github.com/Xylphy/Backpropagation-Neural-Networks/). The course materials and the cloned reference repository are intentionally excluded from the published repository; see [README.md](README.md) for details.

The system is governed by two strict tenets:
1. **Absolute Primitiveness**: Zero external packages or machine learning frameworks (no ML.NET, no TorchSharp, no third-party linear algebra libraries). The engine relies exclusively on core C# primitive types (`double[]`, `double[][]`), standard loops, and standard `System.Math`. Every forward pass dot product, activation function, error derivative, gradient propagation, and momentum update is fully explicit and observable.
2. **Autonomous Self-Documentation**: The project documents itself at runtime. It includes a built-in step-by-step mathematical tracer that prints intermediate arithmetic for each node, an ASCII terminal topology visualizer, and an automated Markdown report generator that writes run summaries with truth tables and weight matrices.

---

## 2. Reference Synthesis

```
┌────────────────────────────────────────────────────────────────────────┐
│                        REFERENCE SYNTHESIS                             │
├──────────────────────────────────┬─────────────────────────────────────┤
│ Reference A: CS345 / McCaffrey   │ Reference B: Xylphy C# Reference    │
│ (Back-Prop PDF & CS345 IS 1.cs)  │ (Backpropagation-Neural-Networks)   │
├──────────────────────────────────┼─────────────────────────────────────┤
│ • Flat jagged array architecture │ • Object-oriented neuron hierarchy  │
│   (double[][], double[])         │   (INeuron, HNeuron, ONeuron)       │
│ • 4 inputs -> 5 hidden -> 3 out  │ • 4 inputs -> N hidden -> 1 output  │
│ • Softmax output + Tanh hidden   │ • Sigmoid activations throughout    │
│ • Stochastic / Online SGD + Mom  │ • 16-row Truth Table Logic Gate     │
│ • Train/Test 80/20 data split    │ • Train-until-fit convergence loop  │
│ • Detailed mathematical symbols  │ • Hidden neuron count optimization  │
└──────────────────────────────────┴─────────────────────────────────────┘
```

---

## 3. C# Project Architecture & Structure

```
Back-Propagation/
├── .github/workflows/ci.yml            # Cross-platform build and CLI smoke tests
├── .editorconfig                       # C# formatting conventions
├── .gitattributes                      # Line-ending normalization
├── .gitignore                          # Ignores materials/, references/, reports/, bin/, obj/
├── LICENSE                             # MIT license
├── PROJECT_PLAN.md                     # This architectural document
├── README.md                           # Documentation, mathematical derivations, quickstart
├── materials/                          # [LOCAL ONLY / GITIGNORED] Original CS345 PDF & C# files
├── references/                         # [LOCAL ONLY / GITIGNORED] Cloned Xylphy repository
├── reports/                            # [GENERATED / GITIGNORED] Markdown training reports
│
├── BackPropagation.sln                 # Visual Studio / .NET Solution
│
└── src/
    ├── BackPropagation.csproj          # .NET 8; net8.0-windows on Windows, net8.0 elsewhere
    │
    ├── Core/                           # Primitive Neural Network Engine
    │   ├── INeuron.cs                  # Input neuron representation (Xylphy model)
    │   ├── HNeuron.cs                  # Hidden neuron with error backprop & bias
    │   ├── ONeuron.cs                  # Output neuron with delta signal computation
    │   ├── NeuralNet.cs                # High-level object-oriented network coordinator
    │   ├── MatrixNeuralNet.cs          # Primitive matrix/array engine (CS345 McCaffrey model)
    │   └── Activations.cs              # Sigmoid, Tanh, Softmax and their derivatives
    │
    ├── Data/                           # Datasets & Generation
    │   ├── LogicGateDataset.cs         # 16-state 4-input boolean truth tables (AND, parity)
    │   └── SyntheticDataset.cs         # CS345 4-5-3 synthetic classification dataset generator
    │
    ├── SelfDoc/                        # Self-Documentation Subsystem
    │   ├── MathTracer.cs               # Step-by-step arithmetic logger for forward/backward pass
    │   ├── AsciiVisualizer.cs          # Console ASCII topology renderer (nodes & weights)
    │   └── MarkdownReporter.cs         # Generates GitHub-flavored markdown run reports
    │
    ├── Experiments/                    # Core Tasks & Benchmarks
    │   ├── LogicGateExperiment.cs      # Train-until-fit on the 16-row truth table
    │   ├── HiddenOptimizer.cs          # Finds minimum hidden neurons needed to converge
    │   └── SyntheticBenchmark.cs       # 4-5-3 multi-class classifier demo (MSE, CE, weight decay)
    │
    ├── GUI/                            # Windows-only Desktop Interface
    │   └── MainForm.cs                 # Windows Forms dashboard
    │
    └── Program.cs                      # CLI/menu entry point; launches GUI on Windows by default
```

---

## 4. Mathematical Specifications in C#

All core equations directly correspond to the CS345 documentation:

### 4.1 Forward Pass
For hidden node $j$:
$$\text{sum}_j = \sum_{i=0}^{\text{numInput}-1} x_i \cdot w_{ij}^{(ih)} + b_j^{(h)}$$
$$h_j = \frac{1}{1 + e^{-\text{sum}_j}} \quad \text{or} \quad \tanh(\text{sum}_j)$$

For output node $k$:
$$\text{sum}_k = \sum_{j=0}^{\text{numHidden}-1} h_j \cdot w_{jk}^{(ho)} + b_k^{(o)}$$
$$y_k = \frac{1}{1 + e^{-\text{sum}_k}} \quad \text{or} \quad \text{Softmax}(\text{sum}_k)$$

### 4.2 Backward Pass (Error Gradients)
Output error gradient signal $\delta_k$:
$$\delta_k = (t_k - y_k) \cdot y_k \cdot (1 - y_k)$$

Hidden error gradient signal $\delta_j$:
$$\delta_j = (1 - h_j^2) \cdot \sum_{k=0}^{\text{numOutput}-1} \delta_k \cdot w_{jk}^{(ho)}$$

### 4.3 Weight & Bias Updates with Momentum
$$\Delta w_{jk}^{(ho)} = \eta \cdot \delta_k \cdot h_j + \alpha \cdot \Delta w_{jk}^{(ho, \text{prev})}$$
$$w_{jk}^{(ho)} \leftarrow w_{jk}^{(ho)} + \Delta w_{jk}^{(ho)}$$
$$\Delta b_k^{(o)} = \eta \cdot \delta_k \cdot 1.0 + \alpha \cdot \Delta b_k^{(o, \text{prev})}$$
$$b_k^{(o)} \leftarrow b_k^{(o)} + \Delta b_k^{(o)}$$

### 4.4 Implemented Extensions

- **Cross-entropy loss** with Softmax output: the output signal simplifies to $\delta_k = t_k - y_k$.
- **L2 weight decay**: the gradient used for weight updates becomes $\text{grad} - \lambda w$; biases are not regularized.
- **Windows Forms GUI**: generates data, trains, predicts, traces, and exports reports.
- **Packaging hygiene**: OS-conditional target framework, `.gitignore`, `.gitattributes`, `.editorconfig`, MIT license, and CI builds on Linux and Windows.

---

## 5. Self-Documentation Subsystem

1. **Step-by-Step Math Tracer (`MathTracer.cs`)**:
   - Traces arithmetic calculations with actual values for any selected sample.
   - Outputs intermediate sums, activation results, target errors, local gradients ($\delta$), and the weight/bias update equations.

2. **Automated Markdown Reporter (`MarkdownReporter.cs`)**:
   - Generates formatted reports in `reports/cs345_run_<timestamp>.md` and `reports/run_<timestamp>.md` upon training completion.
   - Includes hyperparameter summary, epoch loss table, weight matrices, and truth table verification table (comparing expected vs. computed output for all 16 states).
   - Filenames include milliseconds and a collision suffix, so concurrent runs never overwrite each other.

3. **ASCII Network Visualizer (`AsciiVisualizer.cs`)**:
   - Displays a dynamic ASCII diagram in the console showing input, hidden, and output layers with their connections.

---

## 6. Implementation Milestones

* **Milestone 1**: Initialize C# project structure (`BackPropagation.csproj`, `Program.cs`).
* **Milestone 2**: Implement primitive core (`Activations.cs`, `MatrixNeuralNet.cs`, and `NeuralNet.cs` with `INeuron`, `HNeuron`, `ONeuron`).
* **Milestone 3**: Implement self-documentation engine (`MathTracer.cs`, `AsciiVisualizer.cs`, `MarkdownReporter.cs`).
* **Milestone 4**: Implement the 16-row Truth Table Logic Gate and Hidden Neuron Optimizer.
* **Milestone 5**: Implement the CS345 4-5-3 Synthetic Classifier benchmark.
* **Milestone 6**: Provide interactive CLI menu and self-contained README documentation.
* **Milestone 7**: Add Cross-Entropy Loss and L2 Weight Decay extensions, plus the Windows Forms GUI.
* **Milestone 8**: Harden the repository for publication: license, CI, line-ending/formatting metadata, and generated-output cleanup.
