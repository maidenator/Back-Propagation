# Project Plan: C# Backpropagation Neural Network (CS345)

## 1. Overview

A from-scratch, dependency-free implementation of a feed-forward backpropagation network in C# / .NET 8, following the 4-5-3 architecture of Dr. James McCaffrey's CS345 reference program, plus the two "WHAT TO DO" extensions (cross-entropy loss, L2 weight decay).

Guiding principles: explicit arithmetic over performance, no third-party packages, reproducible seeded data, and the same run configuration for every configuration being compared.

## 2. Structure

```
src/
├── Program.cs            # CLI entry point, interactive menu
├── Core/
│   ├── Activations.cs    # sigmoid, tanh, softmax + derivatives
│   ├── MatrixNeuralNet.cs# jagged-array engine (CE + weight decay)
│   └── NeuralNet.cs      # OO coordinator with INeuron/HNeuron/ONeuron
├── Data/
│   ├── SyntheticDataset.cs
│   └── LogicGateDataset.cs
├── Experiments/
│   ├── ComparisonRunner.cs  # 3-way comparison + summary table
│   ├── SyntheticBenchmark.cs
│   ├── LogicGateExperiment.cs
│   └── HiddenOptimizer.cs
├── SelfDoc/
│   ├── MathTracer.cs
│   ├── AsciiVisualizer.cs
│   ├── ConsoleDisplay.cs
│   └── MarkdownReporter.cs
└── GUI/MainForm.cs       # Windows Forms dashboard (Windows only)
```

## 3. Mathematics

**Forward pass.** Hidden node $j$: $z_j = \sum_i x_i w_{ij} + b_j$, $h_j = \tanh(z_j)$. Output node $k$: $z_k = \sum_j h_j w_{jk} + b_k$, $y_k = \mathrm{Softmax}(z_k)$.

**Backward pass.** Output signal $\delta_k$, hidden signal $\delta_j = (1 - h_j^2) \sum_k \delta_k w_{jk}$.

**Updates.** $\Delta w = \eta \,\delta \,\text{input} + \alpha \,\Delta w_{prev}$, with the same form for biases.

**Loss variants.**
- MSE: $\delta_k = (t_k - y_k)\,y_k\,(1 - y_k)$.
- Cross-entropy: $\delta_k = t_k - y_k$ — the softmax derivative cancels analytically, so the gradient never saturates.

**Weight decay.** Applied after each weight update as $w \leftarrow w - \lambda w$, matching the CS345 reference implementation. Biases are never decayed.

## 4. Experiments

| Run | Purpose |
|---|---|
| Comparison runner | Trains MSE, cross-entropy, and cross-entropy + decay on one shared dataset, then prints the comparison table (loss, weight norm, train/test accuracy). |
| Synthetic benchmark | One configuration, verbose epoch log, markdown report, sample prediction. |
| Logic gate experiment | Trains a 4-input network until all 16 truth-table rows are classified correctly. |
| Hidden optimizer | Sweeps hidden neuron counts and reports the fastest converging size. |

## 5. Self-documentation

- `MathTracer` prints the actual arithmetic of one forward and backward pass.
- `AsciiVisualizer` renders the topology and parameter count.
- `MarkdownReporter` writes run reports to `reports/` (gitignored).
- `ConsoleDisplay` prints matrices and vectors in the layout used by the reference program.

## 6. Packaging

- `.csproj` picks `net8.0-windows` with WinForms on Windows and plain `net8.0` elsewhere, so `dotnet run` works on every platform.
- CI builds on Linux and Windows and smoke-tests the CLI.
- MIT license, `.editorconfig`, `.gitattributes`, and `.gitignore` keep the tree clean.
