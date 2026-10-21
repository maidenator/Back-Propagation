# CS345 Backpropagation Neural Network

A backpropagation neural network in C#: 4 inputs, 5 hidden nodes (tanh), 3 outputs
(softmax), 43 parameters. No NuGet packages — every forward pass, gradient, and
weight update is written out by hand.

## Run

```bash
git clone https://github.com/maidenator/Back-Propagation.git
cd Back-Propagation
dotnet run --project src/BackPropagation.csproj
```

Requires the .NET 8 SDK. The default run trains three configurations (mean squared
error, cross-entropy, cross-entropy + L2 weight decay) on the same seeded data and
prints a comparison table.

## Results

| Configuration | Weight Norm | Train Acc | Test Acc |
|---|---|---|---|
| Mean squared error | 15.83 | 99.75% | 100.00% |
| Cross-entropy | 45.20 | 95.00% | 93.50% |
| Cross-entropy + weight decay | 3.32 | 90.62% | 87.00% |

## More

`--help` lists every flag: single benchmarks, logic gate experiments, and a Windows
Forms GUI.

Built from Dr. James McCaffrey's CS345 program and modeled on
[eitophetamine9/BackPropProgram](https://github.com/eitophetamine9/BackPropProgram).

MIT license.
