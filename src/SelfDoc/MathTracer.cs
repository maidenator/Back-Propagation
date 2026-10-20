using System;
using BackPropagation.Core;

namespace BackPropagation.SelfDoc
{
    /// <summary>
    /// Autonomous step-by-step mathematical tracer.
    /// Breaks down each mathematical operation during forward and backward passes,
    /// printing intermediate dot products, activation calculations, error derivatives, and weight adjustments.
    /// </summary>
    public static class MathTracer
    {
        public static void TraceSample(NeuralNet net, double[] sampleInputs, double[] desiredOutputs)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("                  MATHEMATICAL STEP-BY-STEP CALCULATION TRACE                  ");
            Console.WriteLine("================================================================================");

            // Set inputs
            for (int i = 0; i < sampleInputs.Length; ++i)
                net.SetInput(i, sampleInputs[i]);

            for (int o = 0; o < desiredOutputs.Length; ++o)
                net.SetDesiredOutput(o, desiredOutputs[o]);

            Console.WriteLine("\n--- [1. FORWARD PROPAGATION] ---");
            Console.WriteLine("Input Vector x: [ " + string.Join(", ", sampleInputs) + " ]");

            // Hidden layer forward pass trace
            Console.WriteLine("\n[Hidden Layer Activations: z_j = sum(x_i * w_ij) + b_j,  h_j = σ(z_j)]");
            for (int h = 0; h < net.HiddenCount; ++h)
            {
                double sum = 0.0;
                string terms = "";
                for (int i = 0; i < net.InputCount; ++i)
                {
                    double x = net.InputNeurons[i].InputValue;
                    double w = net.InputNeurons[i].GetWeight(h);
                    sum += x * w;
                    terms += $"({x:F1} * {w:F4})" + (i < net.InputCount - 1 ? " + " : "");
                }
                double bias = net.HiddenNeurons[h].Bias;
                double netInput = sum + bias;
                double activation = Activations.Sigmoid(netInput);

                Console.WriteLine($"  Node h{h}:");
                Console.WriteLine($"    z_{h} = {terms} + bias({bias:F4}) = {netInput:F4}");
                Console.WriteLine($"    h_{h} = σ({netInput:F4}) = 1 / (1 + e^(-({netInput:F4}))) = {activation:F4}");
            }

            net.CalculateHiddenActivations();

            // Output layer forward pass trace
            Console.WriteLine("\n[Output Layer Activations: z_k = sum(h_j * w_jk) + b_k,  y_k = σ(z_k)]");
            for (int o = 0; o < net.OutputCount; ++o)
            {
                double sum = 0.0;
                string terms = "";
                for (int h = 0; h < net.HiddenCount; ++h)
                {
                    double hAct = net.HiddenNeurons[h].Activation;
                    double w = net.HiddenNeurons[h].GetWeight(o);
                    sum += hAct * w;
                    if (h < 3 || h == net.HiddenCount - 1)
                        terms += $"({hAct:F3} * {w:F3})" + (h < net.HiddenCount - 1 ? " + " : "");
                    else if (h == 3)
                        terms += "... + ";
                }
                double bias = net.OutputNeurons[o].Bias;
                double netInput = sum + bias;
                double activation = Activations.Sigmoid(netInput);

                Console.WriteLine($"  Node o{o}:");
                Console.WriteLine($"    z_{o} = {terms} + bias({bias:F4}) = {netInput:F4}");
                Console.WriteLine($"    y_{o} = σ({netInput:F4}) = {activation:F4} (Target t_{o} = {desiredOutputs[o]:F1})");
            }

            net.CalculateOutputActivations();

            Console.WriteLine("\n--- [2. BACKWARD PROPAGATION & ERROR SIGNALS] ---");
            net.CalculateErrorComponents();
            net.CalculateOutputDerivatives();

            Console.WriteLine("\n[Output Layer Error Signals: e_k = t_k - y_k,  δ_k = y_k * (1 - y_k) * e_k]");
            for (int o = 0; o < net.OutputCount; ++o)
            {
                double y = net.OutputNeurons[o].Activation;
                double err = net.ErrorComponent[o];
                double derivative = y * (1.0 - y);
                double signal = net.ErrorDerivative[o];
                Console.WriteLine($"  Node o{o}:");
                Console.WriteLine($"    Error e_{o} = {desiredOutputs[o]:F1} - {y:F4} = {err:F4}");
                Console.WriteLine($"    σ'(y_{o}) = {y:F4} * (1 - {y:F4}) = {derivative:F4}");
                Console.WriteLine($"    Gradient Signal δ_{o} = {err:F4} * {derivative:F4} = {signal:F6}");
            }

            Console.WriteLine("\n[Hidden Layer Error Signals: δ_j = h_j * (1 - h_j) * sum_k(δ_k * w_jk)]");
            for (int h = 0; h < Math.Min(4, net.HiddenCount); ++h)
            {
                double hAct = net.HiddenNeurons[h].Activation;
                double derivative = hAct * (1.0 - hAct);
                double sum = 0.0;
                for (int o = 0; o < net.OutputCount; ++o)
                    sum += net.ErrorDerivative[o] * net.HiddenNeurons[h].GetWeight(o);
                double hiddenSignal = derivative * sum;

                Console.WriteLine($"  Node h{h}: δ_{h} = ({hAct:F3} * {1.0 - hAct:F3}) * ({sum:F6}) = {hiddenSignal:F6}");
            }
            if (net.HiddenCount > 4) Console.WriteLine($"  ... (and remaining {net.HiddenCount - 4} hidden nodes)");

            Console.WriteLine("\n--- [3. WEIGHT & BIAS UPDATE EQUATIONS] ---");
            Console.WriteLine($"  Output Weights: Δw_jk = η_out({net.LrpOut:F2}) * δ_k * h_j");
            Console.WriteLine($"  Input Weights:  Δw_ij = η_in({net.LrpIn:F2})  * δ_j * x_i");
            Console.WriteLine($"  Output Biases:  Δb_k  = η_out({net.LrpOut:F2}) * δ_k");
            Console.WriteLine($"  Hidden Biases:  Δb_j  = η_in({net.LrpIn:F2})  * δ_j");
            Console.WriteLine("================================================================================\n");
        }
    }
}
