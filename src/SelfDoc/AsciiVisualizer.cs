using System;
using BackPropagation.Core;

namespace BackPropagation.SelfDoc
{
    /// <summary>
    /// Renders ASCII terminal visualizations of neural network topologies and weights.
    /// Provides immediate structural insight directly in console environments.
    /// </summary>
    public static class AsciiVisualizer
    {
        public static void RenderTopology(int numInput, int numHidden, int numOutput)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("                      NEURAL NETWORK TOPOLOGY ARCHITECTURE                      ");
            Console.WriteLine("================================================================================");
            Console.WriteLine($"    Input Layer ({numInput})       Hidden Layer ({numHidden})       Output Layer ({numOutput})");
            Console.WriteLine("    ┌─────────────┐             ┌──────────────┐             ┌─────────────┐");

            int maxRows = Math.Max(numInput, Math.Max(numHidden, numOutput));
            maxRows = Math.Min(maxRows, 8); // Keep compact for console display

            for (int r = 0; r < maxRows; ++r)
            {
                string inNode = (r < numInput) ? $"  │  Node x{r}   │" : "  │             │";
                if (r == maxRows - 1 && numInput > maxRows) inNode = "  │   ...       │";

                string inToHidArrow = (r < numInput && r < numHidden) ? " ────> " : "       ";

                string hidNode = (r < numHidden) ? $"  │   Node h{r}  │" : "  │              │";
                if (r == maxRows - 1 && numHidden > maxRows) hidNode = $"  │ ... (+{numHidden - maxRows + 1}) │";

                string hidToOutArrow = (r < numHidden && r < numOutput) ? " ────> " : "       ";

                string outNode = (r < numOutput) ? $"  │  Node y{r}   │" : "  │             │";
                if (r == maxRows - 1 && numOutput > maxRows) outNode = "  │   ...       │";

                Console.WriteLine($"{inNode}{inToHidArrow}{hidNode}{hidToOutArrow}{outNode}");
            }

            Console.WriteLine("    └─────────────┘             └──────────────┘             └─────────────┘");
            int totalConnections = (numInput * numHidden) + (numHidden * numOutput);
            int totalBiases = numHidden + numOutput;
            Console.WriteLine($" Total Trainable Parameters: {totalConnections + totalBiases} ({totalConnections} weights + {totalBiases} biases)");
            Console.WriteLine("================================================================================\n");
        }

        public static void DisplayWeightSummary(NeuralNet net)
        {
            Console.WriteLine("--- [WEIGHT & BIAS MATRIX SUMMARY] ---");
            Console.WriteLine($"Input-to-Hidden Weights ({net.InputCount}x{net.HiddenCount}):");
            for (int i = 0; i < net.InputCount; ++i)
            {
                Console.Write($"  x{i} -> [ ");
                for (int h = 0; h < Math.Min(6, net.HiddenCount); ++h)
                    Console.Write($"{net.InputNeurons[i].GetWeight(h),6:F2} ");
                if (net.HiddenCount > 6) Console.Write($"... (+{net.HiddenCount - 6})");
                Console.WriteLine("]");
            }

            Console.WriteLine($"\nHidden Biases ({net.HiddenCount}): [ " +
                string.Join(", ", Array.ConvertAll(net.HiddenNeurons, h => h.Bias.ToString("F3"))) + " ]");

            Console.WriteLine($"\nHidden-to-Output Weights ({net.HiddenCount}x{net.OutputCount}):");
            for (int h = 0; h < Math.Min(6, net.HiddenCount); ++h)
            {
                Console.Write($"  h{h} -> [ ");
                for (int o = 0; o < net.OutputCount; ++o)
                    Console.Write($"{net.HiddenNeurons[h].GetWeight(o),6:F2} ");
                Console.WriteLine("]");
            }
            if (net.HiddenCount > 6) Console.WriteLine($"  ... ({net.HiddenCount - 6} more hidden nodes)");

            Console.WriteLine($"\nOutput Biases ({net.OutputCount}): [ " +
                string.Join(", ", Array.ConvertAll(net.OutputNeurons, o => o.Bias.ToString("F3"))) + " ]");
            Console.WriteLine("---------------------------------------\n");
        }
    }
}
