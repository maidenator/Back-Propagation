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

            const int cellWidth = 16;
            const string leading = "   ";
            const string gap = "  ──>  ";
            string horizontal = new string('─', cellWidth);

            Console.WriteLine(
                $"{leading}{Center($"Input Layer ({numInput})", cellWidth + 2)}{gap}" +
                $"{Center($"Hidden Layer ({numHidden})", cellWidth + 2)}{gap}" +
                $"{Center($"Output Layer ({numOutput})", cellWidth + 2)}");
            Console.WriteLine(
                $"{leading}┌{horizontal}┐     ┌{horizontal}┐     ┌{horizontal}┐");

            int maxRows = Math.Max(numInput, Math.Max(numHidden, numOutput));
            bool truncated = maxRows > 8;
            int rowsToShow = Math.Min(maxRows, 8);

            for (int r = 0; r < rowsToShow; ++r)
            {
                string inLabel = r < numInput ? $"Node x{r}" : string.Empty;
                string hidLabel = r < numHidden ? $"Node h{r}" : string.Empty;
                string outLabel = r < numOutput ? $"Node y{r}" : string.Empty;

                Console.WriteLine(
                    $"{leading}│{Center(inLabel, cellWidth)}│{gap}" +
                    $"│{Center(hidLabel, cellWidth)}│{gap}" +
                    $"│{Center(outLabel, cellWidth)}│");
            }

            if (truncated)
            {
                Console.WriteLine(
                    $"{leading}│{Center("...", cellWidth)}│{gap}" +
                    $"│{Center("...", cellWidth)}│{gap}" +
                    $"│{Center("...", cellWidth)}│");
            }

            Console.WriteLine(
                $"{leading}└{horizontal}┘     └{horizontal}┘     └{horizontal}┘");

            int totalConnections = (numInput * numHidden) + (numHidden * numOutput);
            int totalBiases = numHidden + numOutput;
            Console.WriteLine($" Total Trainable Parameters: {totalConnections + totalBiases} ({totalConnections} weights + {totalBiases} biases)");
            Console.WriteLine("================================================================================\n");
        }

        private static string Center(string text, int width)
        {
            if (text.Length >= width)
                return text.Substring(0, width);

            int leftPadding = (width - text.Length) / 2;
            return new string(' ', leftPadding) + text + new string(' ', width - text.Length - leftPadding);
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
