using System;
using BackPropagation.Core;
using BackPropagation.Data;

namespace BackPropagation.Experiments
{
    /// <summary>
    /// Implements the 16-row 4-input boolean logic gate experiment from the Xylphy reference.
    /// Provides train-until-fit convergence logic, manual testing, and automated reporting.
    /// </summary>
    public static class LogicGateExperiment
    {
        public static bool FitTest(NeuralNet net, double[][] dataset)
        {
            for (int r = 0; r < dataset.Length; ++r)
            {
                for (int i = 0; i < net.InputCount; ++i)
                    net.SetInput(i, dataset[r][i]);

                net.Run();
                double rawOut = net.GetOutputData(0);
                int target = (int)dataset[r][net.InputCount];

                if ((int)Math.Round(rawOut) != target)
                    return false;
            }
            return true;
        }

        public static void RunTrainUntilFit(int hiddenNeurons = 20, int maxEpochs = 50000)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine($"         4-INPUT BOOLEAN LOGIC GATE EXPERIMENT (Hidden Neurons: {hiddenNeurons})");
            Console.WriteLine("================================================================================");

            double[][] dataset = LogicGateDataset.Get4InputAndDataset();
            var net = new NeuralNet(4, hiddenNeurons, 1, lrpOut: 0.2, lrpIn: 0.15);

            Console.WriteLine("Initial truth table check before training:");
            PrintVerificationTable(net, dataset);

            Console.WriteLine("\nStarting training loop (training until 100% fit or max epochs)...");
            int epoch = 0;
            bool converged = false;

            while (epoch < maxEpochs)
            {
                ++epoch;

                // Train 1 full epoch across all 16 rows
                for (int r = 0; r < dataset.Length; ++r)
                {
                    for (int i = 0; i < 4; ++i)
                        net.SetInput(i, dataset[r][i]);
                    net.SetDesiredOutput(0, dataset[r][4]);
                    net.Learn();
                }

                if (FitTest(net, dataset))
                {
                    converged = true;
                    Console.WriteLine($"\n>>> SUCCESS! Network converged at epoch {epoch}! All 16 truth table patterns matched perfectly.");
                    break;
                }

                if (epoch % 1000 == 0)
                {
                    Console.WriteLine($"  Epoch {epoch,5}: Still training...");
                }
            }

            if (!converged)
            {
                Console.WriteLine($"\n[!] Reached maximum limit of {maxEpochs} epochs without full convergence.");
            }

            Console.WriteLine("\nFinal truth table verification after training:");
            PrintVerificationTable(net, dataset);


        }

        public static void PrintVerificationTable(NeuralNet net, double[][] dataset)
        {
            Console.WriteLine("┌─────┬───────────────────────────┬────────┬───────────┬─────────┬────────┐");
            Console.WriteLine("│ Row │ Inputs (x0, x1, x2, x3)   │ Target │ Predicted │ Rounded │ Status │");
            Console.WriteLine("├─────┼───────────────────────────┼────────┼───────────┼─────────┼────────┤");

            int correctCount = 0;
            for (int r = 0; r < dataset.Length; ++r)
            {
                for (int i = 0; i < 4; ++i)
                    net.SetInput(i, dataset[r][i]);

                net.Run();
                double rawOut = net.GetOutputData(0);
                int target = (int)dataset[r][4];
                int rounded = (int)Math.Round(rawOut);
                bool match = (rounded == target);
                if (match) ++correctCount;

                string status = match ? " PASS " : " FAIL ";
                Console.WriteLine($"│ {r,3} │ ({dataset[r][0]}, {dataset[r][1]}, {dataset[r][2]}, {dataset[r][3]})               │   {target}    │  {rawOut,6:F4}   │    {rounded}    │ {status} │");
            }

            Console.WriteLine("└─────┴───────────────────────────┴────────┴───────────┴─────────┴────────┘");
            Console.WriteLine($" Accuracy: {correctCount}/{dataset.Length} ({(double)correctCount / dataset.Length * 100.0:F1}%)\n");
        }

        public static void InteractiveManualTest(NeuralNet net)
        {
            Console.WriteLine("\n--- [INTERACTIVE MANUAL TESTING] ---");
            Console.WriteLine("Enter 4 binary values (0 or 1) separated by spaces (e.g. '1 1 1 1'):");
            Console.Write("> ");
            string? line = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) return;

            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 4)
            {
                Console.WriteLine("Error: Exactly 4 inputs required.");
                return;
            }

            double[] inputs = new double[4];
            for (int i = 0; i < 4; ++i)
            {
                if (!double.TryParse(parts[i], out inputs[i]))
                {
                    Console.WriteLine($"Error: '{parts[i]}' is not a valid numeric input.");
                    return;
                }
                net.SetInput(i, inputs[i]);
            }

            net.Run();
            double output = net.GetOutputData(0);
            Console.WriteLine($"Input: [ {string.Join(", ", inputs)} ]");
            Console.WriteLine($"Raw Continuous Output:   {output:F6}");
            Console.WriteLine($"Classified Discrete (0/1): {Math.Round(output)}");
        }
    }
}
