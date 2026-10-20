using System;
using BackPropagation.Core;
using BackPropagation.Data;

namespace BackPropagation.Experiments
{
    /// <summary>
    /// Replicates the hidden neuron optimization algorithms from buttons 4 & 5 of the reference implementation.
    /// Empirically evaluates how varying the hidden neuron count influences convergence speed and feasibility.
    /// </summary>
    public static class HiddenOptimizer
    {
        public static void RunOptimization(int minNeurons = 1, int maxNeurons = 25, int epochLimit = 10000)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("              HIDDEN NEURON COUNT OPTIMIZATION BENCHMARK                        ");
            Console.WriteLine("================================================================================");
            Console.WriteLine($"Evaluating hidden layer sizes from {minNeurons} to {maxNeurons} neurons (Max {epochLimit} epochs per run)...");

            double[][] dataset = LogicGateDataset.Get4InputAndDataset();

            Console.WriteLine("\n┌─────────────────┬──────────────────────┬───────────────────────┐");
            Console.WriteLine("│ Hidden Neurons  │ Status               │ Epochs to Converge    │");
            Console.WriteLine("├─────────────────┼──────────────────────┼───────────────────────┤");

            int bestNeurons = -1;
            int minEpochs = int.MaxValue;

            for (int h = minNeurons; h <= maxNeurons; ++h)
            {
                var net = new NeuralNet(4, h, 1, lrpOut: 0.2, lrpIn: 0.15, rnd: new Random(100 + h));
                int epoch = 0;
                bool converged = false;

                while (epoch < epochLimit)
                {
                    ++epoch;

                    for (int r = 0; r < dataset.Length; ++r)
                    {
                        for (int i = 0; i < 4; ++i)
                            net.SetInput(i, dataset[r][i]);
                        net.SetDesiredOutput(0, dataset[r][4]);
                        net.Learn();
                    }

                    if (LogicGateExperiment.FitTest(net, dataset))
                    {
                        converged = true;
                        break;
                    }
                }

                if (converged)
                {
                    Console.WriteLine($"│ {h,15} │ Converged            │ {epoch,21} │");
                    if (epoch < minEpochs)
                    {
                        minEpochs = epoch;
                        bestNeurons = h;
                    }
                }
                else
                {
                    Console.WriteLine($"│ {h,15} │ Timed Out (>{epochLimit})  │ {"N/A",21} │");
                }
            }

            Console.WriteLine("└─────────────────┴──────────────────────┴───────────────────────┘");
            if (bestNeurons != -1)
            {
                Console.WriteLine($"\n>>> OPTIMAL RESULT: {bestNeurons} hidden neurons converged fastest (in {minEpochs} epochs).");
            }
            else
            {
                Console.WriteLine("\n[!] No configuration converged within the epoch limit.");
            }
            Console.WriteLine("================================================================================\n");
        }
    }
}
