using System;
using BackPropagation.Core;
using BackPropagation.Data;
using BackPropagation.SelfDoc;

namespace BackPropagation.Experiments
{
    /// <summary>
    /// Runs the three backpropagation configurations of the CS345 study in a single pass and
    /// prints a comparison table, mirroring the reference program by eitophetamine9:
    /// 1. Baseline       - Mean Squared Error, no weight decay.
    /// 2. Cross-Entropy  - Cross-Entropy loss with Softmax, no weight decay.
    /// 3. Regularized   - Cross-Entropy loss plus L2 weight decay (lambda = 0.001).
    /// </summary>
    public static class ComparisonRunner
    {
        private const int NumInput = 4;
        private const int NumHidden = 5;
        private const int NumOutput = 3;
        private const int NumRows = 1000;
        private const int Seed = 1;
        private const double TrainPct = 0.80;
        private const double LearnRate = 0.05;
        private const double Momentum = 0.01;
        private const double DecayRate = 0.001;

        private sealed class RunResult
        {
            public string Label = "";
            public string LossLabel = "";
            public double Loss;
            public double WeightNorm;
            public double TrainAccuracy;
            public double TestAccuracy;
        }

        /// <summary>
        /// Trains all three configurations on the same seeded dataset and prints a comparison table.
        /// </summary>
        /// <param name="maxEpochs">Number of training epochs per configuration.</param>
        public static void RunComparison(int maxEpochs = 1000)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("  NEURAL NETWORK BACK-PROPAGATION COMPARISON (CS345)");
            Console.WriteLine("================================================================================");
            Console.WriteLine($"Architecture : {NumInput}-{NumHidden}-{NumOutput} ({NumInput * NumHidden + NumHidden + NumHidden * NumOutput + NumOutput} parameters)");
            Console.WriteLine($"Dataset      : {NumRows} synthetic items, seed {Seed}, {TrainPct * 100:F0}/{100 - TrainPct * 100:F0} train/test split");
            Console.WriteLine($"Hyperparams  : maxEpochs = {maxEpochs}, learnRate = {LearnRate:F2}, momentum = {Momentum:F2}, decayRate = {DecayRate:F4}");

            Console.WriteLine($"\nGenerating {NumRows} artificial data items with {NumInput} features...");
            double[][] allData = SyntheticDataset.MakeAllData(NumInput, NumHidden, NumOutput, NumRows, Seed);
            Console.WriteLine("Done");

            Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
            SyntheticDataset.SplitTrainTest(allData, TrainPct, Seed, out double[][] trainData, out double[][] testData);
            Console.WriteLine("Done\n");

            Console.WriteLine("Training data:");
            ConsoleDisplay.ShowMatrix(trainData, 4, 2, true);
            Console.WriteLine("Test data:");
            ConsoleDisplay.ShowMatrix(testData, 4, 2, true);

            RunResult baseline = Train("EXPERIMENT 1: Baseline (Mean Squared Error, No Decay)",
                "1. Baseline (MSE, No Decay)", "MSE", LossFunction.MeanSquaredError, 0.0,
                trainData, testData, maxEpochs);

            RunResult crossEntropy = Train("EXPERIMENT 2: Cross-Entropy Error (No Decay)",
                "2. Cross-Entropy (No Decay)", "Cross-Entropy", LossFunction.CrossEntropy, 0.0,
                trainData, testData, maxEpochs);

            RunResult regularized = Train("EXPERIMENT 3: Cross-Entropy WITH Weight Decay (Regularization)",
                "3. Cross-Entropy + Weight Decay", "Cross-Entropy", LossFunction.CrossEntropy, DecayRate,
                trainData, testData, maxEpochs);

            PrintSummary(baseline, crossEntropy, regularized);
        }

        private static RunResult Train(string title, string label, string lossLabel,
            LossFunction lossFunction, double weightDecay, double[][] trainData, double[][] testData,
            int maxEpochs)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine($" {title}");
            Console.WriteLine("================================================================================");
            Console.WriteLine($"Creating a {NumInput}-{NumHidden}-{NumOutput} neural network");
            Console.WriteLine($"\nSetting maxEpochs = {maxEpochs}");
            Console.WriteLine($"Setting learnRate = {LearnRate:F2}");
            Console.WriteLine($"Setting momentum  = {Momentum:F2}");
            Console.WriteLine($"Setting errorType = {lossLabel}");
            Console.WriteLine($"Setting decayRate = {(weightDecay > 0.0 ? weightDecay.ToString("F4") : "0.00 (No Weight Decay)")}");

            Console.WriteLine("\nStarting training");
            var nn = new MatrixNeuralNet(NumInput, NumHidden, NumOutput, Seed);
            double[] weights = nn.Train(trainData, maxEpochs, LearnRate, Momentum, lossFunction, weightDecay,
                (epoch, loss) => Console.WriteLine($"epoch = {epoch}  error = {loss:F4}"));
            Console.WriteLine("Done");
            Console.WriteLine("\nFinal neural network model weights and biases:\n");
            ConsoleDisplay.ShowVector(weights, 2, 10, true);

            var result = new RunResult
            {
                Label = label,
                LossLabel = lossLabel,
                Loss = (lossFunction == LossFunction.CrossEntropy)
                    ? nn.CrossEntropyError(trainData)
                    : nn.MeanSquaredError(trainData),
                WeightNorm = nn.WeightNorm(),
                TrainAccuracy = nn.Accuracy(trainData),
                TestAccuracy = nn.Accuracy(testData)
            };

            Console.WriteLine($"\nFinal {lossLabel} on training data = {result.Loss:F4}");
            Console.WriteLine($"Final weight norm (L2 norm)     = {result.WeightNorm:F4}");
            Console.WriteLine($"Final accuracy on training data = {result.TrainAccuracy:F4}");
            Console.WriteLine($"Final accuracy on test data     = {result.TestAccuracy:F4}");
            Console.WriteLine();

            return result;
        }

        private static void PrintSummary(RunResult baseline, RunResult crossEntropy, RunResult regularized)
        {
            Console.WriteLine("==========================================================================");
            Console.WriteLine(" COMPARISON OF BACK-PROPAGATION CONFIGURATIONS");
            Console.WriteLine("==========================================================================");
            Console.WriteLine("{0,-32} {1,-12} {2,-14} {3,-12} {4,-12}",
                "Configuration", "Train Loss", "Weight Norm", "Train Acc", "Test Acc");
            Console.WriteLine(new string('-', 84));
            WriteRow(baseline);
            WriteRow(crossEntropy);
            WriteRow(regularized);
            Console.WriteLine(new string('-', 84));

            Console.WriteLine("\nKey Observations:");
            Console.WriteLine("  1. Cross-Entropy avoids the softmax derivative saturation term (1-y)*y,");
            Console.WriteLine("     yielding faster learning signals for classification.");
            Console.WriteLine($"  2. Weight Decay penalizes large weights, significantly reducing the L2 weight norm");
            Console.WriteLine($"     ({crossEntropy.WeightNorm:F2} -> {regularized.WeightNorm:F2}), helping prevent over-fitting.");
            Console.WriteLine("==========================================================================\n");
        }

        private static void WriteRow(RunResult result)
        {
            Console.WriteLine("{0,-32} {1,-12:F4} {2,-14:F4} {3,-12:P2} {4,-12:P2}",
                result.Label, result.Loss, result.WeightNorm, result.TrainAccuracy, result.TestAccuracy);
        }
    }
}
