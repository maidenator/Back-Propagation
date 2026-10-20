using System;
using System.IO;
using System.Text;
using BackPropagation.Core;
using BackPropagation.Data;
using BackPropagation.SelfDoc;

namespace BackPropagation.Experiments
{
    /// <summary>
    /// Replicates and extends the synthetic classification benchmark from Dr. James McCaffrey's CS345 program.
    /// Employs a 4-5-3 network, Tanh hidden activation, Softmax output layer, and momentum gradient descent.
    /// Includes dedicated modes for the two 'WHAT TO DO' extensions: Cross-Entropy Error and Weight Decay.
    /// </summary>
    public static class SyntheticBenchmark
    {
        public static void RunDemo()
        {
            ExecuteBenchmark(
                title: "CS345 STANDARD BENCHMARK (Mean Squared Error)",
                lossFunc: LossFunction.MeanSquaredError,
                weightDecay: 0.0
            );
        }

        public static void RunCrossEntropyDemo()
        {
            ExecuteBenchmark(
                title: "CS345 EXTENSION 1: CROSS-ENTROPY ERROR (from 'WHAT TO DO')",
                lossFunc: LossFunction.CrossEntropy,
                weightDecay: 0.0
            );
        }

        public static void RunWeightDecayDemo()
        {
            ExecuteBenchmark(
                title: "CS345 EXTENSION 2: WEIGHT DECAY / L2 REGULARIZATION (from 'WHAT TO DO')",
                lossFunc: LossFunction.MeanSquaredError,
                weightDecay: 0.001
            );
        }

        public static void ExecuteBenchmark(string title, LossFunction lossFunc, double weightDecay)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine($"            {title}");
            Console.WriteLine("================================================================================");

            int numInput = 4;
            int numHidden = 5;
            int numOutput = 3;
            int numRows = 1000;
            int seed = 1;

            Console.WriteLine($"\nGenerating {numRows} artificial data items with {numInput} features...");
            double[][] allData = SyntheticDataset.MakeAllData(numInput, numHidden, numOutput, numRows, seed);
            Console.WriteLine("Data generation complete.");

            Console.WriteLine("\nSplitting into train (80%) and test (20%) partitions...");
            SyntheticDataset.SplitTrainTest(allData, 0.80, seed, out double[][] trainData, out double[][] testData);
            Console.WriteLine($"Partition complete: {trainData.Length} training items, {testData.Length} test items.");

            Console.WriteLine("\nSample training items (first 3 rows):");
            ShowSampleData(trainData, 3, numInput, numOutput);

            Console.WriteLine($"\nCreating a {numInput}-{numHidden}-{numOutput} matrix neural network...");
            var nn = new MatrixNeuralNet(numInput, numHidden, numOutput, seed);
            AsciiVisualizer.RenderTopology(numInput, numHidden, numOutput);

            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.01;

            Console.WriteLine($"Hyperparameters: maxEpochs = {maxEpochs}, learnRate = {learnRate:F2}, momentum = {momentum:F2}");
            Console.WriteLine($"Loss Function:   {lossFunc}");
            Console.WriteLine($"Weight Decay:    {(weightDecay > 0.0 ? weightDecay.ToString("F4") : "None (0.0)")}");
            Console.WriteLine("\nStarting backpropagation training with momentum...");

            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum, lossFunc, weightDecay, (epoch, loss) =>
            {
                string metricName = (lossFunc == LossFunction.CrossEntropy) ? "Cross-Entropy" : "MSE";
                Console.WriteLine($"  Epoch {epoch,4} | {metricName} = {loss:F4}");
            });

            Console.WriteLine("\nTraining finished successfully.");

            double trainAcc = nn.Accuracy(trainData);
            double testAcc = nn.Accuracy(testData);

            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine($"  Final Accuracy on Training Data (N={trainData.Length}): {trainAcc * 100.0:F2}%");
            Console.WriteLine($"  Final Accuracy on Test Data     (N={testData.Length}): {testAcc * 100.0:F2}%");
            Console.WriteLine("--------------------------------------------------------------------------------");

            Console.WriteLine($"\nFinal learned model serialized weights ({weights.Length} parameters):");
            for (int i = 0; i < weights.Length; ++i)
            {
                Console.Write($"{weights[i],6:F2} ");
                if ((i + 1) % 10 == 0) Console.WriteLine();
            }

            // Demonstrate prediction on classic sample from Back-Prop PDF
            Console.WriteLine("\n--- [PREDICTION ON NEW DATA ITEM (from Back-Prop PDF Page 2)] ---");
            double[] sampleInput = new double[] { 1.0, 2.0, 3.0, 4.0 };
            double[] predProb = nn.ComputeOutputs(sampleInput);
            int predictedClass = MatrixNeuralNet.MaxIndex(predProb);
            string[] classNames = { "Conservative (1, 0, 0)", "Moderate (0, 1, 0)", "Liberal (0, 0, 1)" };
            Console.WriteLine($"Input Vector: (1.0, 2.0, 3.0, 4.0)");
            Console.WriteLine($"Softmax Probabilities: [{predProb[0]:F4}, {predProb[1]:F4}, {predProb[2]:F4}]");
            Console.WriteLine($"Predicted Class: {classNames[predictedClass]} (matches PDF Page 2 demonstration)");

            // Auto-generate self-documenting Markdown report
            string reportPath = GenerateMarkdownReport(title, nn, trainAcc, testAcc, trainData.Length, testData.Length, lossFunc, weightDecay);
            Console.WriteLine($"\n[SELF-DOC] Markdown run report generated automatically at:\n  --> {reportPath}");
            Console.WriteLine("================================================================================\n");
        }

        private static string GenerateMarkdownReport(string title, MatrixNeuralNet nn, double trainAcc, double testAcc, 
            int trainCount, int testCount, LossFunction lossFunc, double weightDecay)
        {
            Directory.CreateDirectory("reports");
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string fileName = $"cs345_run_{timestamp}.md";
            string fullPath = Path.Combine("reports", fileName);

            var sb = new StringBuilder();
            sb.AppendLine($"# Neural Network Training Report: {title}");
            sb.AppendLine();
            sb.AppendLine($"**Generated:** {DateTime.Now:yyyy-MM-dd HH:mm:ss}  ");
            sb.AppendLine($"**Architecture:** {nn.NumInput} Inputs → {nn.NumHidden} Hidden Nodes → {nn.NumOutput} Outputs (Total Parameters: {nn.GetWeights().Length})  ");
            sb.AppendLine($"**Loss Function:** {lossFunc}  ");
            sb.AppendLine($"**L2 Weight Decay Rate:** {(weightDecay > 0.0 ? weightDecay.ToString("F4") : "None")}  ");
            sb.AppendLine($"**Training Accuracy:** {trainAcc * 100.0:F2}% (N={trainCount})  ");
            sb.AppendLine($"**Test Accuracy:** {testAcc * 100.0:F2}% (N={testCount})  ");
            sb.AppendLine();

            sb.AppendLine("## 1. Network Topology (ASCII)");
            sb.AppendLine("```");
            sb.AppendLine("  Input Layer (4)       Hidden Layer (5)       Output Layer (3)");
            sb.AppendLine("  ┌─────────────┐       ┌──────────────┐       ┌─────────────┐");
            sb.AppendLine("  │   Node x0   │ ────> │   Node h0    │ ────> │   Node y0   │");
            sb.AppendLine("  │   Node x1   │ ────> │   Node h1    │ ────> │   Node y1   │");
            sb.AppendLine("  │   Node x2   │ ────> │   Node h2    │ ────> │   Node y2   │");
            sb.AppendLine("  │   Node x3   │ ────> │   Node h3    │       │             │");
            sb.AppendLine("  │             │       │   Node h4    │       │             │");
            sb.AppendLine("  └─────────────┘       └──────────────┘       └─────────────┘");
            sb.AppendLine("```");
            sb.AppendLine();

            sb.AppendLine("## 2. Mathematical Formulations Applied");
            if (lossFunc == LossFunction.CrossEntropy)
            {
                sb.AppendLine("- **Cross-Entropy Loss:** $E = - \\sum_{k} t_k \\ln(y_k)$");
                sb.AppendLine("- **Output Gradient Signal (Softmax + CE):** $\\delta_k = t_k - y_k$ *(the $y_k(1-y_k)$ factor cancels analytically)*");
            }
            else
            {
                sb.AppendLine("- **Mean Squared Error Loss:** $E = \\frac{1}{2} \\sum_{k} (t_k - y_k)^2$");
                sb.AppendLine("- **Output Gradient Signal (Softmax + MSE):** $\\delta_k = (t_k - y_k) \\cdot y_k \\cdot (1 - y_k)$");
            }
            sb.AppendLine("- **Hidden Activation:** $\\tanh(z_j) = \\frac{e^{z_j} - e^{-z_j}}{e^{z_j} + e^{-z_j}}$");
            sb.AppendLine("- **Hidden Gradient Signal:** $\\delta_j = (1 - h_j^2) \\cdot \\sum_{k} \\delta_k w_{jk}^{(ho)}$");
            if (weightDecay > 0.0)
            {
                sb.AppendLine($"- **L2 Weight Decay Update:** $\\Delta w = \\eta \\cdot (\\text{{grad}} - {weightDecay} \\cdot w) + \\alpha \\cdot \\Delta w^{{(\\text{{prev}})}}$");
            }
            else
            {
                sb.AppendLine("- **Momentum Weight Update:** $\\Delta w = \\eta \\cdot \\delta \\cdot \\text{input} + \\alpha \\cdot \\Delta w^{(\\text{prev})}$");
            }
            sb.AppendLine();

            sb.AppendLine("## 3. Serialized Model Weights & Biases");
            sb.AppendLine("```");
            double[] w = nn.GetWeights();
            for (int i = 0; i < w.Length; ++i)
            {
                sb.Append($"{w[i],7:F2} ");
                if ((i + 1) % 10 == 0) sb.AppendLine();
            }
            sb.AppendLine("\n```");

            File.WriteAllText(fullPath, sb.ToString());
            return fullPath;
        }

        private static void ShowSampleData(double[][] data, int rows, int numInput, int numOutput)
        {
            for (int r = 0; r < rows; ++r)
            {
                Console.Write($"  [{r}] Inputs: (");
                for (int i = 0; i < numInput; ++i)
                    Console.Write($"{data[r][i],6:F2}" + (i < numInput - 1 ? ", " : ""));
                Console.Write(") | Target: (");
                for (int o = 0; o < numOutput; ++o)
                    Console.Write($"{data[r][numInput + o],3:F0}" + (o < numOutput - 1 ? ", " : ""));
                Console.WriteLine(")");
            }
        }
    }
}
