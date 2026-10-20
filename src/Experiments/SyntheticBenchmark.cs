using System;
using BackPropagation.Core;
using BackPropagation.Data;
using BackPropagation.SelfDoc;

namespace BackPropagation.Experiments
{
    /// <summary>
    /// Replicates the synthetic classification benchmark from Dr. James McCaffrey's CS345 program.
    /// Employs a 4-5-3 network, Tanh hidden activation, Softmax output layer, and momentum gradient descent.
    /// </summary>
    public static class SyntheticBenchmark
    {
        public static void RunDemo()
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("            CS345 IS 1: 4-5-3 SYNTHETIC CLASSIFIER DEMO                         ");
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
            Console.WriteLine("\nStarting backpropagation training with momentum...");

            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum, (epoch, err) =>
            {
                Console.WriteLine($"  Epoch {epoch,4} | Mean Squared Error = {err:F4}");
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
            Console.WriteLine("\n================================================================================\n");
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
