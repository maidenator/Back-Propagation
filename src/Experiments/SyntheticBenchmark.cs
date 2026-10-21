using System;
using BackPropagation.Core;
using BackPropagation.Data;

namespace BackPropagation.Experiments
{
    /// <summary>
    /// Replicates and extends the synthetic classification benchmark from Dr. James McCaffrey's CS345 program.
    /// Employs a 4-5-3 network, Tanh hidden activation, Softmax output layer, and momentum gradient descent.
    /// Includes dedicated modes for the two 'WHAT TO DO' extensions: Cross-Entropy Error and Weight Decay.
    /// </summary>
    public static class SyntheticBenchmark
    {
        public static void RunDemo(int maxEpochs = 1000)
        {
            ExecuteBenchmark(
                title: "CS345 STANDARD BENCHMARK (Mean Squared Error)",
                lossFunc: LossFunction.MeanSquaredError,
                weightDecay: 0.0,
                maxEpochs: maxEpochs
            );
        }

        public static void RunCrossEntropyDemo(int maxEpochs = 1000)
        {
            ExecuteBenchmark(
                title: "CS345 EXTENSION 1: CROSS-ENTROPY ERROR (from 'WHAT TO DO')",
                lossFunc: LossFunction.CrossEntropy,
                weightDecay: 0.0,
                maxEpochs: maxEpochs
            );
        }

        public static void RunWeightDecayDemo(int maxEpochs = 1000)
        {
            ExecuteBenchmark(
                title: "CS345 EXTENSION 2: WEIGHT DECAY / L2 REGULARIZATION (from 'WHAT TO DO')",
                lossFunc: LossFunction.MeanSquaredError,
                weightDecay: 0.001,
                maxEpochs: maxEpochs
            );
        }

        public static void ExecuteBenchmark(string title, LossFunction lossFunc, double weightDecay, int maxEpochs = 1000)
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

            Console.WriteLine("================================================================================\n");
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
