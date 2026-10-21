using System;
using BackPropagation.Core;

namespace BackPropagation.Data
{
    /// <summary>
    /// Generates and partitions the synthetic classification dataset from Dr. James McCaffrey's CS345 specification.
    /// Creates reproducible artificial data using a seeded generator neural network with 1-of-N one-hot encoded targets.
    /// </summary>
    public static class SyntheticDataset
    {
        /// <summary>
        /// Generates artificial dataset where inputs are in [-10.0, 10.0] and targets are 1-of-N encoded.
        /// </summary>
        public static double[][] MakeAllData(int numInput, int numHidden, int numOutput, int numRows, int seed)
        {
            if (numRows <= 0)
                throw new ArgumentOutOfRangeException(nameof(numRows), "The dataset must contain at least one row.");

            var rnd = new Random(seed);
            int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
            double[] weights = new double[numWeights];
            for (int i = 0; i < numWeights; ++i)
                weights[i] = (20.0 * rnd.NextDouble()) - 10.0;

            var gnn = new MatrixNeuralNet(numInput, numHidden, numOutput, seed);
            gnn.SetWeights(weights);

            double[][] result = new double[numRows][];
            for (int r = 0; r < numRows; ++r)
            {
                result[r] = new double[numInput + numOutput];
                double[] inputs = new double[numInput];
                for (int i = 0; i < numInput; ++i)
                    inputs[i] = (20.0 * rnd.NextDouble()) - 10.0;

                double[] outputs = gnn.ComputeOutputs(inputs);

                // Convert to 1-of-N encoding
                int maxIndex = MatrixNeuralNet.MaxIndex(outputs);
                for (int i = 0; i < numInput; ++i)
                    result[r][i] = inputs[i];

                for (int k = 0; k < numOutput; ++k)
                    result[r][numInput + k] = (k == maxIndex) ? 1.0 : 0.0;
            }

            return result;
        }

        /// <summary>
        /// Splits dataset into training and testing partitions using Fisher-Yates scrambling.
        /// </summary>
        public static void SplitTrainTest(double[][] allData, double trainPct, int seed,
            out double[][] trainData, out double[][] testData)
        {
            if (allData.Length == 0)
                throw new ArgumentException("Dataset must contain at least one row.", nameof(allData));
            if (trainPct <= 0.0 || trainPct >= 1.0)
                throw new ArgumentOutOfRangeException(nameof(trainPct), "Training percentage must be between 0.0 and 1.0 (exclusive).");

            var rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct);
            int numTestRows = totRows - numTrainRows;

            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[totRows][];
            for (int i = 0; i < totRows; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i)
            {
                int r = rnd.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }

            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        }

        /// <summary>
        /// Prints the first <paramref name="numRows"/> rows of a matrix, then an ellipsis row.
        /// </summary>
        public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool showIndices)
        {
            int len = matrix.Length.ToString().Length;

            for (int i = 0; i < numRows; ++i)
            {
                if (showIndices)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");

                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" ");
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine();
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (showIndices)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");

                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0)
                        Console.Write(" ");
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }

            Console.WriteLine();
            Console.WriteLine();
        }

        /// <summary>
        /// Prints a vector, wrapping after <paramref name="lineLen"/> values.
        /// </summary>
        public static void ShowVector(double[] vector, int decimals, int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0)
                    Console.WriteLine();

                if (vector[i] >= 0.0)
                    Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }

            if (newLine)
                Console.WriteLine();
        }
    }
}
