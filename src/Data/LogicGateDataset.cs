using System;

namespace BackPropagation.Data
{
    /// <summary>
    /// Generates 4-input binary truth table datasets (16 possible state permutations).
    /// Directly matches the truth table configuration used in the Xylphy reference implementation.
    /// </summary>
    public static class LogicGateDataset
    {
        /// <summary>
        /// Returns the exact 16-row 4-input truth table from Form1.cs in the reference repository.
        /// Represents a 4-input logical AND gate (outputs 1 if and only if all four inputs are 1).
        /// Format per row: [x0, x1, x2, x3, desired_y]
        /// </summary>
        public static double[][] Get4InputAndDataset()
        {
            return new double[16][]
            {
                new double[] { 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 1, 0 },
                new double[] { 0, 0, 1, 0, 0 },
                new double[] { 0, 0, 1, 1, 0 },
                new double[] { 0, 1, 0, 0, 0 },
                new double[] { 0, 1, 0, 1, 0 },
                new double[] { 0, 1, 1, 0, 0 },
                new double[] { 0, 1, 1, 1, 0 },
                new double[] { 1, 0, 0, 0, 0 },
                new double[] { 1, 0, 0, 1, 0 },
                new double[] { 1, 0, 1, 0, 0 },
                new double[] { 1, 0, 1, 1, 0 },
                new double[] { 1, 1, 0, 0, 0 },
                new double[] { 1, 1, 0, 1, 0 },
                new double[] { 1, 1, 1, 0, 0 },
                new double[] { 1, 1, 1, 1, 1 }
            };
        }

        /// <summary>
        /// Generates an exhaustive 4-input Parity / XOR dataset:
        /// Output is 1 if an odd number of inputs are 1; otherwise 0.
        /// This is a non-linearly separable problem requiring hidden layer representations.
        /// </summary>
        public static double[][] Get4InputParityDataset()
        {
            double[][] data = new double[16][];
            for (int i = 0; i < 16; ++i)
            {
                double x0 = (i & 8) != 0 ? 1.0 : 0.0;
                double x1 = (i & 4) != 0 ? 1.0 : 0.0;
                double x2 = (i & 2) != 0 ? 1.0 : 0.0;
                double x3 = (i & 1) != 0 ? 1.0 : 0.0;
                double y = ((int)(x0 + x1 + x2 + x3) % 2 == 1) ? 1.0 : 0.0;
                data[i] = new double[] { x0, x1, x2, x3, y };
            }
            return data;
        }
    }
}
