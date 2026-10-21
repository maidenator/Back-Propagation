using System;

namespace BackPropagation.SelfDoc
{
    /// <summary>
    /// Console display helpers for matrices and vectors.
    /// Mirrors the ShowMatrix / ShowVector helpers of the CS345 reference program.
    /// </summary>
    public static class ConsoleDisplay
    {
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
