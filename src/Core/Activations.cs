using System;

namespace BackPropagation.Core
{
    /// <summary>
    /// Provides primitive mathematical activation functions and their analytical derivatives.
    /// Implemented using foundational arithmetic and standard System.Math operations without external dependencies.
    /// </summary>
    public static class Activations
    {
        /// <summary>
        /// Logistic Sigmoid activation function:
        /// σ(x) = 1 / (1 + e^(-x))
        /// Maps arbitrary real values into the open interval (0, 1).
        /// </summary>
        /// <param name="x">Weighted sum input to the neuron.</param>
        /// <returns>Activation output in range (0, 1).</returns>
        public static double Sigmoid(double x)
        {
            // Numerical stability clamp to avoid exp overflow
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Analytical derivative of the Logistic Sigmoid expressed with respect to its output y = σ(x):
        /// dσ/dx = σ(x) * (1 - σ(x)) = y * (1 - y)
        /// </summary>
        /// <param name="y">The pre-computed sigmoid activation output y = σ(x).</param>
        /// <returns>The rate of change dσ/dx at that activation level.</returns>
        public static double SigmoidDerivative(double y)
        {
            return y * (1.0 - y);
        }

        /// <summary>
        /// Hyperbolic Tangent activation function:
        /// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        /// Maps arbitrary real values into (-1, 1), with zero-centered outputs.
        /// </summary>
        /// <param name="x">Weighted sum input to the neuron.</param>
        /// <returns>Activation output in range (-1, 1).</returns>
        public static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0;
            if (x > 20.0) return 1.0;
            return Math.Tanh(x);
        }

        /// <summary>
        /// Analytical derivative of Hyperbolic Tangent expressed with respect to output y = tanh(x):
        /// dtanh/dx = 1 - tanh²(x) = (1 + y) * (1 - y)
        /// </summary>
        /// <param name="y">The pre-computed tanh output y = tanh(x).</param>
        /// <returns>The derivative value at that activation.</returns>
        public static double HyperTanDerivative(double y)
        {
            return (1.0 + y) * (1.0 - y);
        }

        /// <summary>
        /// Softmax activation for multi-class categorical outputs:
        /// Softmax(z_k) = e^(z_k) / sum_j(e^(z_j))
        /// Normalizes an array of raw logits into a valid probability distribution where all elements sum to 1.0.
        /// </summary>
        /// <param name="oSums">Array of weighted sums for all output nodes.</param>
        /// <returns>Array of normalized probabilities summing to 1.0.</returns>
        public static double[] Softmax(double[] oSums)
        {
            if (oSums.Length == 0)
                return Array.Empty<double>();

            // Find max for numerical stability (preventing Math.Exp overflow)
            double max = oSums[0];
            for (int i = 1; i < oSums.Length; ++i)
            {
                if (oSums[i] > max) max = oSums[i];
            }

            double sum = 0.0;
            double[] expValues = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
            {
                expValues[i] = Math.Exp(oSums[i] - max);
                sum += expValues[i];
            }

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = expValues[i] / sum;
            }

            return result;
        }
    }
}
