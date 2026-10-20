using System;

namespace BackPropagation.Core
{
    /// <summary>
    /// Represents an output layer neuron in the object-oriented network model.
    /// Stores the output bias and the final activated prediction.
    /// </summary>
    public class ONeuron
    {
        private readonly int _id;
        private double _activation;
        private double _bias;

        public int Id => _id;
        public double Activation => _activation;
        public double Bias => _bias;

        /// <summary>
        /// Initializes an output neuron with an identifier and small initial bias.
        /// </summary>
        public ONeuron(int id)
        {
            _id = id;
            _activation = 0.0;
            _bias = 0.01;
        }

        public void SetActivation(double value) => _activation = value;
        public void SetBias(double value) => _bias = value;

        /// <summary>
        /// Updates the output neuron's internal bias:
        /// Δb_k = η * δ_k
        /// b_k = b_k + Δb_k
        /// </summary>
        /// <param name="learningRate">Learning rate for output layer (η_out).</param>
        /// <param name="outputErrorDerivative">Local error gradient signal (δ_k) for this specific output neuron.</param>
        public void UpdateBias(double learningRate, double outputErrorDerivative)
        {
            _bias += learningRate * outputErrorDerivative;
        }
    }
}
