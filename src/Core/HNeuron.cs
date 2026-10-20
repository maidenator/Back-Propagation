using System;

namespace BackPropagation.Core
{
    /// <summary>
    /// Represents a hidden layer neuron in the object-oriented network model.
    /// Stores outgoing weights to output neurons, hidden bias, forward activation, and backward error gradient signal.
    /// </summary>
    public class HNeuron
    {
        private readonly int _id;
        private double _activation;
        private double _bias;
        private double _errorSignal; // Local error gradient signal (δ_j)
        private readonly double[] _weights; // Weights connecting this hidden neuron to each output neuron

        public int Id => _id;
        public double Activation => _activation;
        public double Bias => _bias;
        public double ErrorSignal => _errorSignal;
        public int WeightCount => _weights.Length;

        /// <summary>
        /// Initializes a hidden neuron with an identifier, target output count, and small random weights and bias.
        /// </summary>
        public HNeuron(int id, int outputSize, Random? rnd = null)
        {
            _id = id;
            _activation = 0.0;
            _errorSignal = 0.0;
            _weights = new double[outputSize];

            rnd ??= new Random(id + 100);
            _bias = 0.01;
            for (int o = 0; o < outputSize; ++o)
            {
                _weights[o] = rnd.NextDouble() - 0.5;
            }
        }

        public void SetActivation(double value) => _activation = value;
        public void SetBias(double value) => _bias = value;
        public void SetError(double value) => _errorSignal = value;

        public double GetWeight(int outputIndex) => _weights[outputIndex];
        public void SetWeight(int outputIndex, double value) => _weights[outputIndex] = value;

        /// <summary>
        /// Calculates the hidden neuron error gradient signal (δ_j):
        /// δ_j = (1 - h_j) * h_j * sum_k(δ_k * w_jk)
        /// Backpropagates output error signals through outgoing connection weights.
        /// </summary>
        /// <param name="outputErrorDerivatives">Array of output layer error gradient signals (δ_k).</param>
        public void CalculateError(double[] outputErrorDerivatives)
        {
            double sum = 0.0;
            for (int k = 0; k < outputErrorDerivatives.Length; ++k)
            {
                sum += outputErrorDerivatives[k] * _weights[k];
            }
            // Analytical derivative for logistic sigmoid: σ'(z) = σ(z) * (1 - σ(z))
            _errorSignal = sum * _activation * (1.0 - _activation);
        }

        /// <summary>
        /// Updates outgoing weights to output layer neurons:
        /// Δw_jk = η * δ_k * h_j
        /// w_jk = w_jk + Δw_jk
        /// </summary>
        /// <param name="learningRate">Learning rate for output layer (η_out).</param>
        /// <param name="outputErrorDerivatives">Output layer error gradient signals (δ_k).</param>
        public void UpdateWeights(double learningRate, double[] outputErrorDerivatives)
        {
            double step = _activation * learningRate;
            for (int k = 0; k < outputErrorDerivatives.Length; ++k)
            {
                _weights[k] += step * outputErrorDerivatives[k];
            }
        }

        /// <summary>
        /// Updates the hidden neuron's internal bias:
        /// Δb_j = η * δ_j
        /// b_j = b_j + Δb_j
        /// </summary>
        /// <param name="learningRate">Learning rate for hidden layer (η_in).</param>
        public void UpdateBias(double learningRate)
        {
            _bias += _errorSignal * learningRate;
        }
    }
}
