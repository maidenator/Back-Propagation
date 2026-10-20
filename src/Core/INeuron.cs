using System;

namespace BackPropagation.Core
{
    /// <summary>
    /// Represents an input layer neuron in the object-oriented network model.
    /// Responsible for receiving external feature values and storing outgoing weights to each hidden neuron.
    /// </summary>
    public class INeuron
    {
        private readonly int _id;
        private double _inputValue;
        private readonly double[] _weights; // Weights connecting this input neuron to each hidden neuron

        public int Id => _id;
        public double InputValue => _inputValue;
        public int WeightCount => _weights.Length;

        /// <summary>
        /// Initializes an input neuron with an identifier and target hidden layer width.
        /// </summary>
        /// <param name="id">Zero-based index of this input neuron.</param>
        /// <param name="hiddenSize">Total number of downstream hidden neurons.</param>
        /// <param name="rnd">Random number generator for weight initialization.</param>
        public INeuron(int id, int hiddenSize, Random? rnd = null)
        {
            _id = id;
            _inputValue = 0.0;
            _weights = new double[hiddenSize];

            rnd ??= new Random(id);
            for (int h = 0; h < hiddenSize; ++h)
            {
                // Small symmetric random initialization: [-0.5, 0.5]
                _weights[h] = rnd.NextDouble() - 0.5;
            }
        }

        public void SetInput(double value)
        {
            _inputValue = value;
        }

        public double GetWeight(int hiddenIndex)
        {
            return _weights[hiddenIndex];
        }

        public void SetWeight(int hiddenIndex, double value)
        {
            _weights[hiddenIndex] = value;
        }

        /// <summary>
        /// Applies the backpropagation weight update rule for input-to-hidden connection:
        /// Δw_ih = η * hiddenError * x_i
        /// w_ih = w_ih + Δw_ih
        /// </summary>
        /// <param name="hiddenIndex">Index of the connected hidden neuron.</param>
        /// <param name="hiddenError">Error gradient signal δ_j from the hidden neuron.</param>
        /// <param name="learningRate">Learning rate parameter η.</param>
        public void UpdateWeight(int hiddenIndex, double hiddenError, double learningRate)
        {
            _weights[hiddenIndex] += learningRate * hiddenError * _inputValue;
        }
    }
}
