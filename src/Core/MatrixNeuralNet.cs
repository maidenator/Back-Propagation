using System;

namespace BackPropagation.Core
{
    /// <summary>
    /// High-performance primitive matrix/array-based backpropagation neural network.
    /// Directly implements the mathematical specification and jagged-array architecture
    /// from Dr. James McCaffrey's CS345 reference program with optional momentum and configurable activation functions.
    /// </summary>
    public class MatrixNeuralNet
    {
        private readonly int _numInput;
        private readonly int _numHidden;
        private readonly int _numOutput;

        private readonly double[] _inputs;
        private readonly double[][] _ihWeights; // input-to-hidden weights [numInput][numHidden]
        private readonly double[] _hBiases;     // hidden biases [numHidden]
        private readonly double[] _hOutputs;    // hidden node outputs [numHidden]

        private readonly double[][] _hoWeights; // hidden-to-output weights [numHidden][numOutput]
        private readonly double[] _oBiases;     // output biases [numOutput]
        private readonly double[] _outputs;     // final outputs [numOutput]

        private readonly Random _rnd;

        public int NumInput => _numInput;
        public int NumHidden => _numHidden;
        public int NumOutput => _numOutput;

        public double[] Inputs => _inputs;
        public double[][] IhWeights => _ihWeights;
        public double[] HBiases => _hBiases;
        public double[] HOutputs => _hOutputs;
        public double[][] HoWeights => _hoWeights;
        public double[] OBiases => _oBiases;
        public double[] Outputs => _outputs;

        public MatrixNeuralNet(int numInput, int numHidden, int numOutput, int seed = 0)
        {
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;

            _inputs = new double[numInput];
            _ihWeights = MakeMatrix(numInput, numHidden, 0.0);
            _hBiases = new double[numHidden];
            _hOutputs = new double[numHidden];

            _hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            _oBiases = new double[numOutput];
            _outputs = new double[numOutput];

            _rnd = new Random(seed);
            InitializeWeights();
        }

        public static double[][] MakeMatrix(int rows, int cols, double initialValue)
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < rows; ++r)
            {
                result[r] = new double[cols];
                if (initialValue != 0.0)
                {
                    for (int c = 0; c < cols; ++c)
                        result[r][c] = initialValue;
                }
            }
            return result;
        }

        private void InitializeWeights()
        {
            int numWeights = (_numInput * _numHidden) + _numHidden +
                             (_numHidden * _numOutput) + _numOutput;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
            {
                // Small random values between -0.01 and +0.01
                initialWeights[i] = (0.02 * _rnd.NextDouble()) - 0.01;
            }
            SetWeights(initialWeights);
        }

        public void SetWeights(double[] weights)
        {
            int expectedCount = (_numInput * _numHidden) + _numHidden +
                                (_numHidden * _numOutput) + _numOutput;
            if (weights.Length != expectedCount)
                throw new ArgumentException($"Expected {expectedCount} weights, received {weights.Length}");

            int k = 0;
            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    _ihWeights[i][j] = weights[k++];

            for (int j = 0; j < _numHidden; ++j)
                _hBiases[j] = weights[k++];

            for (int j = 0; j < _numHidden; ++j)
                for (int o = 0; o < _numOutput; ++o)
                    _hoWeights[j][o] = weights[k++];

            for (int o = 0; o < _numOutput; ++o)
                _oBiases[o] = weights[k++];
        }

        public double[] GetWeights()
        {
            int total = (_numInput * _numHidden) + _numHidden +
                        (_numHidden * _numOutput) + _numOutput;
            double[] result = new double[total];
            int k = 0;

            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    result[k++] = _ihWeights[i][j];

            for (int j = 0; j < _numHidden; ++j)
                result[k++] = _hBiases[j];

            for (int j = 0; j < _numHidden; ++j)
                for (int o = 0; o < _numOutput; ++o)
                    result[k++] = _hoWeights[j][o];

            for (int o = 0; o < _numOutput; ++o)
                result[k++] = _oBiases[o];

            return result;
        }

        /// <summary>
        /// Computes outputs for given x-values using Tanh for hidden nodes and Softmax for output nodes.
        /// </summary>
        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[_numHidden];
            double[] oSums = new double[_numOutput];

            for (int i = 0; i < _numInput; ++i)
                _inputs[i] = xValues[i];

            // 1. Hidden node weighted sums
            for (int j = 0; j < _numHidden; ++j)
            {
                for (int i = 0; i < _numInput; ++i)
                    hSums[j] += _inputs[i] * _ihWeights[i][j];
                hSums[j] += _hBiases[j];
                _hOutputs[j] = Activations.HyperTan(hSums[j]);
            }

            // 2. Output node weighted sums
            for (int k = 0; k < _numOutput; ++k)
            {
                for (int j = 0; j < _numHidden; ++j)
                    oSums[k] += _hOutputs[j] * _hoWeights[j][k];
                oSums[k] += _oBiases[k];
            }

            // 3. Softmax normalized outputs
            double[] softOut = Activations.Softmax(oSums);
            Array.Copy(softOut, _outputs, softOut.Length);

            double[] result = new double[_numOutput];
            Array.Copy(_outputs, result, _numOutput);
            return result;
        }

        /// <summary>
        /// Trains the model using stochastic online backpropagation with momentum.
        /// </summary>
        public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum, 
            Action<int, double>? epochCallback = null)
        {
            double[][] hoGrads = MakeMatrix(_numHidden, _numOutput, 0.0);
            double[] obGrads = new double[_numOutput];
            double[][] ihGrads = MakeMatrix(_numInput, _numHidden, 0.0);
            double[] hbGrads = new double[_numHidden];

            double[] oSignals = new double[_numOutput];
            double[] hSignals = new double[_numHidden];

            double[][] ihPrevWeightsDelta = MakeMatrix(_numInput, _numHidden, 0.0);
            double[] hPrevBiasesDelta = new double[_numHidden];
            double[][] hoPrevWeightsDelta = MakeMatrix(_numHidden, _numOutput, 0.0);
            double[] oPrevBiasesDelta = new double[_numOutput];

            int epoch = 0;
            double[] xValues = new double[_numInput];
            double[] tValues = new double[_numOutput];

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i) sequence[i] = i;

            int errInterval = Math.Max(1, maxEpochs / 10);

            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % errInterval == 0 || epoch == maxEpochs)
                {
                    double trainErr = MeanSquaredError(trainData);
                    epochCallback?.Invoke(epoch, trainErr);
                }

                Shuffle(sequence);

                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], 0, xValues, 0, _numInput);
                    Array.Copy(trainData[idx], _numInput, tValues, 0, _numOutput);

                    ComputeOutputs(xValues);

                    // 1. Output error gradient signals (Softmax: δ_k = (t_k - y_k) * y_k * (1 - y_k))
                    for (int k = 0; k < _numOutput; ++k)
                    {
                        double errorSignal = tValues[k] - _outputs[k];
                        double derivative = (1.0 - _outputs[k]) * _outputs[k];
                        oSignals[k] = errorSignal * derivative;
                    }

                    // 2. Hidden-to-output gradients
                    for (int j = 0; j < _numHidden; ++j)
                        for (int k = 0; k < _numOutput; ++k)
                            hoGrads[j][k] = oSignals[k] * _hOutputs[j];

                    for (int k = 0; k < _numOutput; ++k)
                        obGrads[k] = oSignals[k] * 1.0;

                    // 3. Hidden node signals (Tanh: derivative = (1 + h) * (1 - h))
                    for (int j = 0; j < _numHidden; ++j)
                    {
                        double derivative = (1.0 + _hOutputs[j]) * (1.0 - _hOutputs[j]);
                        double sum = 0.0;
                        for (int k = 0; k < _numOutput; ++k)
                            sum += oSignals[k] * _hoWeights[j][k];
                        hSignals[j] = derivative * sum;
                    }

                    // 4. Input-to-hidden gradients
                    for (int i = 0; i < _numInput; ++i)
                        for (int j = 0; j < _numHidden; ++j)
                            ihGrads[i][j] = hSignals[j] * _inputs[i];

                    for (int j = 0; j < _numHidden; ++j)
                        hbGrads[j] = hSignals[j] * 1.0;

                    // == Update weights and biases with momentum ==

                    // Update input-to-hidden weights
                    for (int i = 0; i < _numInput; ++i)
                    {
                        for (int j = 0; j < _numHidden; ++j)
                        {
                            double delta = ihGrads[i][j] * learnRate;
                            _ihWeights[i][j] += delta;
                            _ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta;
                        }
                    }

                    // Update hidden biases
                    for (int j = 0; j < _numHidden; ++j)
                    {
                        double delta = hbGrads[j] * learnRate;
                        _hBiases[j] += delta;
                        _hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // Update hidden-to-output weights
                    for (int j = 0; j < _numHidden; ++j)
                    {
                        for (int k = 0; k < _numOutput; ++k)
                        {
                            double delta = hoGrads[j][k] * learnRate;
                            _hoWeights[j][k] += delta;
                            _hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // Update output biases
                    for (int k = 0; k < _numOutput; ++k)
                    {
                        double delta = obGrads[k] * learnRate;
                        _oBiases[k] += delta;
                        _oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }
                }
            }

            return GetWeights();
        }

        private void Shuffle(int[] sequence)
        {
            // Fisher-Yates shuffle
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = _rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        public double MeanSquaredError(double[][] dataset)
        {
            double sumSquaredError = 0.0;
            double[] xValues = new double[_numInput];
            double[] tValues = new double[_numOutput];

            for (int i = 0; i < dataset.Length; ++i)
            {
                Array.Copy(dataset[i], 0, xValues, 0, _numInput);
                Array.Copy(dataset[i], _numInput, tValues, 0, _numOutput);
                double[] yValues = ComputeOutputs(xValues);

                for (int k = 0; k < _numOutput; ++k)
                {
                    double err = tValues[k] - yValues[k];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / dataset.Length;
        }

        public double Accuracy(double[][] dataset)
        {
            int correct = 0;
            double[] xValues = new double[_numInput];
            double[] tValues = new double[_numOutput];

            for (int i = 0; i < dataset.Length; ++i)
            {
                Array.Copy(dataset[i], 0, xValues, 0, _numInput);
                Array.Copy(dataset[i], _numInput, tValues, 0, _numOutput);
                double[] yValues = ComputeOutputs(xValues);

                if (MaxIndex(yValues) == MaxIndex(tValues))
                    ++correct;
            }
            return (double)correct / dataset.Length;
        }

        public static int MaxIndex(double[] vector)
        {
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 1; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }
    }
}
