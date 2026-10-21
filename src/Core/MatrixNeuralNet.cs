using System;

namespace BackPropagation.Core
{
    public enum LossFunction
    {
        MeanSquaredError,
        CrossEntropy
    }

    /// <summary>
    /// High-performance primitive matrix/array-based backpropagation neural network.
    /// Directly implements the mathematical specification and jagged-array architecture
    /// from Dr. James McCaffrey's CS345 reference program.
    /// 
    /// Includes the two extensions described in the 'WHAT TO DO' section:
    /// 1. Cross-Entropy Error (with Softmax output derivative simplification: δ_k = t_k - y_k)
    /// 2. Weight Decay (L2 Regularization) to penalize large weights and prevent overfitting.
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
            if (numInput <= 0) throw new ArgumentOutOfRangeException(nameof(numInput), "The network must have at least one input.");
            if (numHidden <= 0) throw new ArgumentOutOfRangeException(nameof(numHidden), "The network must have at least one hidden neuron.");
            if (numOutput <= 0) throw new ArgumentOutOfRangeException(nameof(numOutput), "The network must have at least one output.");

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
                // Small random values between 0.0001 and 0.001, matching the CS345 reference program.
                initialWeights[i] = (0.001 - 0.0001) * _rnd.NextDouble() + 0.0001;
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
            if (xValues.Length != _numInput)
                throw new ArgumentException($"Expected {_numInput} input values, received {xValues.Length}.", nameof(xValues));

            double[] hSums = new double[_numHidden];
            double[] oSums = new double[_numOutput];

            for (int i = 0; i < _numInput; ++i)
                _inputs[i] = xValues[i];

            // 1. Hidden node weighted sums: net_j = sum(x_i * w_ij) + b_j
            for (int j = 0; j < _numHidden; ++j)
            {
                for (int i = 0; i < _numInput; ++i)
                    hSums[j] += _inputs[i] * _ihWeights[i][j];
                hSums[j] += _hBiases[j];
                _hOutputs[j] = Activations.HyperTan(hSums[j]);
            }

            // 2. Output node weighted sums: net_k = sum(h_j * w_jk) + b_k
            for (int k = 0; k < _numOutput; ++k)
            {
                for (int j = 0; j < _numHidden; ++j)
                    oSums[k] += _hOutputs[j] * _hoWeights[j][k];
                oSums[k] += _oBiases[k];
            }

            // 3. Softmax normalized outputs: y_k = exp(net_k) / sum(exp(net_j))
            double[] softOut = Activations.Softmax(oSums);
            Array.Copy(softOut, _outputs, softOut.Length);

            double[] result = new double[_numOutput];
            Array.Copy(_outputs, result, _numOutput);
            return result;
        }

        /// <summary>
        /// Trains the model using stochastic online backpropagation with momentum,
        /// configurable loss function (MSE vs Cross-Entropy), and optional L2 Weight Decay.
        /// </summary>
        /// <param name="trainData">Training matrix [rows][numInput + numOutput].</param>
        /// <param name="maxEpochs">Maximum training iterations.</param>
        /// <param name="learnRate">Learning rate parameter η.</param>
        /// <param name="momentum">Momentum rate parameter α.</param>
        /// <param name="lossFunction">Loss function: MeanSquaredError or CrossEntropy.</param>
        /// <param name="weightDecay">L2 regularization rate λ (0.0 means no regularization).</param>
        /// <param name="epochCallback">Callback invoked with (epoch, currentLoss) at regular intervals.</param>
        /// <returns>Serialized array of best learned weights and biases.</returns>
        public double[] Train(
            double[][] trainData, 
            int maxEpochs, 
            double learnRate, 
            double momentum, 
            LossFunction lossFunction = LossFunction.MeanSquaredError,
            double weightDecay = 0.0,
            Action<int, double>? epochCallback = null)
        {
            ValidateDataset(trainData, nameof(trainData));
            if (maxEpochs < 0)
                throw new ArgumentOutOfRangeException(nameof(maxEpochs), "Epoch count cannot be negative.");

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
                    double currentLoss = (lossFunction == LossFunction.CrossEntropy)
                        ? CrossEntropyError(trainData)
                        : MeanSquaredError(trainData);
                    epochCallback?.Invoke(epoch, currentLoss);
                }

                Shuffle(sequence);

                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], 0, xValues, 0, _numInput);
                    Array.Copy(trainData[idx], _numInput, tValues, 0, _numOutput);

                    ComputeOutputs(xValues);

                    // 1. Output error gradient signals
                    if (lossFunction == LossFunction.CrossEntropy)
                    {
                        // Cross-Entropy with Softmax:
                        // dE/dz_k = y_k - t_k, so error gradient signal is simply (t_k - y_k)
                        for (int k = 0; k < _numOutput; ++k)
                        {
                            oSignals[k] = tValues[k] - _outputs[k];
                        }
                    }
                    else
                    {
                        // Mean Squared Error (MSE) with Softmax (Figure 3 in Back-Prop PDF):
                        // δ_k = (t_k - y_k) * y_k * (1 - y_k)
                        for (int k = 0; k < _numOutput; ++k)
                        {
                            double errorSignal = tValues[k] - _outputs[k];
                            double derivative = (1.0 - _outputs[k]) * _outputs[k];
                            oSignals[k] = errorSignal * derivative;
                        }
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

                    // == Update weights and biases with momentum & optional weight decay ==

                    // Update input-to-hidden weights
                    for (int i = 0; i < _numInput; ++i)
                    {
                        for (int j = 0; j < _numHidden; ++j)
                        {
                            double delta = ihGrads[i][j] * learnRate;
                            _ihWeights[i][j] += delta;
                            // L2 weight decay: shrink the weight after the update (matches the CS345 reference).
                            if (weightDecay > 0.0)
                                _ihWeights[i][j] -= weightDecay * _ihWeights[i][j];
                            _ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta;
                        }
                    }

                    // Update hidden biases (biases are not decayed)
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
                            // L2 weight decay: shrink the weight after the update (matches the CS345 reference).
                            if (weightDecay > 0.0)
                                _hoWeights[j][k] -= weightDecay * _hoWeights[j][k];
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

        /// <summary>
        /// Calculates Mean Squared Error across a dataset:
        /// MSE = (1 / N) * sum_i sum_k (t_ik - y_ik)^2
        /// </summary>
        public double MeanSquaredError(double[][] dataset)
        {
            ValidateDataset(dataset, nameof(dataset));

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

        /// <summary>
        /// Calculates Cross-Entropy error across a dataset (from 'WHAT TO DO' section):
        /// CE = - (1 / N) * sum_i sum_k t_ik * ln(y_ik + 1e-15)
        /// </summary>
        public double CrossEntropyError(double[][] dataset)
        {
            ValidateDataset(dataset, nameof(dataset));

            double sumCrossEntropy = 0.0;
            double[] xValues = new double[_numInput];
            double[] tValues = new double[_numOutput];

            for (int i = 0; i < dataset.Length; ++i)
            {
                Array.Copy(dataset[i], 0, xValues, 0, _numInput);
                Array.Copy(dataset[i], _numInput, tValues, 0, _numOutput);
                double[] yValues = ComputeOutputs(xValues);

                for (int k = 0; k < _numOutput; ++k)
                {
                    if (tValues[k] > 0.0)
                    {
                        // Small epsilon to guard against ln(0)
                        double prob = Math.Max(1e-15, yValues[k]);
                        sumCrossEntropy += -1.0 * tValues[k] * Math.Log(prob);
                    }
                }
            }
            return sumCrossEntropy / dataset.Length;
        }

        /// <summary>
        /// Euclidean (L2) norm of all trainable weights, excluding biases.
        /// Used to measure the effect of weight decay on weight magnitude.
        /// </summary>
        public double WeightNorm()
        {
            double sumSquared = 0.0;

            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    sumSquared += _ihWeights[i][j] * _ihWeights[i][j];

            for (int j = 0; j < _numHidden; ++j)
                for (int k = 0; k < _numOutput; ++k)
                    sumSquared += _hoWeights[j][k] * _hoWeights[j][k];

            return Math.Sqrt(sumSquared);
        }

        public double Accuracy(double[][] dataset)
        {
            ValidateDataset(dataset, nameof(dataset));

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

        private void ValidateDataset(double[][] dataset, string paramName)
        {
            if (dataset.Length == 0)
                throw new ArgumentException("Dataset must contain at least one row.", paramName);

            int rowWidth = _numInput + _numOutput;
            for (int i = 0; i < dataset.Length; ++i)
            {
                if (dataset[i].Length < rowWidth)
                {
                    throw new ArgumentException(
                        $"Dataset row {i} has {dataset[i].Length} values; expected at least {rowWidth} ({_numInput} inputs + {_numOutput} targets).",
                        paramName);
                }
            }
        }

        public static int MaxIndex(double[] vector)
        {
            if (vector.Length == 0)
                throw new ArgumentException("Vector must contain at least one value.", nameof(vector));

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
