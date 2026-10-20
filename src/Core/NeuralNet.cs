using System;
using System.IO;

namespace BackPropagation.Core
{
    /// <summary>
    /// Object-oriented 3-layer neural network replicating the reference implementation from Xylphy.
    /// Uses INeuron, HNeuron, and ONeuron instances with explicit layer-by-layer message passing.
    /// </summary>
    public class NeuralNet
    {
        private readonly INeuron[] _ineuron;
        private readonly HNeuron[] _hneuron;
        private readonly ONeuron[] _oneuron;

        private readonly double _lrpOut; // Learning rate for output layer
        private readonly double _lrpIn;  // Learning rate for hidden layer

        private readonly double[] _desiredOutput;
        private readonly double[] _errorComponent;  // e_k = target - output
        private readonly double[] _errorDerivative; // δ_k = y_k * (1 - y_k) * e_k

        public int InputCount => _ineuron.Length;
        public int HiddenCount => _hneuron.Length;
        public int OutputCount => _oneuron.Length;

        public INeuron[] InputNeurons => _ineuron;
        public HNeuron[] HiddenNeurons => _hneuron;
        public ONeuron[] OutputNeurons => _oneuron;
        public double[] ErrorComponent => _errorComponent;
        public double[] ErrorDerivative => _errorDerivative;

        public double LrpOut => _lrpOut;
        public double LrpIn => _lrpIn;

        public NeuralNet(int inputCount, int hiddenCount, int outputCount, 
            double lrpOut = 0.2, double lrpIn = 0.15, Random? rnd = null)
        {
            _lrpOut = lrpOut;
            _lrpIn = lrpIn;

            _ineuron = new INeuron[inputCount];
            _hneuron = new HNeuron[hiddenCount];
            _oneuron = new ONeuron[outputCount];

            _desiredOutput = new double[outputCount];
            _errorComponent = new double[outputCount];
            _errorDerivative = new double[outputCount];

            rnd ??= new Random(42);

            for (int i = 0; i < inputCount; ++i)
                _ineuron[i] = new INeuron(i, hiddenCount, rnd);

            for (int h = 0; h < hiddenCount; ++h)
                _hneuron[h] = new HNeuron(h, outputCount, rnd);

            for (int o = 0; o < outputCount; ++o)
                _oneuron[o] = new ONeuron(o);
        }

        public void SetInput(int index, double value) => _ineuron[index].SetInput(value);
        public void SetDesiredOutput(int index, double value) => _desiredOutput[index] = value;
        public double GetOutputData(int index) => _oneuron[index].Activation;

        /// <summary>
        /// Computes hidden layer activations via forward propagation:
        /// net_j = sum_i (x_i * w_ij) + b_j
        /// h_j = σ(net_j)
        /// </summary>
        public void CalculateHiddenActivations()
        {
            for (int h = 0; h < _hneuron.Length; ++h)
            {
                double sum = 0.0;
                for (int i = 0; i < _ineuron.Length; ++i)
                {
                    sum += _ineuron[i].InputValue * _ineuron[i].GetWeight(h);
                }
                double activation = Activations.Sigmoid(sum + _hneuron[h].Bias);
                _hneuron[h].SetActivation(activation);
            }
        }

        /// <summary>
        /// Computes output layer activations via forward propagation:
        /// net_k = sum_h (h_j * w_jk) + b_k
        /// y_k = σ(net_k)
        /// </summary>
        public void CalculateOutputActivations()
        {
            for (int o = 0; o < _oneuron.Length; ++o)
            {
                double sum = 0.0;
                for (int h = 0; h < _hneuron.Length; ++h)
                {
                    sum += _hneuron[h].Activation * _hneuron[h].GetWeight(o);
                }
                double activation = Activations.Sigmoid(sum + _oneuron[o].Bias);
                _oneuron[o].SetActivation(activation);
            }
        }

        /// <summary>
        /// Performs full forward pass across hidden and output layers.
        /// </summary>
        public void Run()
        {
            CalculateHiddenActivations();
            CalculateOutputActivations();
        }

        /// <summary>
        /// Computes raw error components: e_k = target_k - actual_k.
        /// </summary>
        public void CalculateErrorComponents()
        {
            for (int o = 0; o < _oneuron.Length; ++o)
            {
                _errorComponent[o] = _desiredOutput[o] - _oneuron[o].Activation;
            }
        }

        /// <summary>
        /// Computes local error gradient signals for output layer:
        /// δ_k = y_k * (1 - y_k) * (target_k - y_k)
        /// </summary>
        public void CalculateOutputDerivatives()
        {
            for (int o = 0; o < _oneuron.Length; ++o)
            {
                double y = _oneuron[o].Activation;
                _errorDerivative[o] = y * (1.0 - y) * _errorComponent[o];
            }
        }

        /// <summary>
        /// Executes a single online backpropagation learning cycle for the current input and desired target.
        /// </summary>
        public void Learn()
        {
            Run();
            CalculateErrorComponents();
            CalculateOutputDerivatives();

            // 1. Calculate hidden layer error signals δ_j from output signals
            for (int h = 0; h < _hneuron.Length; ++h)
                _hneuron[h].CalculateError(_errorDerivative);

            // 2. Update hidden-to-output weights
            for (int h = 0; h < _hneuron.Length; ++h)
                _hneuron[h].UpdateWeights(_lrpOut, _errorDerivative);

            // 3. Update input-to-hidden weights
            for (int i = 0; i < _ineuron.Length; ++i)
            {
                for (int h = 0; h < _hneuron.Length; ++h)
                {
                    _ineuron[i].UpdateWeight(h, _hneuron[h].ErrorSignal, _lrpIn);
                }
            }

            // 4. Update output neuron biases
            for (int o = 0; o < _oneuron.Length; ++o)
                _oneuron[o].UpdateBias(_lrpOut, _errorDerivative[o]);

            // 5. Update hidden neuron biases
            for (int h = 0; h < _hneuron.Length; ++h)
                _hneuron[h].UpdateBias(_lrpIn);
        }

        /// <summary>
        /// Saves all weights and biases to a clean, human-readable plain text file.
        /// </summary>
        public void SaveWeights(string path)
        {
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            using var sw = new StreamWriter(path);
            sw.WriteLine($"# NeuralNet Weights & Biases (Inputs:{InputCount}, Hidden:{HiddenCount}, Outputs:{OutputCount})");
            
            // Input-to-hidden weights
            for (int i = 0; i < _ineuron.Length; ++i)
                for (int h = 0; h < _hneuron.Length; ++h)
                    sw.WriteLine(_ineuron[i].GetWeight(h).ToString("R"));

            // Hidden-to-output weights
            for (int h = 0; h < _hneuron.Length; ++h)
                for (int o = 0; o < _oneuron.Length; ++o)
                    sw.WriteLine(_hneuron[h].GetWeight(o).ToString("R"));

            // Hidden biases
            for (int h = 0; h < _hneuron.Length; ++h)
                sw.WriteLine(_hneuron[h].Bias.ToString("R"));

            // Output biases
            for (int o = 0; o < _oneuron.Length; ++o)
                sw.WriteLine(_oneuron[o].Bias.ToString("R"));
        }

        /// <summary>
        /// Loads weights and biases previously serialized to a plain text file.
        /// </summary>
        public void LoadWeights(string path)
        {
            using var sr = new StreamReader(path);
            string? line = sr.ReadLine();
            // Skip comments if present
            while (line != null && line.StartsWith("#"))
                line = sr.ReadLine();

            for (int i = 0; i < _ineuron.Length; ++i)
                for (int h = 0; h < _hneuron.Length; ++h)
                {
                    if (line == null) throw new InvalidDataException("Unexpected EOF while reading input weights.");
                    _ineuron[i].SetWeight(h, double.Parse(line));
                    line = sr.ReadLine();
                }

            for (int h = 0; h < _hneuron.Length; ++h)
                for (int o = 0; o < _oneuron.Length; ++o)
                {
                    if (line == null) throw new InvalidDataException("Unexpected EOF while reading hidden weights.");
                    _hneuron[h].SetWeight(o, double.Parse(line));
                    line = sr.ReadLine();
                }

            for (int h = 0; h < _hneuron.Length; ++h)
            {
                if (line == null) throw new InvalidDataException("Unexpected EOF while reading hidden biases.");
                _hneuron[h].SetBias(double.Parse(line));
                line = sr.ReadLine();
            }

            for (int o = 0; o < _oneuron.Length; ++o)
            {
                if (line == null) throw new InvalidDataException("Unexpected EOF while reading output biases.");
                _oneuron[o].SetBias(double.Parse(line));
                line = sr.ReadLine();
            }
        }
    }
}
