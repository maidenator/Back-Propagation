using System;
using System.Linq;
using System.Runtime.InteropServices;
using BackPropagation.Core;
using BackPropagation.Data;
using BackPropagation.Experiments;
using BackPropagation.SelfDoc;
#if NET8_0_WINDOWS || WINDOWS
using System.Windows.Forms;
using BackPropagation.GUI;
#endif

namespace BackPropagation
{
    public class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            bool forceCli = args.Contains("--cli") || args.Contains("-c");
            bool isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

#if NET8_0_WINDOWS || WINDOWS
            // If running on Windows and not explicitly in CLI mode, launch the Windows Forms GUI
            if (isWindows && !forceCli && args.Length == 0)
            {
                ApplicationConfiguration.Initialize();
                Application.Run(new MainForm());
                return;
            }
#endif

            if (args.Length > 0 && !forceCli)
            {
                HandleCommandLineArgs(args);
                return;
            }

            // Interactive terminal interface (works everywhere, including Linux)
            RunInteractiveMenu();
        }

        private static void HandleCommandLineArgs(string[] args)
        {
            string flag = args[0].ToLowerInvariant();
            switch (flag)
            {
#if NET8_0_WINDOWS || WINDOWS
                case "--gui":
                case "-g":
                    ApplicationConfiguration.Initialize();
                    Application.Run(new MainForm());
                    break;
#endif
                case "--synthetic":
                case "-s":
                    SyntheticBenchmark.RunDemo();
                    break;
                case "--cross-entropy":
                case "-ce":
                    SyntheticBenchmark.RunCrossEntropyDemo();
                    break;
                case "--weight-decay":
                case "-wd":
                    SyntheticBenchmark.RunWeightDecayDemo();
                    break;
                case "--logic":
                case "-l":
                    LogicGateExperiment.RunTrainUntilFit();
                    break;
                case "--optimize":
                case "-o":
                    HiddenOptimizer.RunOptimization();
                    break;
                case "--trace":
                case "-t":
                    var net = new NeuralNet(4, 5, 1);
                    MathTracer.TraceSample(net, new double[] { 1, 1, 1, 0 }, new double[] { 0 });
                    break;
                case "--visualize":
                case "-v":
                    AsciiVisualizer.RenderTopology(4, 5, 3);
                    break;
                default:
                    Console.WriteLine($"Unknown option: {flag}");
                    PrintHelp();
                    break;
            }
        }

        private static void PrintHelp()
        {
            Console.WriteLine("Usage: dotnet run --project src/BackPropagation.csproj [option]");
            Console.WriteLine("Options:");
#if NET8_0_WINDOWS || WINDOWS
            Console.WriteLine("  --gui, -g             Launch Windows Forms graphical interface (Windows)");
#endif
            Console.WriteLine("  --synthetic, -s       Run CS345 4-5-3 benchmark (Standard MSE)");
            Console.WriteLine("  --cross-entropy, -ce  Run CS345 with Cross-Entropy Loss (from 'WHAT TO DO')");
            Console.WriteLine("  --weight-decay, -wd   Run CS345 with L2 Weight Decay (from 'WHAT TO DO')");
            Console.WriteLine("  --logic, -l           Run 4-input boolean logic gate experiment");
            Console.WriteLine("  --optimize, -o        Run hidden neuron count optimization benchmark");
            Console.WriteLine("  --trace, -t           Run single-sample mathematical trace");
            Console.WriteLine("  --visualize, -v       Render ASCII network topology");
        }

        private static void RunInteractiveMenu()
        {
            while (true)
            {
                Console.WriteLine("================================================================================");
                Console.WriteLine("       CS345 BACKPROPAGATION NEURAL NETWORK (WITH WHAT-TO-DO EXTENSIONS)        ");
                Console.WriteLine("================================================================================");
                Console.WriteLine("  [1] Run CS345 4-5-3 Benchmark: Standard Mean Squared Error (MSE)");
                Console.WriteLine("  [2] Run CS345 4-5-3 Benchmark: Cross-Entropy Loss (WHAT TO DO)");
                Console.WriteLine("  [3] Run CS345 4-5-3 Benchmark: Weight Decay / L2 Regularization (WHAT TO DO)");
                Console.WriteLine("  [4] Mathematical Step-by-Step Calculation Trace (--trace)");
                Console.WriteLine("  [5] Classify Custom Prediction Input (e.g. 1.0, 2.0, 3.0, 4.0)");
                Console.WriteLine("  [6] Display ASCII Network Topology Diagram");
                Console.WriteLine("  [7] Run 4-Input Logic Gate Experiment (Train-Until-Fit)");
                Console.WriteLine("  [8] Run Hidden Neuron Optimization Benchmark");
                Console.WriteLine("  [9] Exit");
                Console.WriteLine("================================================================================");
                Console.Write("Select an option (1-9): ");

                string? choice = Console.ReadLine()?.Trim();
                Console.WriteLine();

                switch (choice)
                {
                    case "1":
                        SyntheticBenchmark.RunDemo();
                        break;
                    case "2":
                        SyntheticBenchmark.RunCrossEntropyDemo();
                        break;
                    case "3":
                        SyntheticBenchmark.RunWeightDecayDemo();
                        break;
                    case "4":
                        var traceNet = new NeuralNet(4, 4, 1);
                        MathTracer.TraceSample(traceNet, new double[] { 1, 1, 1, 0 }, new double[] { 0 });
                        break;
                    case "5":
                        RunInteractivePrediction();
                        break;
                    case "6":
                        AsciiVisualizer.RenderTopology(4, 5, 3);
                        break;
                    case "7":
                        LogicGateExperiment.RunTrainUntilFit();
                        break;
                    case "8":
                        HiddenOptimizer.RunOptimization();
                        break;
                    case "9":
                        Console.WriteLine("Exiting. Thank you!");
                        return;
                    default:
                        Console.WriteLine("Invalid selection. Please choose an option between 1 and 9.");
                        break;
                }

                Console.WriteLine("\nPress Enter to return to main menu...");
                Console.ReadLine();
                Console.WriteLine();
            }
        }

        private static void RunInteractivePrediction()
        {
            Console.WriteLine("Training a baseline 4-5-3 CS345 network for live prediction...");
            double[][] allData = SyntheticDataset.MakeAllData(4, 5, 3, 1000, 1);
            SyntheticDataset.SplitTrainTest(allData, 0.80, 1, out double[][] train, out _);
            var nn = new MatrixNeuralNet(4, 5, 3, 1);
            nn.Train(train, 1000, 0.05, 0.01);
            Console.WriteLine("Model trained.\n");

            Console.WriteLine("Enter 4 predictor values separated by spaces (e.g. '1.0 2.0 3.0 4.0'):");
            Console.Write("> ");
            string? line = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) return;

            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 4)
            {
                Console.WriteLine("Error: Exactly 4 numbers required.");
                return;
            }

            double[] x = new double[4];
            for (int i = 0; i < 4; ++i)
                x[i] = double.Parse(parts[i]);

            double[] y = nn.ComputeOutputs(x);
            int predClass = MatrixNeuralNet.MaxIndex(y);
            string[] classNames = { "Class 0 (Conservative)", "Class 1 (Moderate)", "Class 2 (Liberal)" };

            Console.WriteLine($"\nInput Vector: [ {string.Join(", ", x)} ]");
            Console.WriteLine($"Softmax Probabilities: [ {y[0]:F4}, {y[1]:F4}, {y[2]:F4} ]");
            Console.WriteLine($"Predicted Classification: {classNames[predClass]} (Index {predClass})");
        }
    }
}
