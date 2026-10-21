using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using BackPropagation.Core;
using BackPropagation.Data;
using BackPropagation.Experiments;
#if NET8_0_WINDOWS || WINDOWS
using System.Windows.Forms;
using BackPropagation.GUI;
#endif

namespace BackPropagation
{
    public class Program
    {
        private const int DefaultEpochs = 1000;

        [STAThread]
        public static void Main(string[] args)
        {
            try
            {
                Console.OutputEncoding = System.Text.Encoding.UTF8;
            }
            catch (Exception)
            {
                // No console handle (GUI launch) or a console that refuses the encoding change.
            }

            if (args.Contains("--help") || args.Contains("-h") || args.Contains("-?"))
            {
                PrintHelp();
                return;
            }

            int epochs = ParseEpochs(args, out string[] options);

            bool forceCli = options.Contains("--cli") || options.Contains("-c");
            bool wantsMenu = options.Contains("--menu") || options.Contains("-m");
            bool wantsComparison = options.Contains("--compare") || options.Contains("--all") || options.Contains("-a");
            bool noArguments = args.Length == 0;

#if NET8_0_WINDOWS || WINDOWS
            // On Windows the dashboard is the default entry point. Pass --compare (or --cli)
            // to run the console comparison instead.
            if (options.Contains("--gui") || options.Contains("-g") || noArguments)
            {
                ApplicationConfiguration.Initialize();
                Application.Run(new MainForm());
                return;
            }
#endif

            if (wantsMenu)
            {
                RunInteractiveMenu(epochs);
                return;
            }

            // Default run: train all three configurations and print the comparison table.
            if (wantsComparison || forceCli || options.Length == 0)
            {
                ComparisonRunner.RunComparison(epochs);
                return;
            }

            HandleCommandLineArgs(options, epochs);
        }

        /// <summary>
        /// Extracts an optional --epochs N override and returns the remaining arguments.
        /// </summary>
        private static int ParseEpochs(string[] args, out string[] remaining)
        {
            int epochs = DefaultEpochs;
            var kept = new List<string>();

            for (int i = 0; i < args.Length; ++i)
            {
                bool isEpochFlag = (args[i] == "--epochs" || args[i] == "-e");
                if (isEpochFlag && i + 1 < args.Length &&
                    int.TryParse(args[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) &&
                    value > 0)
                {
                    epochs = value;
                    ++i;
                    continue;
                }

                kept.Add(args[i]);
            }

            remaining = kept.ToArray();
            return epochs;
        }

        private static void HandleCommandLineArgs(string[] args, int epochs)
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
                    SyntheticBenchmark.RunDemo(epochs);
                    break;
                case "--cross-entropy":
                case "-ce":
                    SyntheticBenchmark.RunCrossEntropyDemo(epochs);
                    break;
                case "--weight-decay":
                case "-wd":
                    SyntheticBenchmark.RunWeightDecayDemo(epochs);
                    break;
                case "--logic":
                case "-l":
                    LogicGateExperiment.RunTrainUntilFit();
                    break;
                case "--optimize":
                case "-o":
                    HiddenOptimizer.RunOptimization();
                    break;

                case "--help":
                case "-h":
                case "-?":
                    PrintHelp();
                    break;
                default:
                    Console.WriteLine($"Unknown option: {flag}");
                    PrintHelp();
                    break;
            }
        }

        private static void PrintHelp()
        {
            Console.WriteLine("Usage: dotnet run --project src/BackPropagation.csproj -- [option] [--epochs N]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --compare, -a        Train all three configurations and print the comparison table");
            Console.WriteLine("  --cli, -c            Force the console comparison");
            Console.WriteLine("  --menu, -m           Launch the interactive terminal menu");
#if NET8_0_WINDOWS || WINDOWS
            Console.WriteLine("  --gui, -g            Launch the Windows Forms graphical interface (Windows only)");
            Console.WriteLine();
            Console.WriteLine("  With no arguments the Windows Forms dashboard opens (Linux and macOS");
            Console.WriteLine("  run the comparison instead).");
#endif
            Console.WriteLine("  --synthetic, -s      Run CS345 4-5-3 benchmark (Standard MSE)");
            Console.WriteLine("  --cross-entropy, -ce Run CS345 benchmark using Cross-Entropy loss");
            Console.WriteLine("  --weight-decay, -wd  Run CS345 benchmark using L2 weight decay (lambda = 0.001)");
            Console.WriteLine("  --logic, -l          Run 4-input boolean logic gate experiment");
            Console.WriteLine("  --optimize, -o       Run hidden neuron count optimization benchmark");
            Console.WriteLine("  --epochs, -e N       Override the number of training epochs (default 1000)");
            Console.WriteLine("  --help, -h           Show this help text");
        }

        private static void RunInteractiveMenu(int epochs)
        {
            while (true)
            {
                Console.WriteLine("================================================================================");
                Console.WriteLine("       CS345 BACKPROPAGATION NEURAL NETWORK (WITH WHAT-TO-DO EXTENSIONS)        ");
                Console.WriteLine("================================================================================");
                Console.WriteLine("  [1] Run full comparison: MSE vs Cross-Entropy vs Weight Decay");
                Console.WriteLine("  [2] Run CS345 4-5-3 Benchmark: Standard Mean Squared Error (MSE)");
                Console.WriteLine("  [3] Run CS345 4-5-3 Benchmark: Cross-Entropy Loss (WHAT TO DO)");
                Console.WriteLine("  [4] Run CS345 4-5-3 Benchmark: Weight Decay / L2 Regularization (WHAT TO DO)");
                Console.WriteLine("  [5] Classify Custom Prediction Input (e.g. 1.0, 2.0, 3.0, 4.0)");
                Console.WriteLine("  [6] Run 4-Input Logic Gate Experiment (Train-Until-Fit)");
                Console.WriteLine("  [7] Run Hidden Neuron Optimization Benchmark");
                Console.WriteLine("  [0] Exit");
                Console.WriteLine("================================================================================");
                Console.Write("Select an option (0-7): ");

                string? choice = Console.ReadLine()?.Trim();
                if (choice is null)
                {
                    // End of input (Ctrl+D / redirected stdin): exit cleanly instead of looping.
                    Console.WriteLine();
                    return;
                }

                Console.WriteLine();

                switch (choice)
                {
                    case "1":
                        ComparisonRunner.RunComparison(epochs);
                        break;
                    case "2":
                        SyntheticBenchmark.RunDemo(epochs);
                        break;
                    case "3":
                        SyntheticBenchmark.RunCrossEntropyDemo(epochs);
                        break;
                    case "4":
                        SyntheticBenchmark.RunWeightDecayDemo(epochs);
                        break;

                    case "5":
                        RunInteractivePrediction();
                        break;

                    case "6":
                        LogicGateExperiment.RunTrainUntilFit();
                        break;
                    case "7":
                        HiddenOptimizer.RunOptimization();
                        break;
                    case "0":
                        Console.WriteLine("Exiting. Thank you!");
                        return;
                    default:
                        Console.WriteLine("Invalid selection. Please choose an option between 0 and 7.");
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

            string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 4)
            {
                Console.WriteLine("Error: Exactly 4 numbers required.");
                return;
            }

            double[] x = new double[4];
            for (int i = 0; i < 4; ++i)
            {
                if (!double.TryParse(parts[i], out x[i]))
                {
                    Console.WriteLine($"Error: '{parts[i]}' is not a valid number.");
                    return;
                }
            }

            double[] y = nn.ComputeOutputs(x);
            int predClass = MatrixNeuralNet.MaxIndex(y);
            string[] classNames = { "Class 0 (Conservative)", "Class 1 (Moderate)", "Class 2 (Liberal)" };

            Console.WriteLine($"\nInput Vector: [ {string.Join(", ", x)} ]");
            Console.WriteLine($"Softmax Probabilities: [ {y[0]:F4}, {y[1]:F4}, {y[2]:F4} ]");
            Console.WriteLine($"Predicted Classification: {classNames[predClass]} (Index {predClass})");
        }
    }
}
