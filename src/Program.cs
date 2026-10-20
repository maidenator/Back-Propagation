using System;
using BackPropagation.Core;
using BackPropagation.Data;
using BackPropagation.Experiments;
using BackPropagation.SelfDoc;

namespace BackPropagation
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            if (args.Length > 0)
            {
                HandleCommandLineArgs(args);
                return;
            }

            RunInteractiveMenu();
        }

        private static void HandleCommandLineArgs(string[] args)
        {
            string flag = args[0].ToLowerInvariant();
            switch (flag)
            {
                case "--logic":
                case "-l":
                    LogicGateExperiment.RunTrainUntilFit();
                    break;
                case "--synthetic":
                case "-s":
                    SyntheticBenchmark.RunDemo();
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
            Console.WriteLine("  --logic, -l       Run 4-input boolean logic gate experiment (Train-until-fit)");
            Console.WriteLine("  --synthetic, -s   Run CS345 4-5-3 synthetic classification benchmark");
            Console.WriteLine("  --optimize, -o    Run hidden neuron count optimization benchmark");
            Console.WriteLine("  --trace, -t       Run single-sample mathematical trace");
            Console.WriteLine("  --visualize, -v   Render ASCII network topology");
        }

        private static void RunInteractiveMenu()
        {
            NeuralNet? activeLogicNet = null;

            while (true)
            {
                Console.WriteLine("================================================================================");
                Console.WriteLine("          PRIMITIVE SELF-DOCUMENTING BACKPROPAGATION NEURAL NETWORK             ");
                Console.WriteLine("================================================================================");
                Console.WriteLine("  [1] Run 4-Input Logic Gate Experiment (Train-Until-Fit & Generate Report)");
                Console.WriteLine("  [2] Run CS345 4-5-3 Synthetic Classifier Benchmark (McCaffrey)");
                Console.WriteLine("  [3] Run Hidden Neuron Count Optimization Benchmark");
                Console.WriteLine("  [4] Mathematical Step-by-Step Calculation Trace (--trace)");
                Console.WriteLine("  [5] Interactive Manual Input Testing (4 Inputs -> Output)");
                Console.WriteLine("  [6] Display ASCII Network Topology Diagram");
                Console.WriteLine("  [7] Exit");
                Console.WriteLine("================================================================================");
                Console.Write("Select an option (1-7): ");

                string? choice = Console.ReadLine()?.Trim();
                Console.WriteLine();

                switch (choice)
                {
                    case "1":
                        LogicGateExperiment.RunTrainUntilFit();
                        // Pre-train a network for option 5
                        activeLogicNet = new NeuralNet(4, 20, 1);
                        var data = LogicGateDataset.Get4InputAndDataset();
                        for (int ep = 0; ep < 10000 && !LogicGateExperiment.FitTest(activeLogicNet, data); ++ep)
                        {
                            for (int r = 0; r < data.Length; ++r)
                            {
                                for (int i = 0; i < 4; ++i) activeLogicNet.SetInput(i, data[r][i]);
                                activeLogicNet.SetDesiredOutput(0, data[r][4]);
                                activeLogicNet.Learn();
                            }
                        }
                        break;
                    case "2":
                        SyntheticBenchmark.RunDemo();
                        break;
                    case "3":
                        HiddenOptimizer.RunOptimization(minNeurons: 1, maxNeurons: 20, epochLimit: 5000);
                        break;
                    case "4":
                        var traceNet = new NeuralNet(4, 4, 1);
                        MathTracer.TraceSample(traceNet, new double[] { 1, 1, 1, 0 }, new double[] { 0 });
                        break;
                    case "5":
                        if (activeLogicNet == null)
                        {
                            Console.WriteLine("[!] Training a 20-hidden-neuron 4-input logic model first...");
                            activeLogicNet = new NeuralNet(4, 20, 1);
                            var ds = LogicGateDataset.Get4InputAndDataset();
                            for (int ep = 0; ep < 10000 && !LogicGateExperiment.FitTest(activeLogicNet, ds); ++ep)
                            {
                                for (int r = 0; r < ds.Length; ++r)
                                {
                                    for (int i = 0; i < 4; ++i) activeLogicNet.SetInput(i, ds[r][i]);
                                    activeLogicNet.SetDesiredOutput(0, ds[r][4]);
                                    activeLogicNet.Learn();
                                }
                            }
                            Console.WriteLine("Done.\n");
                        }
                        LogicGateExperiment.InteractiveManualTest(activeLogicNet);
                        break;
                    case "6":
                        AsciiVisualizer.RenderTopology(4, 5, 3);
                        break;
                    case "7":
                        Console.WriteLine("Exiting. Thank you!");
                        return;
                    default:
                        Console.WriteLine("Invalid selection. Please choose an option between 1 and 7.");
                        break;
                }

                Console.WriteLine("\nPress Enter to return to main menu...");
                Console.ReadLine();
                Console.WriteLine();
            }
        }
    }
}
