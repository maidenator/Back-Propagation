using System;
using System.IO;
using System.Text;
using BackPropagation.Core;

namespace BackPropagation.SelfDoc
{
    /// <summary>
    /// Autonomous Markdown report generator.
    /// Creates persistent, structured documentation summarizing training runs, metrics, and convergence results.
    /// </summary>
    public static class MarkdownReporter
    {
        /// <summary>
        /// Creates a unique Markdown report path inside the generated <c>reports/</c> directory.
        /// Millisecond precision plus a collision suffix prevents reports from overwriting
        /// each other when several benchmarks finish within the same second.
        /// </summary>
        public static string CreateReportPath(string prefix)
        {
            Directory.CreateDirectory("reports");
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            string filePath = Path.Combine("reports", $"{prefix}_{timestamp}.md");

            int suffix = 2;
            while (File.Exists(filePath))
            {
                filePath = Path.Combine("reports", $"{prefix}_{timestamp}_{suffix}.md");
                ++suffix;
            }

            return filePath;
        }

        public static string GenerateLogicGateReport(NeuralNet net, double[][] dataset, 
            int epochsTrained, bool isConverged, string taskName = "4-Input Logic Gate")
        {
            string filePath = CreateReportPath("run");

            var sb = new StringBuilder();
            sb.AppendLine($"# Neural Network Training Report: {taskName}");
            sb.AppendLine();
            sb.AppendLine($"**Generated:** {DateTime.Now:yyyy-MM-dd HH:mm:ss}  ");
            sb.AppendLine($"**Architecture:** {net.InputCount} Inputs → {net.HiddenCount} Hidden Nodes → {net.OutputCount} Outputs  ");
            sb.AppendLine($"**Learning Rates:** Output Layer η = {net.LrpOut:F2}, Hidden Layer η = {net.LrpIn:F2}  ");
            sb.AppendLine($"**Epochs Completed:** {epochsTrained}  ");
            sb.AppendLine($"**Final Status:** {(isConverged ? "✅ **CONVERGED (All Samples Passed)**" : "⚠️ **Incomplete / Not Fully Converged**")}  ");
            sb.AppendLine();

            sb.AppendLine("## 1. Network Topology");
            sb.AppendLine("```");
            sb.AppendLine($"[Input Layer: {net.InputCount}] ──(weights)──> [Hidden Layer: {net.HiddenCount}] ──(weights)──> [Output Layer: {net.OutputCount}]");
            sb.AppendLine("```");
            sb.AppendLine();

            sb.AppendLine("## 2. Truth Table Verification & Results");
            sb.AppendLine("| # | Input (x0, x1, x2, x3) | Desired Target | Predicted (Raw) | Predicted (Rounded) | Status |");
            sb.AppendLine("|---|:---------------------:|:--------------:|:---------------:|:-------------------:|:------:|");

            int passed = 0;
            for (int r = 0; r < dataset.Length; ++r)
            {
                for (int i = 0; i < net.InputCount; ++i)
                    net.SetInput(i, dataset[r][i]);

                net.Run();
                double rawOut = net.GetOutputData(0);
                int roundedOut = (int)Math.Round(rawOut);
                int target = (int)dataset[r][net.InputCount];
                bool correct = (roundedOut == target);
                if (correct) ++passed;

                string status = correct ? "✅ PASS" : "❌ FAIL";
                string inputStr = $"({dataset[r][0]}, {dataset[r][1]}, {dataset[r][2]}, {dataset[r][3]})";
                sb.AppendLine($"| {r,2} | `{inputStr}` | `{target}` | `{rawOut:F4}` | `{roundedOut}` | {status} |");
            }
            sb.AppendLine();
            sb.AppendLine($"**Final Accuracy:** {passed}/{dataset.Length} ({(double)passed / dataset.Length * 100.0:F1}%)");
            sb.AppendLine();

            sb.AppendLine("## 3. Mathematical Equations Applied");
            sb.AppendLine("- **Hidden Activation:** $h_j = \\sigma\\left(\\sum_{i} x_i w_{ij} + b_j\\right) = \\frac{1}{1 + e^{-z_j}}$");
            sb.AppendLine("- **Output Activation:** $y_k = \\sigma\\left(\\sum_{j} h_j w_{jk} + b_k\\right)$");
            sb.AppendLine("- **Output Error Gradient:** $\\delta_k = (t_k - y_k) \\cdot y_k(1 - y_k)$");
            sb.AppendLine("- **Hidden Error Gradient:** $\\delta_j = h_j(1 - h_j) \\cdot \\sum_{k} \\delta_k w_{jk}$");
            sb.AppendLine("- **Weight Updates:** $\\Delta w = \\eta \\cdot \\delta \\cdot \\text{input}$");
            sb.AppendLine();

            sb.AppendLine("## 4. Final Learned Biases");
            sb.AppendLine("**Hidden Layer Biases:**");
            sb.Append("`[ ");
            for (int h = 0; h < net.HiddenCount; ++h)
                sb.Append($"{net.HiddenNeurons[h].Bias:F4} ");
            sb.AppendLine("]`\n");

            sb.AppendLine("**Output Layer Biases:**");
            sb.Append("`[ ");
            for (int o = 0; o < net.OutputCount; ++o)
                sb.Append($"{net.OutputNeurons[o].Bias:F4} ");
            sb.AppendLine("]`\n");

            File.WriteAllText(filePath, sb.ToString());
            return filePath;
        }
    }
}
