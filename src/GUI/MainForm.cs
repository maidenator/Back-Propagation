#if NET8_0_WINDOWS || WINDOWS
using System;
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using BackPropagation.Core;
using BackPropagation.Data;

namespace BackPropagation.GUI
{
    /// <summary>
    /// Windows Forms user interface replicating and extending the CS345 Backpropagation assignment.
    /// Provides controls for generating synthetic data, configuring hyperparameters,
    /// selecting between Mean Squared Error and Cross-Entropy, applying Weight Decay (L2 regularization),
    /// observing real-time training progress, making predictions, and exporting a training report.
    /// </summary>
    public class MainForm : Form
    {
        // Hyperparameter Controls
        private NumericUpDown nudInputs = null!;
        private NumericUpDown nudHidden = null!;
        private NumericUpDown nudOutputs = null!;
        private NumericUpDown nudRows = null!;
        private NumericUpDown nudEpochs = null!;
        private TextBox txtLearnRate = null!;
        private TextBox txtMomentum = null!;
        private ComboBox cmbLossFunction = null!;
        private CheckBox chkWeightDecay = null!;
        private TextBox txtDecayRate = null!;

        // Action Buttons
        private Button btnGenerateData = null!;
        private Button btnCreateNet = null!;
        private Button btnTrain = null!;
        private Button btnPredict = null!;
        private Button btnMathTrace = null!;
        private Button btnExportReport = null!;

        // Prediction Inputs
        private TextBox txtIn0 = null!;
        private TextBox txtIn1 = null!;
        private TextBox txtIn2 = null!;
        private TextBox txtIn3 = null!;
        private Label lblPredictionResult = null!;

        // Status & Output
        private TextBox txtConsoleLog = null!;
        private Label lblTrainAcc = null!;
        private Label lblTestAcc = null!;
        private Label lblStatus = null!;
        private TextBox txtWeights = null!;

        // Internal State
        private double[][]? allData;
        private double[][]? trainData;
        private double[][]? testData;
        private MatrixNeuralNet? neuralNet;

        public MainForm()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            Text = "CS345 IS 1: Neural Network Back-Propagation (With Cross-Entropy & L2 Regularization)";
            Size = new Size(1180, 820);
            StartPosition = FormStartPosition.CenterScreen;
            Font = new Font("Segoe UI", 9F, FontStyle.Regular, GraphicsUnit.Point);
            BackColor = Color.FromArgb(245, 247, 250);

            // Main Table Layout (Left: Controls & Predict, Right: Logs & Results)
            var mainLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                Padding = new Padding(12)
            };
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 430F));
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));

            // Left Side Panel
            var leftPanel = new Panel
            {
                Dock = DockStyle.Fill,
                AutoScroll = true
            };

            // Right Side Panel
            var rightPanel = new Panel
            {
                Dock = DockStyle.Fill
            };

            mainLayout.Controls.Add(leftPanel, 0, 0);
            mainLayout.Controls.Add(rightPanel, 1, 0);
            Controls.Add(mainLayout);

            BuildLeftControls(leftPanel);
            BuildRightControls(rightPanel);
        }

        private void BuildLeftControls(Panel panel)
        {
            int top = 0;

            // 1. GroupBox: Architecture & Hyperparameters
            var grpHyper = new GroupBox
            {
                Text = "1. Architecture & Hyperparameters",
                Location = new Point(0, top),
                Size = new Size(410, 245),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };
            panel.Controls.Add(grpHyper);

            // The CS345 demo uses a fixed 4-input / 3-output classification topology.
            // The hidden-layer width remains configurable.
            AddLabelAndControl(grpHyper, "Inputs:", nudInputs = new NumericUpDown { Minimum = 4, Maximum = 4, Value = 4, Width = 60 }, 15, 25);
            AddLabelAndControl(grpHyper, "Hidden:", nudHidden = new NumericUpDown { Minimum = 1, Maximum = 50, Value = 5, Width = 60 }, 145, 25);
            AddLabelAndControl(grpHyper, "Outputs:", nudOutputs = new NumericUpDown { Minimum = 3, Maximum = 3, Value = 3, Width = 60 }, 275, 25);

            AddLabelAndControl(grpHyper, "Total Rows:", nudRows = new NumericUpDown { Minimum = 50, Maximum = 10000, Value = 1000, Increment = 100, Width = 80 }, 15, 60);
            AddLabelAndControl(grpHyper, "Max Epochs:", nudEpochs = new NumericUpDown { Minimum = 10, Maximum = 50000, Value = 1000, Increment = 100, Width = 80 }, 205, 60);

            AddLabelAndControl(grpHyper, "Learn Rate (η):", txtLearnRate = new TextBox { Text = "0.05", Width = 80 }, 15, 95);
            AddLabelAndControl(grpHyper, "Momentum (α):", txtMomentum = new TextBox { Text = "0.01", Width = 80 }, 205, 95);

            // Loss Function selector (WHAT TO DO extension)
            var lblLoss = new Label { Text = "Loss Metric:", Location = new Point(15, 135), AutoSize = true, Font = new Font("Segoe UI", 8.5F, FontStyle.Regular) };
            cmbLossFunction = new ComboBox
            {
                Location = new Point(115, 132),
                Width = 275,
                DropDownStyle = ComboBoxStyle.DropDownList,
                Font = new Font("Segoe UI", 8.5F, FontStyle.Regular)
            };
            cmbLossFunction.Items.Add("Mean Squared Error (MSE)");
            cmbLossFunction.Items.Add("Cross-Entropy Error (from 'WHAT TO DO')");
            cmbLossFunction.SelectedIndex = 0;
            grpHyper.Controls.Add(lblLoss);
            grpHyper.Controls.Add(cmbLossFunction);

            // Weight Decay (L2 Regularization) (WHAT TO DO extension)
            chkWeightDecay = new CheckBox
            {
                Text = "Enable Weight Decay (L2 Regularization)",
                Location = new Point(15, 172),
                AutoSize = true,
                Font = new Font("Segoe UI", 8.5F, FontStyle.Regular)
            };
            var lblDecay = new Label { Text = "Decay Rate (λ):", Location = new Point(15, 205), AutoSize = true, Font = new Font("Segoe UI", 8.5F, FontStyle.Regular) };
            txtDecayRate = new TextBox { Text = "0.001", Location = new Point(125, 202), Width = 80, Font = new Font("Segoe UI", 8.5F, FontStyle.Regular) };
            grpHyper.Controls.Add(chkWeightDecay);
            grpHyper.Controls.Add(lblDecay);
            grpHyper.Controls.Add(txtDecayRate);

            top += 255;

            // 2. GroupBox: Action Buttons
            var grpActions = new GroupBox
            {
                Text = "2. Model Operations",
                Location = new Point(0, top),
                Size = new Size(410, 160),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };
            panel.Controls.Add(grpActions);

            btnGenerateData = new Button { Text = "1. Generate Data (1000)", Location = new Point(15, 25), Size = new Size(180, 36) };
            btnCreateNet = new Button { Text = "2. Create 4-5-3 Network", Location = new Point(210, 25), Size = new Size(180, 36) };
            btnTrain = new Button { Text = "3. Train Model (Backprop)", Location = new Point(15, 68), Size = new Size(180, 38), BackColor = Color.FromArgb(41, 128, 185), ForeColor = Color.White };
            btnMathTrace = new Button { Text = "4. Trace Math", Location = new Point(210, 68), Size = new Size(180, 38) };
            btnExportReport = new Button { Text = "5. Export Markdown Report", Location = new Point(15, 112), Size = new Size(375, 34) };

            btnGenerateData.Click += BtnGenerateData_Click;
            btnCreateNet.Click += BtnCreateNet_Click;
            btnTrain.Click += BtnTrain_Click;
            btnMathTrace.Click += BtnMathTrace_Click;
            btnExportReport.Click += BtnExportReport_Click;

            grpActions.Controls.Add(btnGenerateData);
            grpActions.Controls.Add(btnCreateNet);
            grpActions.Controls.Add(btnTrain);
            grpActions.Controls.Add(btnMathTrace);
            grpActions.Controls.Add(btnExportReport);

            top += 170;

            // 3. GroupBox: Predict on Custom Inputs
            var grpPredict = new GroupBox
            {
                Text = "3. Prediction & Classification",
                Location = new Point(0, top),
                Size = new Size(410, 185),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };
            panel.Controls.Add(grpPredict);

            var lblPrompt = new Label { Text = "Enter 4 predictor values (e.g. 1.0, 2.0, 3.0, 4.0):", Location = new Point(15, 25), AutoSize = true, Font = new Font("Segoe UI", 8.5F, FontStyle.Regular) };
            grpPredict.Controls.Add(lblPrompt);

            txtIn0 = new TextBox { Text = "1.0", Location = new Point(15, 50), Width = 80 };
            txtIn1 = new TextBox { Text = "2.0", Location = new Point(110, 50), Width = 80 };
            txtIn2 = new TextBox { Text = "3.0", Location = new Point(205, 50), Width = 80 };
            txtIn3 = new TextBox { Text = "4.0", Location = new Point(300, 50), Width = 80 };
            grpPredict.Controls.Add(txtIn0);
            grpPredict.Controls.Add(txtIn1);
            grpPredict.Controls.Add(txtIn2);
            grpPredict.Controls.Add(txtIn3);

            btnPredict = new Button { Text = "Classify Input", Location = new Point(15, 85), Size = new Size(365, 34) };
            btnPredict.Click += BtnPredict_Click;
            grpPredict.Controls.Add(btnPredict);

            lblPredictionResult = new Label
            {
                Text = "Prediction: [Ready]",
                Location = new Point(15, 128),
                Size = new Size(375, 45),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                ForeColor = Color.FromArgb(44, 62, 80)
            };
            grpPredict.Controls.Add(lblPredictionResult);
        }

        private void BuildRightControls(Panel panel)
        {
            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                RowCount = 3,
                ColumnCount = 1
            };
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 50F));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 65F));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 35F));

            // Top Status Bar
            var pnlStatus = new Panel { Dock = DockStyle.Fill };
            lblStatus = new Label { Text = "Status: Initialized. Click '1. Generate Data' to begin.", Location = new Point(5, 5), AutoSize = true, Font = new Font("Segoe UI", 9.5F, FontStyle.Bold) };
            lblTrainAcc = new Label { Text = "Train Accuracy: N/A", Location = new Point(5, 28), AutoSize = true, ForeColor = Color.DarkGreen };
            lblTestAcc = new Label { Text = "Test Accuracy: N/A", Location = new Point(240, 28), AutoSize = true, ForeColor = Color.DarkBlue };
            pnlStatus.Controls.Add(lblStatus);
            pnlStatus.Controls.Add(lblTrainAcc);
            pnlStatus.Controls.Add(lblTestAcc);
            layout.Controls.Add(pnlStatus, 0, 0);

            // Middle: Console Log
            var grpLog = new GroupBox
            {
                Text = "Training Log & Mathematical Execution Output",
                Dock = DockStyle.Fill,
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };
            txtConsoleLog = new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ScrollBars = ScrollBars.Both,
                ReadOnly = true,
                Font = new Font("Consolas", 9F, FontStyle.Regular),
                BackColor = Color.FromArgb(30, 30, 30),
                ForeColor = Color.FromArgb(220, 220, 220)
            };
            grpLog.Controls.Add(txtConsoleLog);
            layout.Controls.Add(grpLog, 0, 1);

            // Bottom: Weights & Biases
            var grpWeights = new GroupBox
            {
                Text = "Model Parameters (Weights & Biases Serialization)",
                Dock = DockStyle.Fill,
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };
            txtWeights = new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ScrollBars = ScrollBars.Vertical,
                ReadOnly = true,
                Font = new Font("Consolas", 8.5F, FontStyle.Regular),
                BackColor = Color.White
            };
            grpWeights.Controls.Add(txtWeights);
            layout.Controls.Add(grpWeights, 0, 2);

            panel.Controls.Add(layout);
        }

        private void AddLabelAndControl(GroupBox grp, string labelText, Control ctrl, int x, int y)
        {
            var lbl = new Label
            {
                Text = labelText,
                Location = new Point(x, y + 2),
                AutoSize = true,
                Font = new Font("Segoe UI", 8.5F, FontStyle.Regular)
            };
            ctrl.Location = new Point(x + lbl.PreferredWidth + 5, y);
            grp.Controls.Add(lbl);
            grp.Controls.Add(ctrl);
        }

        private void Log(string message)
        {
            txtConsoleLog.AppendText(message + Environment.NewLine);
        }

        private void BtnGenerateData_Click(object? sender, EventArgs e)
        {
            try
            {
                int numInput = (int)nudInputs.Value;
                int numHidden = (int)nudHidden.Value;
                int numOutput = (int)nudOutputs.Value;
                int numRows = (int)nudRows.Value;
                int seed = 1;

                Log($"\n>>> Generating {numRows} synthetic data items with {numInput} features...");
                allData = SyntheticDataset.MakeAllData(numInput, numHidden, numOutput, numRows, seed);
                SyntheticDataset.SplitTrainTest(allData, 0.80, seed, out trainData, out testData);

                Log($"Data successfully partitioned: {trainData.Length} training items (80%), {testData.Length} test items (20%).");
                Log("Sample Training Items (first 3 rows):");
                for (int r = 0; r < 3; ++r)
                {
                    string inStr = $"({trainData[r][0]:F2}, {trainData[r][1]:F2}, {trainData[r][2]:F2}, {trainData[r][3]:F2})";
                    string tStr = $"({trainData[r][4]}, {trainData[r][5]}, {trainData[r][6]})";
                    Log($"  [{r}] Inputs: {inStr} | Target: {tStr}");
                }

                lblStatus.Text = $"Status: Dataset generated ({numRows} items).";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error generating data: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void BtnCreateNet_Click(object? sender, EventArgs e)
        {
            try
            {
                int numInput = (int)nudInputs.Value;
                int numHidden = (int)nudHidden.Value;
                int numOutput = (int)nudOutputs.Value;

                neuralNet = new MatrixNeuralNet(numInput, numHidden, numOutput, seed: 0);

                Log($"\n>>> Instantiated {numInput}-{numHidden}-{numOutput} neural network.");
                int totalParams = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
                Log($"Total parameters to optimize: {totalParams} ({numInput * numHidden + numHidden * numOutput} weights, {numHidden + numOutput} biases).");

                UpdateWeightsDisplay();
                lblStatus.Text = $"Status: Network created ({numInput}-{numHidden}-{numOutput}). Ready to train.";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error creating network: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private async void BtnTrain_Click(object? sender, EventArgs e)
        {
            if (trainData == null)
            {
                MessageBox.Show("Please generate the dataset first.", "Notice", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            if (neuralNet == null)
            {
                BtnCreateNet_Click(sender, e);
            }

            try
            {
                btnTrain.Enabled = false;
                btnGenerateData.Enabled = false;
                btnCreateNet.Enabled = false;

                int maxEpochs = (int)nudEpochs.Value;
                double learnRate = double.Parse(txtLearnRate.Text);
                double momentum = double.Parse(txtMomentum.Text);
                LossFunction lossFunc = (cmbLossFunction.SelectedIndex == 1)
                    ? LossFunction.CrossEntropy
                    : LossFunction.MeanSquaredError;

                double weightDecay = 0.0;
                if (chkWeightDecay.Checked)
                    weightDecay = double.Parse(txtDecayRate.Text);

                Log($"\n=======================================================");
                Log($"Starting Training:");
                Log($"  Epochs: {maxEpochs} | Learn Rate: {learnRate:F2} | Momentum: {momentum:F2}");
                Log($"  Loss Function: {lossFunc}");
                Log($"  Weight Decay (L2): {(chkWeightDecay.Checked ? weightDecay.ToString("F4") : "Disabled")}");
                Log($"=======================================================");

                lblStatus.Text = "Status: Training in progress...";

                await Task.Run(() =>
                {
                    neuralNet!.Train(trainData, maxEpochs, learnRate, momentum, lossFunc, weightDecay, (epoch, loss) =>
                    {
                        Invoke((Action)(() =>
                        {
                            Log($"  Epoch {epoch,5} | Loss = {loss:F4}");
                        }));
                    });
                });

                double trainAcc = neuralNet!.Accuracy(trainData);
                double testAcc = neuralNet.Accuracy(testData!);

                lblTrainAcc.Text = $"Train Accuracy: {trainAcc * 100.0:F2}%";
                lblTestAcc.Text = $"Test Accuracy: {testAcc * 100.0:F2}%";
                lblStatus.Text = "Status: Training complete!";

                Log("-------------------------------------------------------");
                Log($"Final Accuracy on Training Data (N={trainData.Length}): {trainAcc * 100.0:F2}%");
                Log($"Final Accuracy on Test Data     (N={testData!.Length}): {testAcc * 100.0:F2}%");
                Log("-------------------------------------------------------\n");

                UpdateWeightsDisplay();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Training error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                btnTrain.Enabled = true;
                btnGenerateData.Enabled = true;
                btnCreateNet.Enabled = true;
            }
        }

        private void BtnPredict_Click(object? sender, EventArgs e)
        {
            if (neuralNet == null)
            {
                MessageBox.Show("Please create and train the model first.", "Notice", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            try
            {
                double[] x = new double[]
                {
                    double.Parse(txtIn0.Text),
                    double.Parse(txtIn1.Text),
                    double.Parse(txtIn2.Text),
                    double.Parse(txtIn3.Text)
                };

                double[] outputs = neuralNet.ComputeOutputs(x);
                int predictedClass = MatrixNeuralNet.MaxIndex(outputs);

                string[] classLabels = new string[] { "Class 0 (Conservative)", "Class 1 (Moderate)", "Class 2 (Liberal)" };
                string labelName = (predictedClass < classLabels.Length) ? classLabels[predictedClass] : $"Class {predictedClass}";

                lblPredictionResult.Text = $"Prediction: {labelName}\n" +
                    $"Softmax Probabilities: [{outputs[0]:F4}, {outputs[1]:F4}, {outputs[2]:F4}]";

                Log($"\n[PREDICTION] Input: ({x[0]}, {x[1]}, {x[2]}, {x[3]})");
                Log($"  Output Probabilities: [{outputs[0]:F4}, {outputs[1]:F4}, {outputs[2]:F4}]");
                Log($"  Predicted: {labelName}");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Prediction error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void BtnMathTrace_Click(object? sender, EventArgs e)
        {
            if (neuralNet == null)
            {
                MessageBox.Show("Please create the network first.", "Notice", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            Log("\n=======================================================");
            Log("       STEP-BY-STEP MATHEMATICAL TRACE (SAMPLE PASS)   ");
            Log("=======================================================");

            double[] sampleX = new double[] { 1.0, 2.0, 3.0, 4.0 };
            double[] actualOut = neuralNet.ComputeOutputs(sampleX);

            Log($"Input x: [{string.Join(", ", sampleX)}]");
            Log("Hidden Layer Weighted Sums & Activations (Tanh):");
            for (int j = 0; j < neuralNet.NumHidden; ++j)
            {
                double sum = 0.0;
                for (int i = 0; i < neuralNet.NumInput; ++i)
                    sum += sampleX[i] * neuralNet.IhWeights[i][j];
                sum += neuralNet.HBiases[j];
                Log($"  Node h{j}: sum = {sum:F4}, tanh(sum) = {neuralNet.HOutputs[j]:F4}");
            }

            Log("Output Layer Weighted Sums & Probabilities (Softmax):");
            for (int k = 0; k < neuralNet.NumOutput; ++k)
            {
                Log($"  Node o{k}: Softmax prob = {actualOut[k]:F4}");
            }

            Log($"Predicted Class: {MatrixNeuralNet.MaxIndex(actualOut)}");
            Log("=======================================================\n");
        }

        private void BtnExportReport_Click(object? sender, EventArgs e)
        {
            if (neuralNet == null || trainData == null)
            {
                MessageBox.Show("Please generate data and train the model before exporting.", "Notice", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            try
            {
                Directory.CreateDirectory("reports");
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string path = Path.Combine("reports", $"cs345_report_{timestamp}.md");

                double trainAcc = neuralNet.Accuracy(trainData);
                double testAcc = neuralNet.Accuracy(testData!);

                var sb = new StringBuilder();
                sb.AppendLine("# CS345 Backpropagation Neural Network Training Report");
                sb.AppendLine($"**Date:** {DateTime.Now:yyyy-MM-dd HH:mm:ss}  ");
                sb.AppendLine($"**Architecture:** {neuralNet.NumInput}-{neuralNet.NumHidden}-{neuralNet.NumOutput}  ");
                sb.AppendLine($"**Loss Function:** {cmbLossFunction.SelectedItem}  ");
                sb.AppendLine($"**Weight Decay (L2):** {(chkWeightDecay.Checked ? txtDecayRate.Text : "None")}  ");
                sb.AppendLine($"**Training Accuracy:** {trainAcc * 100.0:F2}% (N={trainData.Length})  ");
                sb.AppendLine($"**Test Accuracy:** {testAcc * 100.0:F2}% (N={testData!.Length})  ");
                sb.AppendLine();
                sb.AppendLine("## Final Serialized Model Weights");
                sb.AppendLine("```");
                double[] w = neuralNet.GetWeights();
                for (int i = 0; i < w.Length; ++i)
                {
                    sb.Append($"{w[i],7:F2} ");
                    if ((i + 1) % 10 == 0) sb.AppendLine();
                }
                sb.AppendLine("\n```");

                File.WriteAllText(path, sb.ToString());
                Log($"\n[REPORT] Report successfully saved to: {path}");
                MessageBox.Show($"Report exported successfully to:\n{path}", "Export Complete", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Export error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void UpdateWeightsDisplay()
        {
            if (neuralNet == null) return;
            double[] weights = neuralNet.GetWeights();
            var sb = new StringBuilder();
            sb.AppendLine($"// Serialized Model Parameters (Total: {weights.Length})");
            for (int i = 0; i < weights.Length; ++i)
            {
                sb.Append($"{weights[i],7:F2} ");
                if ((i + 1) % 10 == 0) sb.AppendLine();
            }
            txtWeights.Text = sb.ToString();
        }
    }
}
#endif
