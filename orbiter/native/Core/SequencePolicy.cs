// Finite-history actor of orbiter/sequence_policy.py (MLP, GRU or LSTM core),
// exported by native_export.write_sequence_policy. PyTorch gate layouts:
// GRU rows (r, z, n); LSTM rows (i, f, g, o).
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace OrbiterRL.Core
{
    public sealed class SequencePolicy
    {
        private static readonly byte[] Magic = Encoding.ASCII.GetBytes("ORBSEQ1\n");
        private readonly Dictionary<string, float[]> tensors = new Dictionary<string, float[]>();
        private readonly Dictionary<string, int[]> shapes = new Dictionary<string, int[]>();
        public readonly Dictionary<string, string> Header = new Dictionary<string, string>();
        public readonly Dictionary<string, double> Contract = new Dictionary<string, double>();
        public readonly string Kind, Architecture, Sha256;
        public readonly int Observations, Actions, Context, Features, Width;
        public readonly double ActionScale;
        private readonly bool scaled;

        public static SequencePolicy Load(string path) { return new SequencePolicy(File.ReadAllBytes(path)); }

        public SequencePolicy(byte[] data)
        {
            using (SHA256 hash = SHA256.Create())
                Sha256 = BitConverter.ToString(hash.ComputeHash(data)).Replace("-", "").ToLowerInvariant();
            using (BinaryReader reader = new BinaryReader(new MemoryStream(data)))
            {
                byte[] magic = reader.ReadBytes(Magic.Length);
                for (int i = 0; i < Magic.Length; i++)
                    if (magic.Length != Magic.Length || magic[i] != Magic[i])
                        throw new InvalidDataException("Not an Orbiter sequence policy file");
                int length = reader.ReadInt32();
                if (length <= 0 || length > 65536) throw new InvalidDataException("Bad policy header");
                foreach (string line in Encoding.UTF8.GetString(reader.ReadBytes(length)).Split('\n'))
                {
                    if (line.Length == 0) continue;
                    int split = line.IndexOf('=');
                    if (split <= 0) throw new InvalidDataException("Bad policy header line");
                    string key = line.Substring(0, split), value = line.Substring(split + 1);
                    Header[key] = value;
                    if (key.StartsWith("contract."))
                        Contract[key.Substring(9)] = double.Parse(value, CultureInfo.InvariantCulture);
                }
                int count = reader.ReadInt32();
                if (count < 1 || count > 64) throw new InvalidDataException("Bad tensor count");
                for (int t = 0; t < count; t++)
                {
                    int nameLength = reader.ReadInt32();
                    if (nameLength <= 0 || nameLength > 256) throw new InvalidDataException("Bad tensor name");
                    string name = Encoding.UTF8.GetString(reader.ReadBytes(nameLength));
                    int ndim = reader.ReadInt32();
                    if (ndim < 1 || ndim > 4) throw new InvalidDataException("Bad tensor rank");
                    int[] shape = new int[ndim];
                    long size = 1;
                    for (int d = 0; d < ndim; d++)
                    {
                        shape[d] = reader.ReadInt32();
                        if (shape[d] < 1 || shape[d] > 65536) throw new InvalidDataException("Bad tensor shape");
                        size *= shape[d];
                    }
                    if (size > 16000000) throw new InvalidDataException("Tensor too large");
                    float[] values = new float[size];
                    for (long i = 0; i < size; i++)
                    {
                        values[i] = reader.ReadSingle();
                        if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                            throw new InvalidDataException("Nonfinite policy weight");
                    }
                    tensors[name] = values;
                    shapes[name] = shape;
                }
                if (reader.BaseStream.Position != data.Length) throw new InvalidDataException("Trailing bytes in policy file");
            }
            Kind = Header["kind"];
            Architecture = Header["architecture"];
            Observations = Int("observations");
            Actions = Int("actions");
            Context = Int("context");
            Features = Observations + Actions + 1;
            ActionScale = double.Parse(Header["action_scale"], CultureInfo.InvariantCulture);
            scaled = Header["action_transform"] == "scaled_tanh";
            if (!scaled && Header["action_transform"] != "tanh") throw new InvalidDataException("Unknown action transform");
            if (double.Parse(Header["deadzone"], CultureInfo.InvariantCulture) != 0)
                throw new InvalidDataException("Action deadzones are not supported");
            if (Architecture == "gru" || Architecture == "lstm")
            {
                Require("actor.encoder.0.weight", -1, Features);
                Width = shapes["actor.encoder.0.weight"][0];
                int gates = Architecture == "gru" ? 3 : 4;
                Require("actor.memory.weight_ih_l0", gates * Width, Width);
                Require("actor.memory.weight_hh_l0", gates * Width, Width);
                Require("actor.memory.bias_ih_l0", gates * Width);
                Require("actor.memory.bias_hh_l0", gates * Width);
            }
            else if (Architecture.StartsWith("mlp"))
            {
                Require("actor.network.0.weight", -1, Features);
                Width = shapes["actor.network.0.weight"][0];
                Require("actor.network.2.weight", Width, Width);
            }
            else throw new InvalidDataException("Unknown architecture " + Architecture);
            Require("actor.output.weight", Actions, Width);
            Require("actor.output.bias", Actions);
        }

        private int Int(string key) { return int.Parse(Header[key], CultureInfo.InvariantCulture); }

        private void Require(string name, params int[] shape)
        {
            int[] actual;
            if (!shapes.TryGetValue(name, out actual) || actual.Length != shape.Length)
                throw new InvalidDataException("Policy lacks " + name);
            for (int i = 0; i < shape.Length; i++)
                if (shape[i] >= 0 && actual[i] != shape[i]) throw new InvalidDataException("Bad shape for " + name);
        }

        public double Constant(string name)
        {
            double value;
            if (!Contract.TryGetValue(name, out value)) throw new InvalidDataException("Policy contract lacks " + name);
            return value;
        }

        private static double Sigmoid(double x) { return 1.0 / (1.0 + Math.Exp(-x)); }

        /// <summary>y = W x + b for a row-major [rows, columns] weight.</summary>
        private double[] Linear(string prefix, double[] x)
        {
            float[] w = tensors[prefix + ".weight"], b = tensors[prefix + ".bias"];
            int rows = b.Length, columns = x.Length;
            double[] y = new double[rows];
            for (int r = 0; r < rows; r++)
            {
                double sum = b[r];
                int offset = r * columns;
                for (int c = 0; c < columns; c++) sum += w[offset + c] * x[c];
                y[r] = sum;
            }
            return y;
        }

        private static double[] MatVec(float[] w, float[] b, double[] x, int rows)
        {
            int columns = x.Length;
            double[] y = new double[rows];
            for (int r = 0; r < rows; r++)
            {
                double sum = b[r];
                int offset = r * columns;
                for (int c = 0; c < columns; c++) sum += w[offset + c] * x[c];
                y[r] = sum;
            }
            return y;
        }

        private static double[] Tanh(double[] x)
        {
            for (int i = 0; i < x.Length; i++) x[i] = Math.Tanh(x[i]);
            return x;
        }

        /// <summary>Actor latent mean for a [context, features] history window.</summary>
        public double[] Latent(float[][] window)
        {
            if (window.Length != Context) throw new ArgumentException("History window length mismatch");
            double[] last;
            if (Architecture.StartsWith("mlp"))
            {
                last = Tanh(Linear("actor.network.2", Tanh(Linear("actor.network.0", ToDouble(window[Context - 1])))));
            }
            else
            {
                int H = Width, gates = Architecture == "gru" ? 3 : 4;
                float[] wih = tensors["actor.memory.weight_ih_l0"], whh = tensors["actor.memory.weight_hh_l0"];
                float[] bih = tensors["actor.memory.bias_ih_l0"], bhh = tensors["actor.memory.bias_hh_l0"];
                double[] h = new double[H], c = new double[H];
                foreach (float[] token in window)
                {
                    double[] x = Tanh(Linear("actor.encoder.0", ToDouble(token)));
                    double[] gi = MatVec(wih, bih, x, gates * H), gh = MatVec(whh, bhh, h, gates * H);
                    double[] next = new double[H];
                    if (gates == 3)
                    {
                        for (int k = 0; k < H; k++)
                        {
                            double r = Sigmoid(gi[k] + gh[k]);
                            double z = Sigmoid(gi[H + k] + gh[H + k]);
                            double n = Math.Tanh(gi[2 * H + k] + r * gh[2 * H + k]);
                            next[k] = (1 - z) * n + z * h[k];
                        }
                    }
                    else
                    {
                        for (int k = 0; k < H; k++)
                        {
                            double i = Sigmoid(gi[k] + gh[k]);
                            double f = Sigmoid(gi[H + k] + gh[H + k]);
                            double g = Math.Tanh(gi[2 * H + k] + gh[2 * H + k]);
                            double o = Sigmoid(gi[3 * H + k] + gh[3 * H + k]);
                            c[k] = f * c[k] + i * g;
                            next[k] = o * Math.Tanh(c[k]);
                        }
                    }
                    h = next;
                }
                last = h;
            }
            return Linear("actor.output", last);
        }

        /// <summary>Deterministic executed action: clamp(scale * tanh(latent), -1, 1).</summary>
        public double[] Act(float[][] window)
        {
            double[] z = Latent(window);
            for (int i = 0; i < z.Length; i++)
                z[i] = scaled ? Math.Max(-1.0, Math.Min(1.0, ActionScale * Math.Tanh(z[i]))) : Math.Tanh(z[i]);
            return z;
        }

        private static double[] ToDouble(float[] x)
        {
            double[] y = new double[x.Length];
            for (int i = 0; i < x.Length; i++) y[i] = x[i];
            return y;
        }
    }

    /// <summary>
    /// sequence_policy.History for one world: trailing tokens of
    /// [observation, executed previous action (zero after reset), valid=1].
    /// </summary>
    public sealed class PolicyHistory
    {
        private readonly int observations, actions;
        public readonly float[][] Window;

        public PolicyHistory(int context, int observations, int actions)
        {
            this.observations = observations;
            this.actions = actions;
            Window = new float[context][];
            for (int i = 0; i < context; i++) Window[i] = new float[observations + actions + 1];
        }

        public void Append(float[] observation, double[] previousAction)
        {
            if (observation.Length != observations) throw new ArgumentException("history observation size mismatch");
            for (int i = 0; i + 1 < Window.Length; i++) Window[i] = Window[i + 1];
            float[] token = new float[observations + actions + 1];
            Array.Copy(observation, token, observations);
            if (previousAction != null)
                for (int i = 0; i < actions; i++) token[observations + i] = (float)previousAction[i];
            token[observations + actions] = 1f;
            Window[Window.Length - 1] = token;
        }
    }
}
