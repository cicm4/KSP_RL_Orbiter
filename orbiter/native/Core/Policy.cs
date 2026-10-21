// Deterministic tanh-MLP actor exported by orbiter/native_export.py.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace OrbiterRL.Core
{
    public sealed class Policy
    {
        private static readonly byte[] Magic = Encoding.ASCII.GetBytes("ORBPOL1\n");
        private readonly float[][] weights, biases;
        private readonly int[] rows, columns;
        public readonly string Kind, Sha256;
        public readonly int Observations, Actions;
        public readonly Dictionary<string, double> Contract = new Dictionary<string, double>();
        public readonly Dictionary<string, string> Header = new Dictionary<string, string>();

        public static Policy Load(string path) { return new Policy(File.ReadAllBytes(path)); }

        public Policy(byte[] data)
        {
            using (SHA256 hash = SHA256.Create())
                Sha256 = BitConverter.ToString(hash.ComputeHash(data)).Replace("-", "").ToLowerInvariant();
            using (BinaryReader reader = new BinaryReader(new MemoryStream(data)))
            {
                byte[] magic = reader.ReadBytes(Magic.Length);
                for (int i = 0; i < Magic.Length; i++)
                    if (magic.Length != Magic.Length || magic[i] != Magic[i])
                        throw new InvalidDataException("Not an Orbiter policy file");
                int length = reader.ReadInt32();
                if (length <= 0 || length > 65536) throw new InvalidDataException("Bad policy header");
                string text = Encoding.UTF8.GetString(reader.ReadBytes(length));
                foreach (string line in text.Split('\n'))
                {
                    if (line.Length == 0) continue;
                    int split = line.IndexOf('=');
                    if (split <= 0) throw new InvalidDataException("Bad policy header line");
                    string key = line.Substring(0, split), value = line.Substring(split + 1);
                    Header[key] = value;
                    if (key.StartsWith("contract."))
                        Contract[key.Substring(9)] = double.Parse(value, CultureInfo.InvariantCulture);
                }
                Kind = Header["kind"];
                int count = reader.ReadInt32();
                if (count < 1 || count > 8) throw new InvalidDataException("Bad policy layer count");
                weights = new float[count][]; biases = new float[count][];
                rows = new int[count]; columns = new int[count];
                for (int layer = 0; layer < count; layer++)
                {
                    rows[layer] = reader.ReadInt32(); columns[layer] = reader.ReadInt32();
                    if (rows[layer] < 1 || columns[layer] < 1 || rows[layer] > 4096 || columns[layer] > 4096)
                        throw new InvalidDataException("Bad policy layer shape");
                    if (layer > 0 && columns[layer] != rows[layer - 1])
                        throw new InvalidDataException("Policy layers do not chain");
                    weights[layer] = ReadFloats(reader, rows[layer] * columns[layer]);
                    biases[layer] = ReadFloats(reader, rows[layer]);
                }
                if (reader.BaseStream.Position != data.Length)
                    throw new InvalidDataException("Trailing bytes in policy file");
                Observations = columns[0]; Actions = rows[count - 1];
                if (Observations != int.Parse(Header["observations"], CultureInfo.InvariantCulture) ||
                    Actions != int.Parse(Header["actions"], CultureInfo.InvariantCulture))
                    throw new InvalidDataException("Policy header disagrees with its layers");
            }
        }

        private static float[] ReadFloats(BinaryReader reader, int count)
        {
            float[] values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = reader.ReadSingle();
                if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                    throw new InvalidDataException("Nonfinite policy weight");
            }
            return values;
        }

        public double Constant(string name)
        {
            double value;
            if (!Contract.TryGetValue(name, out value))
                throw new InvalidDataException("Policy contract lacks " + name);
            return value;
        }

        /// <summary>Deterministic action, clipped to [-1, 1] like SB3 predict.</summary>
        public double[] Act(float[] observation)
        {
            if (observation.Length != Observations) throw new ArgumentException("Observation size mismatch");
            double[] x = new double[observation.Length];
            for (int i = 0; i < x.Length; i++) x[i] = observation[i];
            for (int layer = 0; layer < weights.Length; layer++)
            {
                double[] y = new double[rows[layer]];
                float[] w = weights[layer];
                int width = columns[layer];
                for (int r = 0; r < y.Length; r++)
                {
                    double sum = biases[layer][r];
                    int offset = r * width;
                    for (int c = 0; c < width; c++) sum += w[offset + c] * x[c];
                    y[r] = layer < weights.Length - 1 ? Math.Tanh(sum) : Math.Max(-1.0, Math.Min(1.0, sum));
                }
                x = y;
            }
            return x;
        }
    }
}
