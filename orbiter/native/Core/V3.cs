// Double-precision 3-vector for the Unity-free core. The core always works in
// one right-handed, nonrotating, body-centred frame; the KSP adapter converts.
using System;

namespace OrbiterRL.Core
{
    public struct V3
    {
        public readonly double X, Y, Z;

        public V3(double x, double y, double z) { X = x; Y = y; Z = z; }

        public static readonly V3 Zero = new V3(0, 0, 0);

        public static V3 operator +(V3 a, V3 b) { return new V3(a.X + b.X, a.Y + b.Y, a.Z + b.Z); }
        public static V3 operator -(V3 a, V3 b) { return new V3(a.X - b.X, a.Y - b.Y, a.Z - b.Z); }
        public static V3 operator -(V3 a) { return new V3(-a.X, -a.Y, -a.Z); }
        public static V3 operator *(double s, V3 a) { return new V3(s * a.X, s * a.Y, s * a.Z); }
        public static V3 operator *(V3 a, double s) { return new V3(s * a.X, s * a.Y, s * a.Z); }
        public static V3 operator /(V3 a, double s) { return new V3(a.X / s, a.Y / s, a.Z / s); }

        public static double Dot(V3 a, V3 b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }

        public static V3 Cross(V3 a, V3 b)
        {
            return new V3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
        }

        public double Norm { get { return Math.Sqrt(X * X + Y * Y + Z * Z); } }

        public V3 Unit()
        {
            double norm = Norm;
            if (norm < 1e-10) throw new ArgumentException("Undefined direction");
            return this / norm;
        }

        public bool Finite
        {
            get { return !(double.IsNaN(X) || double.IsNaN(Y) || double.IsNaN(Z) ||
                           double.IsInfinity(X) || double.IsInfinity(Y) || double.IsInfinity(Z)); }
        }

        public override string ToString()
        {
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                                 "[{0:R}, {1:R}, {2:R}]", X, Y, Z);
        }
    }

    /// <summary>Columns radial, transverse, normal (Python ``rtn_basis``).</summary>
    public struct Rtn
    {
        public readonly V3 R, T, N;

        public Rtn(V3 r, V3 v)
        {
            R = r / r.Norm;
            V3 h = V3.Cross(r, v);
            double norm = h.Norm;
            if (norm < 1e-12)
            {
                double ax = Math.Abs(R.X), ay = Math.Abs(R.Y), az = Math.Abs(R.Z);
                V3 axis = ax <= ay && ax <= az ? new V3(1, 0, 0)
                        : ay <= az ? new V3(0, 1, 0) : new V3(0, 0, 1);
                N = V3.Cross(R, axis).Unit();
            }
            else N = h / norm;
            T = V3.Cross(N, R);
        }

        public V3 ToLocal(V3 a) { return new V3(V3.Dot(R, a), V3.Dot(T, a), V3.Dot(N, a)); }
        public V3 ToFrame(V3 a) { return a.X * R + a.Y * T + a.Z * N; }
    }
}
