// Physics-tick coroutine runner with nested IEnumerators. Unlike Unity's
// StartCoroutine, Abort() disposes every active iterator, so each phase's
// finally block (throttle zero, pointing released, node removal) always runs.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using OrbiterRL.Core;

namespace OrbiterRL
{
    /// <summary>Yield to resume at or after a universal time.</summary>
    public sealed class WaitUt
    {
        public readonly double Ut;
        public WaitUt(double ut) { Ut = ut; }
    }

    public sealed class Runner
    {
        private readonly Stack<IEnumerator> stack = new Stack<IEnumerator>();
        private WaitUt waiting;

        public bool Running { get { return stack.Count > 0; } }
        public Exception Failure { get; private set; }

        public void Start(IEnumerator root)
        {
            if (Running) throw new InvalidOperationException("A mission is already running");
            Failure = null;
            waiting = null;
            stack.Push(root);
        }

        /// <summary>Advance until the next yield; call once per FixedUpdate.</summary>
        public void Tick(double now)
        {
            if (!Running) return;
            if (waiting != null && now < waiting.Ut) return;
            waiting = null;
            try
            {
                while (stack.Count > 0)
                {
                    IEnumerator top = stack.Peek();
                    if (!top.MoveNext())
                    {
                        stack.Pop();
                        continue;
                    }
                    object current = top.Current;
                    IEnumerator nested = current as IEnumerator;
                    if (nested != null) { stack.Push(nested); continue; }
                    waiting = current as WaitUt;
                    return;
                }
            }
            catch (Exception error)
            {
                Failure = error;
                // The throwing iterator already ran its finally blocks.
                stack.Pop();
                Abort();
            }
        }

        public void Abort()
        {
            while (stack.Count > 0)
            {
                IEnumerator top = stack.Pop();
                IDisposable disposable = top as IDisposable;
                try { if (disposable != null) disposable.Dispose(); }
                catch (Exception error) { if (Failure == null) Failure = error; }
            }
            waiting = null;
        }
    }

    /// <summary>Line-delimited JSON mission record.</summary>
    public sealed class MissionLog : IDisposable
    {
        private readonly StreamWriter writer;
        public readonly string Path;

        public MissionLog(string directory)
        {
            Directory.CreateDirectory(directory);
            Path = System.IO.Path.Combine(directory, "mission-" + DateTime.UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'",
                CultureInfo.InvariantCulture) + ".jsonl");
            writer = new StreamWriter(Path, false, new UTF8Encoding(false)) { AutoFlush = true };
        }

        public void Write(string name, params object[] pairs)
        {
            StringBuilder line = new StringBuilder();
            line.Append("{\"event\":").Append(Quote(name));
            line.Append(",\"wall_time\":").Append(Value((DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds));
            for (int i = 0; i + 1 < pairs.Length; i += 2)
                line.Append(',').Append(Quote((string)pairs[i])).Append(':').Append(Value(pairs[i + 1]));
            line.Append('}');
            writer.WriteLine(line.ToString());
        }

        private static string Quote(string text)
        {
            StringBuilder s = new StringBuilder("\"");
            foreach (char c in text)
            {
                if (c == '"' || c == '\\') s.Append('\\').Append(c);
                else if (c < ' ') s.AppendFormat("\\u{0:x4}", (int)c);
                else s.Append(c);
            }
            return s.Append('"').ToString();
        }

        private static string Value(object value)
        {
            if (value == null) return "null";
            if (value is string) return Quote((string)value);
            if (value is bool) return (bool)value ? "true" : "false";
            if (value is int) return ((int)value).ToString(CultureInfo.InvariantCulture);
            if (value is double || value is float)
            {
                double d = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                return double.IsNaN(d) || double.IsInfinity(d) ? "null" : d.ToString("R", CultureInfo.InvariantCulture);
            }
            if (value is V3) { V3 v = (V3)value; return "[" + Value(v.X) + "," + Value(v.Y) + "," + Value(v.Z) + "]"; }
            IEnumerable items = value as IEnumerable;
            if (items != null)
            {
                List<string> parts = new List<string>();
                foreach (object item in items) parts.Add(Value(item));
                return "[" + string.Join(",", parts.ToArray()) + "]";
            }
            return Quote(value.ToString());
        }

        public void Dispose() { writer.Dispose(); }
    }
}
