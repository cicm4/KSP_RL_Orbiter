// C# port of orbiter/make_craft.py. Builds the "OrbiterRL Transfer" rocket
// from the player's own stock Orbiter One and Science Jr craft files, so no
// stock-derived craft is distributed. Output is byte-identical to Python's.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace OrbiterRL.Core
{
    public static class CraftBuilder
    {
        private const RegexOptions Multi = RegexOptions.Multiline;

        /// <summary>Full pipeline; inputs are raw file text (newlines normalized like Python text mode).</summary>
        public static string MakeTransfer(string orbiterOne, string scienceJr)
        {
            return MakeTransferFromTerrier(MakeTerrier(Normalize(orbiterOne), Normalize(scienceJr)));
        }

        public static string Normalize(string text) { return text.Replace("\r\n", "\n").Replace('\r', '\n'); }

        /// <summary>Python's format(value, '.10g').</summary>
        public static string G10(double value)
        {
            if (value == 0) return 1 / value < 0 ? "-0" : "0";
            if (double.IsNaN(value)) return "nan";
            if (double.IsInfinity(value)) return value > 0 ? "inf" : "-inf";
            string e = value.ToString("E9", CultureInfo.InvariantCulture); // d.ddddddddde+XXX
            int exponent = int.Parse(e.Substring(e.IndexOf('E') + 1), CultureInfo.InvariantCulture);
            if (exponent < -4 || exponent >= 10)
            {
                string mantissa = e.Substring(0, e.IndexOf('E'));
                if (mantissa.Contains(".")) mantissa = mantissa.TrimEnd('0').TrimEnd('.');
                return mantissa + "e" + (exponent < 0 ? "-" : "+") + Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture);
            }
            string fixedText = value.ToString("F" + Math.Max(0, 9 - exponent), CultureInfo.InvariantCulture);
            if (fixedText.Contains(".")) fixedText = fixedText.TrimEnd('0').TrimEnd('.');
            return fixedText;
        }

        /// <summary>float() semantics, including "-0" (Mono's double.Parse drops the sign of zero).</summary>
        public static double Parse(string text)
        {
            text = text.Trim();
            double value = double.Parse(text, CultureInfo.InvariantCulture);
            return value == 0 && text.StartsWith("-") ? -0.0 : value;
        }

        private static void Split(string source, out string prefix, out List<string> blocks)
        {
            MatchCollection starts = Regex.Matches(source, @"^PART\s*\n\{", Multi);
            if (starts.Count == 0) throw new ArgumentException("No PART blocks found in craft");
            prefix = source.Substring(0, starts[0].Index);
            blocks = new List<string>();
            for (int i = 0; i < starts.Count; i++)
            {
                int end = i + 1 < starts.Count ? starts[i + 1].Index : source.Length;
                blocks.Add(source.Substring(starts[i].Index, end - starts[i].Index));
            }
        }

        private static string PartId(string block) { return Regex.Match(block, @"^\tpart = (\S+)", Multi).Groups[1].Value; }

        private static string Before(string id) { int i = id.LastIndexOf('_'); return i < 0 ? id : id.Substring(0, i); }
        private static string After(string id) { int i = id.LastIndexOf('_'); return i < 0 ? id : id.Substring(i + 1); }

        private static double[] Vector(string block, string key)
        {
            Match m = Regex.Match(block, @"^\t" + key + @" = ([^\n]+)", Multi);
            if (!m.Success) throw new ArgumentException("Missing " + key + " in craft part");
            return m.Groups[1].Value.Split(',').Select(Parse).ToArray();
        }

        private static string SetVector(string block, string key, double[] value)
        {
            Regex pattern = new Regex(@"^\t" + key + @" = [^\n]+", Multi);
            if (pattern.Matches(block).Count != 1) throw new ArgumentException("Expected one " + key + " in craft part");
            string text = "\t" + key + " = " + string.Join(",", value.Select(G10).ToArray());
            return pattern.Replace(block, _ => text);
        }

        private static double[] Attachment(string block, string node)
        {
            Match m = Regex.Match(block, @"^\tattN = " + node + @",([^\n]+)", Multi);
            if (!m.Success) throw new ArgumentException("Missing " + node + " attachment in craft part");
            return After(m.Groups[1].Value).Split('|').Select(Parse).ToArray();
        }

        private static string Peer(string block, string node)
        {
            Match m = Regex.Match(block, @"^\tattN = " + node + @",([^\n]+)", Multi);
            if (!m.Success) throw new ArgumentException("Missing " + node + " attachment in craft part");
            return Before(m.Groups[1].Value);
        }

        private static string SetPeer(string block, string node, string peer)
        {
            Regex pattern = new Regex(@"^(\tattN = " + node + @",)[^\n]+_([^_\n]+)$", Multi);
            if (pattern.Matches(block).Count != 1) throw new ArgumentException("Expected one " + node + " attachment in craft part");
            return pattern.Replace(block, m => m.Groups[1].Value + peer + "_" + m.Groups[2].Value);
        }

        private static List<string> Links(string block)
        {
            return Regex.Matches(block, @"^\tlink = (\S+)", Multi).Cast<Match>().Select(m => m.Groups[1].Value).ToList();
        }

        private static string Header(string prefix, string key, string value)
        {
            return Regex.Replace(prefix, "^" + key + " = .*$", _ => key + " = " + value, Multi);
        }

        private static string MakeLiquid(string source)
        {
            MatchCollection starts = Regex.Matches(source, @"^PART\s*\n\{", Multi);
            if (starts.Count != 21) throw new ArgumentException("Expected the stock 21-part Orbiter One; source layout differs");
            string prefix, unused;
            List<string> blocks;
            Split(source, out prefix, out blocks);
            prefix = Header(prefix, "ship", "OrbiterRL Liquid");
            prefix = Header(prefix, "description", "Two liquid stages for reproducible orbital-launch experiments. Derived locally from stock Orbiter One.");
            prefix = Header(prefix, "vesselType", "Ship");
            List<string> ids = blocks.Select(b => Regex.Match(b, @"^\s*part = (\S+)", Multi).Groups[1].Value).ToList();
            HashSet<string> removed = new HashSet<string>(ids.Where(p =>
                new[] { "radialDecoupler", "solidBooster.v2", "noseCone" }.Contains(p.Split('_')[0])));
            if (removed.Count != 3) throw new ArgumentException("Expected exactly one radial decoupler, solid booster, nose cone");
            StringBuilder result = new StringBuilder(prefix);
            for (int i = 0; i < blocks.Count; i++)
            {
                if (removed.Contains(ids[i])) continue;
                string block = string.Join("\n", blocks[i].Split('\n').Where(line =>
                    !(Regex.IsMatch(line, @"^\t(?:link|attN|srfN) = ") && removed.Any(x => line.Contains(x)))).ToArray());
                string name = Before(ids[i]);
                int oldStage = int.Parse(Regex.Match(block, @"^\tistg = (-?\d+)", Multi).Groups[1].Value, CultureInfo.InvariantCulture);
                int oldDstg = int.Parse(Regex.Match(block, @"^\tdstg = (-?\d+)", Multi).Groups[1].Value, CultureInfo.InvariantCulture);
                int newStage = name == "liquidEngine2" && oldStage == 5 ? 3 : oldStage == 3 ? 2 : oldStage;
                int newDstg = oldDstg == 4 ? 3 : oldDstg == 3 ? 2 : oldDstg;
                block = Regex.Replace(block, @"^\tistg = -?\d+", _ => "\tistg = " + newStage, Multi);
                block = Regex.Replace(block, @"^\tdstg = -?\d+", _ => "\tdstg = " + newDstg, Multi);
                result.Append(block);
            }
            unused = result.ToString();
            if (removed.Any(id => unused.Contains(id))) throw new ArgumentException("A reference to removed booster parts remains");
            return unused;
        }

        public static string MakeTerrier(string source, string terrierSource)
        {
            string prefix, templatePrefix;
            List<string> blocks, templates;
            Split(MakeLiquid(source), out prefix, out blocks);
            Split(terrierSource, out templatePrefix, out templates);
            List<string> terriers = templates.Where(b => Before(PartId(b)) == "liquidEngine3.v2").ToList();
            if (terriers.Count != 1) throw new ArgumentException("Terrier template must contain exactly one liquidEngine3.v2");
            string template = terriers[0];
            List<int> uppers = Enumerable.Range(0, blocks.Count)
                .Where(i => PartId(blocks[i]).StartsWith("liquidEngine2_") && blocks[i].Contains("\tistg = 2\n")).ToList();
            if (uppers.Count != 1) throw new ArgumentException("Expected exactly one stage-2 upper Swivel");
            int index = uppers[0];
            string old = blocks[index], oldId = PartId(old), newId = "liquidEngine3.v2_" + After(oldId);
            double[] oldTop = Attachment(old, "top"), oldBottom = Attachment(old, "bottom");
            double[] newTop = Attachment(template, "top"), newBottom = Attachment(template, "bottom");
            foreach (double[] v in new[] { oldTop, oldBottom, newTop, newBottom })
                if (Math.Abs(v[0]) > 1e-5 || Math.Abs(v[2]) > 1e-5)
                    throw new ArgumentException("Only coaxial vertical stack attachments are supported");
            double engineShift = oldTop[1] - newTop[1];
            double lowerShift = engineShift + newBottom[1] - oldBottom[1];
            List<string> links = Links(old);
            if (links.Count != 1) throw new ArgumentException("Expected the upper engine to have one lower decoupler");
            string lowerId = links[0];
            Dictionary<string, string> parts = new Dictionary<string, string>();
            foreach (string b in blocks) parts[PartId(b)] = b;
            HashSet<string> descendants = new HashSet<string>();
            Stack<string> pending = new Stack<string>(new[] { lowerId });
            while (pending.Count > 0)
            {
                string id = pending.Pop();
                if (!descendants.Add(id)) throw new ArgumentException("Cycle or repeated child in craft attachment tree");
                if (!parts.ContainsKey(id)) throw new ArgumentException("Missing linked craft part " + id);
                foreach (string child in Links(parts[id])) pending.Push(child);
            }
            const string marker = "\tEVENTS\n";
            if (!old.Contains(marker) || !template.Contains(marker)) throw new ArgumentException("Missing stock part state section");
            string replacement = old.Substring(0, old.IndexOf(marker, StringComparison.Ordinal)) + marker
                                 + template.Substring(template.IndexOf(marker, StringComparison.Ordinal) + marker.Length);
            replacement = replacement.Replace("\tpart = " + oldId + "\n", "\tpart = " + newId + "\n");
            double[] position = Vector(replacement, "pos");
            position[1] += engineShift;
            replacement = SetVector(replacement, "pos", position);
            double[] attachment = Vector(replacement, "attPos0");
            attachment[1] += engineShift;
            replacement = SetVector(replacement, "attPos0", attachment);
            foreach (KeyValuePair<string, double[]> node in new[] { new KeyValuePair<string, double[]>("top", newTop),
                                                                     new KeyValuePair<string, double[]>("bottom", newBottom) })
            {
                string coords = string.Join("|", node.Value.Select(G10).ToArray());
                replacement = Regex.Replace(replacement, @"^(\tattN = " + node.Key + @",[^\n]+_)[^_\n]+$",
                                            m => m.Groups[1].Value + coords, Multi);
            }
            blocks[index] = replacement;
            for (int i = 0; i < blocks.Count; i++)
            {
                string block = blocks[i], id = PartId(block);
                if (descendants.Contains(id))
                {
                    double[] p = Vector(block, "pos");
                    p[1] += lowerShift;
                    block = SetVector(block, "pos", p);
                    if (id == lowerId)
                    {
                        double[] a = Vector(block, "attPos0");
                        a[1] += lowerShift - engineShift;
                        block = SetVector(block, "attPos0", a);
                    }
                }
                blocks[i] = block.Replace(oldId, newId);
            }
            prefix = Header(prefix, "ship", "OrbiterRL Terrier");
            prefix = Header(prefix, "description", "Sequential Swivel and Terrier liquid stages for orbital policy validation. Derived locally from stock craft.");
            return prefix + string.Concat(blocks.ToArray());
        }

        public static string MakeTransferFromTerrier(string source)
        {
            string prefix;
            List<string> blocks;
            Split(source, out prefix, out blocks);
            if (!Regex.IsMatch(prefix, @"^ship = OrbiterRL Terrier$", Multi) || blocks.Count != 18)
                throw new ArgumentException("Expected the 18-part OrbiterRL Terrier source craft");
            Dictionary<string, string> parts = new Dictionary<string, string>();
            foreach (string b in blocks)
            {
                if (parts.ContainsKey(PartId(b))) throw new ArgumentException("Source craft contains duplicate part IDs");
                parts[PartId(b)] = b;
            }
            List<string> engines = parts.Keys.Where(id => id.StartsWith("liquidEngine3.v2_")).ToList();
            if (engines.Count != 1) throw new ArgumentException("Expected exactly one upper Terrier engine");
            string engineId = engines[0], engine = parts[engineId];
            string tankId = Peer(engine, "top");
            if (!parts.ContainsKey(tankId) || !tankId.StartsWith("fuelTankSmall_"))
                throw new ArgumentException("The upper Terrier must attach below an FL-T200 tank");
            string tank = parts[tankId];
            List<string> tankLinks = Links(tank);
            if (tankLinks.Count != 1 || tankLinks[0] != engineId) throw new ArgumentException("The cloned upper tank must have only the Terrier child");
            double[] top = Attachment(tank, "top"), bottom = Attachment(tank, "bottom");
            if (Math.Abs(top[0]) > 1e-5 || Math.Abs(top[2]) > 1e-5 || Math.Abs(bottom[0]) > 1e-5 || Math.Abs(bottom[2]) > 1e-5
                || top[1] <= bottom[1])
                throw new ArgumentException("Only coaxial vertical tank stacks are supported");
            Dictionary<string, string> resources = new Dictionary<string, string>();
            foreach (Match m in Regex.Matches(tank, @"\tRESOURCE\n\t\{\n\t\tname = (\S+)\n\t\tamount = ([^\n]+)"))
                resources[m.Groups[1].Value] = m.Groups[2].Value;
            if (resources.Count != 2 || !resources.ContainsKey("LiquidFuel") || resources["LiquidFuel"] != "90"
                || !resources.ContainsKey("Oxidizer") || resources["Oxidizer"] != "110")
                throw new ArgumentException("Expected a full stock FL-T200 with 90 LiquidFuel and 110 Oxidizer");
            List<string> persistentIds = Regex.Matches(source, @"^\tpersistentId = (\d+)$", Multi).Cast<Match>()
                .Select(m => m.Groups[1].Value).ToList();
            if (persistentIds.Count != blocks.Count || persistentIds.Distinct().Count() != blocks.Count)
                throw new ArgumentException("Every source part needs a unique persistentId");
            HashSet<ulong> used = new HashSet<ulong>(parts.Keys.Select(id => ulong.Parse(After(id), CultureInfo.InvariantCulture)));
            foreach (string id in persistentIds) used.Add(ulong.Parse(id, CultureInfo.InvariantCulture));
            List<ulong> numbers = new List<ulong>();
            ulong candidate = (used.Max() + 1) % 4294967296UL;
            while (numbers.Count < 4)
            {
                if (candidate != 0 && !used.Contains(candidate)) { numbers.Add(candidate); used.Add(candidate); }
                candidate = (candidate + 1) % 4294967296UL;
            }
            string[] newIds = { "fuelTankSmall_" + numbers[0], "fuelTankSmall_" + numbers[1] };
            double height = top[1] - bottom[1], shift = -2 * (top[1] - bottom[1]);
            HashSet<string> descendants = new HashSet<string>();
            Stack<string> pending = new Stack<string>(new[] { engineId });
            while (pending.Count > 0)
            {
                string id = pending.Pop();
                if (descendants.Contains(id) || !parts.ContainsKey(id)) throw new ArgumentException("Invalid lower craft attachment tree");
                descendants.Add(id);
                foreach (string child in Links(parts[id])) pending.Push(child);
            }
            List<string> clones = new List<string>();
            for (int i = 0; i < newIds.Length; i++)
            {
                string clone = tank.Replace("\tpart = " + tankId + "\n", "\tpart = " + newIds[i] + "\n");
                ulong persistent = numbers[i + 2];
                clone = Regex.Replace(clone, @"^\tpersistentId = \d+$", _ => "\tpersistentId = " + persistent, Multi);
                double[] position = Vector(tank, "pos");
                position[1] -= height * (i + 1);
                clone = SetVector(clone, "pos", position);
                clone = SetVector(clone, "attPos0", new[] { 0.0, -height, 0.0 });
                string parent = i == 0 ? tankId : newIds[i - 1];
                string child = i + 1 < newIds.Length ? newIds[i + 1] : engineId;
                clone = SetPeer(clone, "top", parent);
                clone = SetPeer(clone, "bottom", child);
                clone = clone.Replace("\tlink = " + engineId + "\n", "\tlink = " + child + "\n");
                clones.Add(clone);
            }
            StringBuilder rebuilt = new StringBuilder();
            foreach (string original in blocks)
            {
                string block = original, id = PartId(block);
                if (id == tankId)
                {
                    block = block.Replace("\tlink = " + engineId + "\n", "\tlink = " + newIds[0] + "\n");
                    block = SetPeer(block, "bottom", newIds[0]);
                }
                if (descendants.Contains(id))
                {
                    double[] position = Vector(block, "pos");
                    position[1] += shift;
                    block = SetVector(block, "pos", position);
                    if (id == engineId)
                    {
                        block = SetPeer(block, "top", newIds[newIds.Length - 1]);
                        double[] parentPosition = Vector(clones[clones.Count - 1], "pos");
                        block = SetVector(block, "attPos0", position.Zip(parentPosition, (a, b) => a - b).ToArray());
                    }
                }
                rebuilt.Append(block);
                if (id == tankId) foreach (string clone in clones) rebuilt.Append(clone);
            }
            prefix = Header(prefix, "ship", "OrbiterRL Transfer");
            prefix = Header(prefix, "description", "Crewed Swivel/Terrier transfer craft with four upper FL-T200 tanks; sequential stages 3 and 2.");
            Match size = Regex.Match(prefix, @"^size = ([^\n]+)$", Multi);
            if (size.Success)
            {
                double[] dimensions = size.Groups[1].Value.Split(',').Select(Parse).ToArray();
                dimensions[1] -= shift;
                prefix = prefix.Substring(0, size.Groups[1].Index) + string.Join(",", dimensions.Select(G10).ToArray())
                         + prefix.Substring(size.Groups[1].Index + size.Groups[1].Length);
            }
            return prefix + rebuilt;
        }
    }
}
