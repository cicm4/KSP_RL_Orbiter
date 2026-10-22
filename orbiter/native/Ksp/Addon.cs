// KSP 1.12 toolbar UI for the self-contained Orbiter autopilot. Everything
// runs inside KSP: no Python, no kRPC, no external process or network port.
using System;
using System.Collections;
using System.Globalization;
using System.IO;
using KSP.UI.Screens;
using OrbiterRL.Core;
using UnityEngine;

namespace OrbiterRL
{
    [KSPAddon(KSPAddon.Startup.Flight, false)]
    public sealed class OrbiterAutopilot : MonoBehaviour
    {
        private const string InputLock = "OrbiterAutopilotWindow";
        private static readonly string[] Destinations = { "Kerbin orbit", "Mun", "Minmus" };
        private static MissionPolicies policies;
        private static string loadError;

        private ApplicationLauncherButton button;
        private Texture2D icon;
        private Rect window = new Rect(0, 100, 360, 300);
        private bool visible;
        private int destination = 1, shownDestination = -1;
        private string altitudeKm = "50";
        private string craftSummary = "";
        private GUIStyle wrapped;
        private readonly Runner runner = new Runner();
        private Mission mission;
        private MissionLog log;
        private string outcome, readiness = "Checking launchpad";
        private float nextCheck;

        public static string Folder
        {
            get { return Path.GetDirectoryName(Path.GetDirectoryName(typeof(OrbiterAutopilot).Assembly.Location)); }
        }

        private static void LoadPolicies()
        {
            if (policies != null || loadError != null) return;
            try
            {
                string folder = Path.Combine(Folder, "Policies");
                MissionPolicies loaded = new MissionPolicies
                {
                    Ascent = Policy.Load(Path.Combine(folder, "ascent.policy")),
                    Capture = Policy.Load(Path.Combine(folder, "capture.policy")),
                    GeneralAscent = SequencePolicy.Load(Path.Combine(folder, "generalized_ascent.policy")),
                    GeneralCapture = SequencePolicy.Load(Path.Combine(folder, "generalized_capture.policy")),
                };
                new AscentContract(loaded.Ascent);
                new GeneralizedAscentContract(loaded.GeneralAscent);
                policies = loaded;
                Debug.Log("[OrbiterAutopilot] Loaded policies " + loaded.Ascent.Sha256 + " / " + loaded.Capture.Sha256
                          + " / " + loaded.GeneralAscent.Sha256 + " / " + loaded.GeneralCapture.Sha256);
            }
            catch (Exception error)
            {
                loadError = "Orbiter policies are missing or damaged: " + error.Message;
                Debug.LogError("[OrbiterAutopilot] " + loadError);
            }
        }

        /// <summary>Altitude range (km) for a destination; moons depend on the policy that will fly.</summary>
        private static void Range(string destination, bool trainedCraft, out double low, out double high)
        {
            if (destination == "Kerbin orbit")
            {
                low = policies.GeneralAscent.Constant("target_min") / 1000;
                high = policies.GeneralAscent.Constant("target_max") / 1000;
                return;
            }
            if (trainedCraft) { low = 40; high = 60; return; }
            CelestialBody body = FlightGlobals.GetBodyByName(destination);
            GeneralizedCaptureConfig.AltitudeBand(policies.GeneralCapture, body.Radius,
                body.atmosphere ? body.atmosphereDepth : 0.0, out low, out high);
            low = Math.Ceiling(low / 1000);
            high = Math.Floor(high / 1000);
        }

        private static MissionPlan Plan(string destination, double altitudeMeters)
        {
            MissionPlan plan = new MissionPlan();
            if (destination == "Kerbin orbit" || destination == "Orbit" || destination == "Kerbin")
                plan.OrbitAltitude = altitudeMeters;
            else
            {
                plan.Target = FlightGlobals.GetBodyByName(destination);
                plan.TargetAltitude = altitudeMeters;
            }
            return plan;
        }

        private void Awake()
        {
            LoadPolicies();
            window.x = Math.Max(10, Screen.width - 405);
            GameEvents.onGUIApplicationLauncherReady.Add(AddButton);
            GameEvents.onGUIApplicationLauncherDestroyed.Add(RemoveButton);
            GameEvents.onGameSceneLoadRequested.Add(SceneRequested);
            AddButton();
        }

        private void Start()
        {
            if (Autotest.Requested) StartCoroutine(Autotest.RunInFlight(this));
        }

        private void AddButton()
        {
            if (button != null || !ApplicationLauncher.Ready || ApplicationLauncher.Instance == null) return;
            if (icon == null) icon = CreateIcon();
            button = ApplicationLauncher.Instance.AddModApplication(
                () => visible = true, HideWindow, null, null, null, null,
                ApplicationLauncher.AppScenes.FLIGHT | ApplicationLauncher.AppScenes.MAPVIEW, icon);
        }

        private void RemoveButton()
        {
            if (button != null && ApplicationLauncher.Instance != null)
                ApplicationLauncher.Instance.RemoveModApplication(button);
            button = null;
        }

        private void HideWindow()
        {
            visible = false;
            InputLockManager.RemoveControlLock(InputLock);
        }

        private void SceneRequested(GameScenes scene) { StopMission("scene change"); }

        public bool Running { get { return runner.Running; } }
        public Mission Current { get { return mission; } }
        public string Outcome { get { return outcome; } }

        public string StartMission(string target, double altitudeMeters)
        {
            if (runner.Running) return "A mission is already running";
            if (loadError != null) return loadError;
            MissionPlan plan = Plan(target, altitudeMeters);
            if (target != "Kerbin orbit" && target != "Orbit" && target != "Kerbin" && plan.Target == null)
                return "Unknown destination " + target;
            Mission check = new Mission(policies, plan, null);
            string problem = check.CheckPreflight(FlightGlobals.ActiveVessel);
            if (problem != null) return problem;
            double low, high;
            Range(plan.Target == null ? "Kerbin orbit" : plan.Target.bodyName, check.Validated, out low, out high);
            if (!(altitudeMeters >= low * 1000 - 1 && altitudeMeters <= high * 1000 + 1))
                return string.Format(CultureInfo.InvariantCulture, "Choose an altitude from {0:0} to {1:0} km", low, high);
            log = new MissionLog(Path.Combine(Folder, "Logs"));
            mission = new Mission(policies, plan, log);
            outcome = null;
            runner.Start(mission.Run());
            Debug.Log("[OrbiterAutopilot] Mission started: " + target + " " + altitudeMeters + " m; log " + log.Path);
            return null;
        }

        public void StopMission(string reason)
        {
            if (!runner.Running) return;
            runner.Abort();
            Finish("Stopped (" + reason + ") · controls released");
        }

        private void Finish(string text)
        {
            outcome = text;
            if (log != null)
            {
                log.Write("mission_end", "success", mission != null && mission.Succeeded, "outcome", text,
                          "phase", mission != null ? mission.Phase : null);
                log.Dispose();
                log = null;
            }
            Debug.Log("[OrbiterAutopilot] " + text);
        }

        private void FixedUpdate()
        {
            if (!runner.Running) return;
            runner.Tick(Planetarium.GetUniversalTime());
            if (runner.Running) return;
            if (runner.Failure != null)
            {
                MissionFailure failure = runner.Failure as MissionFailure;
                Debug.LogError("[OrbiterAutopilot] " + runner.Failure);
                Finish("Mission stopped: " + (failure != null ? failure.Message : runner.Failure.GetType().Name + ": " + runner.Failure.Message));
            }
            else Finish(mission.Succeeded ? mission.Detail : "Mission ended");
        }

        private void Update()
        {
            if (runner.Running || !visible || Time.realtimeSinceStartup < nextCheck) return;
            nextCheck = Time.realtimeSinceStartup + 1f;
            if (loadError != null) { readiness = loadError; return; }
            string name = Destinations[destination];
            double low = 0, high = 0;
            Mission check = new Mission(policies, Plan(name, name == "Kerbin orbit" ? 100000 : 0), null);
            string problem = check.CheckPreflight(FlightGlobals.ActiveVessel);
            bool trained = problem == null && check.Validated;
            Range(name, trained, out low, out high);
            if (shownDestination != destination)
            {
                // Suggest the middle of the range whenever the destination changes.
                altitudeKm = Math.Round((low + high) / 2).ToString(CultureInfo.InvariantCulture);
                shownDestination = destination;
            }
            if (problem != null && problem.StartsWith("Choose a")) problem = null; // altitude is checked on Run
            readiness = problem ?? (check.ResumeFrom == null ? "Ready on the launchpad"
                : "Ready to continue the mission from " + (check.ResumeFrom == "transfer" ? "this orbit"
                    : check.ResumeFrom == "coast" ? "the transfer coast" : "the approach"));
            craftSummary = Summary(check, low, high);
        }

        private void OnGUI()
        {
            if (!visible) return;
            GUISkin previous = GUI.skin;
            GUI.skin = HighLogic.Skin;
            if (wrapped == null) wrapped = new GUIStyle(GUI.skin.label) { wordWrap = true };
            window.x = Mathf.Clamp(window.x, 0, Math.Max(0, Screen.width - window.width));
            window.y = Mathf.Clamp(window.y, 0, Math.Max(0, Screen.height - window.height));
            window = GUILayout.Window(GetInstanceID(), window, DrawWindow, "Orbiter Autopilot", GUILayout.Width(360));
            if (window.Contains(Event.current.mousePosition))
                InputLockManager.SetControlLock(ControlTypes.ALLBUTCAMERAS, InputLock);
            else InputLockManager.RemoveControlLock(InputLock);
            GUI.skin = previous;
        }

        private void DrawWindow(int id)
        {
            Vessel vessel = FlightGlobals.ActiveVessel;
            GUILayout.Label("Learned launch, transfer and capture", wrapped);
            GUILayout.Label(vessel == null ? "No active vessel" : vessel.vesselName + " · " + vessel.mainBody.bodyName, wrapped);
            bool running = runner.Running;
            GUILayout.Space(6);
            GUI.enabled = !running;
            destination = GUILayout.Toolbar(destination, Destinations);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Orbit altitude", GUILayout.Width(105));
            altitudeKm = GUILayout.TextField(altitudeKm, 6, GUILayout.Width(75));
            GUILayout.Label("km");
            GUILayout.EndHorizontal();
            double altitude;
            bool valid = double.TryParse(altitudeKm, NumberStyles.Float, CultureInfo.InvariantCulture, out altitude)
                         && altitude > 0;
            GUI.enabled = true;
            GUILayout.Space(8);
            GUILayout.BeginHorizontal();
            GUI.enabled = !running && valid && loadError == null;
            if (GUILayout.Button("Run", GUILayout.Height(30)))
            {
                string problem = StartMission(Destinations[destination], altitude * 1000);
                if (problem != null) outcome = problem;
            }
            GUI.enabled = running;
            if (GUILayout.Button("Stop", GUILayout.Height(30))) StopMission("Stop button");
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            if (!valid) GUILayout.Label("Enter an orbit altitude in km.", wrapped);
            GUILayout.Space(8);
            if (running && mission != null)
            {
                GUILayout.Label(HumanPhase(mission.Phase), wrapped);
                GUILayout.Label(mission.Detail ?? "", wrapped);
            }
            else
            {
                if (outcome != null) GUILayout.Label(outcome, wrapped);
                GUILayout.Label(readiness, wrapped);
                if (craftSummary.Length > 0) GUILayout.Label(craftSummary, wrapped);
            }
            GUILayout.Space(5);
            GUILayout.Label("Run launches the active rocket from the pad and verifies a full orbit after capture. Stop releases the controls.", wrapped);
            if (GUILayout.Button("Close")) { HideWindow(); if (button != null) button.SetFalse(false); }
            GUI.DragWindow(new Rect(0, 0, window.width, 25));
        }

        private static string Summary(Mission check, double low, double high)
        {
            string range = string.Format(CultureInfo.InvariantCulture, "Altitude range {0:0}–{1:0} km. ", low, high);
            if (check.Validated) return range + "Trained OrbiterRL Transfer rocket: flown by the validated policies.";
            CraftReport report = check.Report;
            if (report == null || report.Craft == null) return range;
            CraftSpec c = report.Craft;
            string text = range + string.Format(CultureInfo.InvariantCulture,
                "Any-rocket policies (simulation-tested): {0} stage{1}, {2:0.0} t, {3:0} m/s, launch TWR {4:0.00}.",
                c.Stages.Length, c.Stages.Length == 1 ? "" : "s", c.InitialMass / 1000, c.IdealDeltaV, report.PadTwr);
            if (report.Warnings.Count > 0) text += " Note: " + report.Warnings[0] + ".";
            return text;
        }

        private static string HumanPhase(string phase)
        {
            switch (phase)
            {
                case "preflight": return "Checking the craft";
                case "ascent": return "Launch · climbing to parking orbit";
                case "transfer_planning": return "Parking orbit · checking an encounter";
                case "maneuver": return "Transfer · executing a maneuver";
                case "transfer_coast": return "Transfer · coasting to the destination";
                case "capture_coast": return "Capture · approaching the burn";
                case "capture": return "Capture · establishing the target orbit";
                case "orbit_verification": return "Verifying a full unpowered orbit";
                case "succeeded": return "Orbit verified";
                default: return "Preparing the mission";
            }
        }

        private void OnDestroy()
        {
            StopMission("scene unloaded");
            GameEvents.onGUIApplicationLauncherReady.Remove(AddButton);
            GameEvents.onGUIApplicationLauncherDestroyed.Remove(RemoveButton);
            GameEvents.onGameSceneLoadRequested.Remove(SceneRequested);
            HideWindow();
            RemoveButton();
            if (icon != null) Destroy(icon);
        }

        private static Texture2D CreateIcon()
        {
            Texture2D texture = new Texture2D(38, 38, TextureFormat.RGBA32, false);
            Color32[] pixels = new Color32[38 * 38];
            for (int y = 0; y < 38; y++) for (int x = 0; x < 38; x++)
            {
                float dx = x - 18.5f, dy = y - 18.5f;
                float radius = Mathf.Sqrt(dx * dx + dy * dy);
                bool orbit = radius >= 11 && radius <= 13;
                bool arrow = y >= 14 && y <= 28 && Math.Abs(x - 19) <= (28 - y) / 3;
                bool center = x >= 17 && x <= 21 && y >= 9 && y <= 23;
                pixels[y * 38 + x] = orbit || arrow || center
                    ? new Color32(225, 242, 235, 255) : new Color32(0, 0, 0, 0);
            }
            texture.SetPixels32(pixels); texture.Apply();
            return texture;
        }
    }

    /// <summary>
    /// Builds Ships/VAB/OrbiterRL Transfer.craft once from the player's own
    /// stock Orbiter One and Science Jr templates (nothing stock-derived is
    /// distributed). Existing files are never overwritten.
    /// </summary>
    [KSPAddon(KSPAddon.Startup.MainMenu, true)]
    public sealed class CraftInstaller : MonoBehaviour
    {
        public const string CraftName = "OrbiterRL Transfer";

        private void Start()
        {
            string ships = Path.Combine(KSPUtil.ApplicationRootPath, "Ships/VAB");
            string output = Path.Combine(ships, CraftName + ".craft");
            try
            {
                if (File.Exists(output)) return;
                string one = Path.Combine(ships, "Orbiter One.craft"), science = Path.Combine(ships, "Science Jr.craft");
                if (!File.Exists(one) || !File.Exists(science))
                {
                    Debug.LogWarning("[OrbiterAutopilot] Stock Orbiter One / Science Jr craft not found; cannot build " + CraftName);
                    return;
                }
                string craft = CraftBuilder.MakeTransfer(File.ReadAllText(one), File.ReadAllText(science));
                using (FileStream stream = new FileStream(output, FileMode.CreateNew))
                using (StreamWriter writer = new StreamWriter(stream, new System.Text.UTF8Encoding(false)))
                    writer.Write(craft);
                Debug.Log("[OrbiterAutopilot] Built " + output + " from stock templates");
            }
            catch (Exception error)
            {
                Debug.LogError("[OrbiterAutopilot] Could not build " + CraftName + ": " + error.Message);
            }
        }
    }

    /// <summary>
    /// Opt-in unattended validation: KSP -orbiter-autotest Mun 50 [save] loads
    /// saves/OrbiterRL/&lt;save&gt;.sfs (default OrbiterRL-pad-transfer), flies one
    /// mission and writes Logs/autotest-result.txt. "fresh [craft]" instead launches
    /// a craft (default the generated OrbiterRL Transfer) from the Space Center.
    /// Destination "Orbit" stops in Kerbin orbit. Without the flag this does nothing.
    /// </summary>
    [KSPAddon(KSPAddon.Startup.MainMenu, true)]
    public sealed class Autotest : MonoBehaviour
    {
        private static string[] Arguments
        {
            get
            {
                string[] args = Environment.GetCommandLineArgs();
                int index = Array.IndexOf(args, "-orbiter-autotest");
                if (index < 0) return null;
                string[] rest = new string[Math.Max(0, Math.Min(4, args.Length - index - 1))];
                Array.Copy(args, index + 1, rest, 0, rest.Length);
                return rest;
            }
        }

        public static bool Requested { get { return Arguments != null && !started; } }
        private static bool loaded, started;

        private IEnumerator Start()
        {
            string[] args = Arguments;
            if (args == null || loaded) yield break;
            loaded = true;
            for (int frame = 0; frame < 120; frame++) yield return null;
            // "fresh" launches the generated stock-VAB craft with default crew,
            // exactly as a new player would; otherwise focus a saved pad vessel.
            bool fresh = args.Length > 2 && args[2] == "fresh";
            string save = fresh ? "persistent" : args.Length > 2 ? args[2] : "OrbiterRL-pad-transfer";
            Game game = GamePersistence.LoadGame(save, "OrbiterRL", true, false);
            if (game == null) { Result("error: cannot load saves/OrbiterRL/" + save + ".sfs"); yield break; }
            HighLogic.SaveFolder = "OrbiterRL";
            HighLogic.CurrentGame = game;
            if (!fresh)
            {
                FlightDriver.StartAndFocusVessel(game, game.flightState.activeVesselIdx);
                yield break;
            }
            // A new launch needs the initialized Space Center scene, as in the stock flow.
            DontDestroyOnLoad(gameObject);
            game.startScene = GameScenes.SPACECENTER;
            game.Start();
            for (int i = 0; i < 1200 && HighLogic.LoadedScene != GameScenes.SPACECENTER; i++)
                yield return new WaitForSeconds(0.1f);
            yield return new WaitForSeconds(3f);
            game = HighLogic.CurrentGame;
            string name = args.Length > 3 ? args[3] : CraftInstaller.CraftName;
            string craft = Path.Combine(KSPUtil.ApplicationRootPath, "saves/OrbiterRL/Ships/VAB/" + name + ".craft");
            if (!File.Exists(craft)) craft = Path.Combine(KSPUtil.ApplicationRootPath, "Ships/VAB/" + name + ".craft");
            ConfigNode node = ConfigNode.Load(craft);
            if (node == null) { Result("error: generated craft missing: " + craft); yield break; }
            VesselCrewManifest crew = game.CrewRoster.DefaultCrewForVessel(node, VesselCrewManifest.FromConfigNode(node), true, true);
            // Test saves can have every kerbal assigned; hire into empty seats so the pod is crewed.
            foreach (PartCrewManifest part in crew.PartManifests)
                for (int seat = 0; part.partCrew != null && seat < part.partCrew.Length; seat++)
                    if (string.IsNullOrEmpty(part.partCrew[seat]))
                        part.AddCrewToSeat(game.CrewRoster.GetNextOrNewKerbal(ProtoCrewMember.KerbalType.Crew), seat);
            FlightDriver.StartWithNewLaunch(craft, game.flagURL, "LaunchPad", crew);
        }

        public static IEnumerator RunInFlight(OrbiterAutopilot addon)
        {
            string[] args = Arguments;
            started = true;
            string target = args.Length > 0 ? args[0] : "Mun";
            double altitude = (args.Length > 1 ? double.Parse(args[1], CultureInfo.InvariantCulture) : 50) * 1000;
            for (int i = 0; i < 600 && (!FlightGlobals.ready || FlightGlobals.ActiveVessel == null
                                        || FlightGlobals.ActiveVessel.packed); i++)
                yield return new WaitForSeconds(0.1f);
            yield return new WaitForSeconds(3f);
            string problem = addon.StartMission(target, altitude);
            if (problem != null) { Result("preflight_failed: " + problem); yield break; }
            // Save resumable checkpoints so later tests can start mid-mission.
            addon.Current.PhaseChanged += phase =>
            {
                if (phase != "capture_coast" && phase != "transfer_planning") return;
                string name = "autotest-" + target + "-" + phase;
                GamePersistence.SaveGame(HighLogic.CurrentGame.Updated(), name, HighLogic.SaveFolder, SaveMode.OVERWRITE);
                Debug.Log("[OrbiterAutopilot] autotest saved " + name);
            };
            Result("running");
            while (addon.Running) yield return new WaitForSeconds(1f);
            Result((addon.Current != null && addon.Current.Succeeded ? "succeeded: " : "failed: ") + addon.Outcome);
        }

        private static void Result(string text)
        {
            string folder = Path.Combine(OrbiterAutopilot.Folder, "Logs");
            Directory.CreateDirectory(folder);
            File.WriteAllText(Path.Combine(folder, "autotest-result.txt"),
                              DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture) + " " + text + "\n");
            Debug.Log("[OrbiterAutopilot] autotest " + text);
        }
    }
}
