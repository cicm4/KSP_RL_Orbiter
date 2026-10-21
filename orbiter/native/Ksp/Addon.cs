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
        private static readonly string[] Destinations = { "Mun", "Minmus" };
        private static Policy ascentPolicy, capturePolicy;
        private static string loadError;

        private ApplicationLauncherButton button;
        private Texture2D icon;
        private Rect window = new Rect(0, 100, 360, 300);
        private bool visible;
        private int destination;
        private string altitudeKm = "50";
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
            if (ascentPolicy != null || loadError != null) return;
            try
            {
                string policies = Path.Combine(Folder, "Policies");
                ascentPolicy = Policy.Load(Path.Combine(policies, "ascent.policy"));
                capturePolicy = Policy.Load(Path.Combine(policies, "capture.policy"));
                new AscentContract(ascentPolicy);
                Debug.Log("[OrbiterAutopilot] Loaded policies " + ascentPolicy.Sha256 + " / " + capturePolicy.Sha256);
            }
            catch (Exception error)
            {
                ascentPolicy = capturePolicy = null;
                loadError = "Orbiter policies are missing or damaged: " + error.Message;
                Debug.LogError("[OrbiterAutopilot] " + loadError);
            }
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
            CelestialBody body = FlightGlobals.GetBodyByName(target);
            if (body == null) return "Unknown destination " + target;
            if (!(altitudeMeters >= 40000 && altitudeMeters <= 60000)) return "Choose an altitude from 40 to 60 km";
            string problem = new Mission(ascentPolicy, capturePolicy, body, altitudeMeters, null)
                .CheckPreflight(FlightGlobals.ActiveVessel);
            if (problem != null) return problem;
            log = new MissionLog(Path.Combine(Folder, "Logs"));
            mission = new Mission(ascentPolicy, capturePolicy, body, altitudeMeters, log);
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
            CelestialBody body = FlightGlobals.GetBodyByName(Destinations[destination]);
            string problem = body == null ? "Unknown destination"
                : new Mission(ascentPolicy, capturePolicy, body, 50000, null).CheckPreflight(FlightGlobals.ActiveVessel);
            readiness = problem ?? "Ready on the launchpad";
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
            GUILayout.Label("Learned launch, transfer and capture · OrbiterRL Transfer rocket", wrapped);
            GUILayout.Label(vessel == null ? "No active vessel" : vessel.vesselName + " · " + vessel.mainBody.bodyName, wrapped);
            bool running = runner.Running;
            GUILayout.Space(6);
            GUI.enabled = !running;
            destination = GUILayout.Toolbar(destination, Destinations);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Orbit altitude", GUILayout.Width(105));
            altitudeKm = GUILayout.TextField(altitudeKm, 6, GUILayout.Width(75));
            GUILayout.Label("km  (40–60)");
            GUILayout.EndHorizontal();
            double altitude;
            bool valid = double.TryParse(altitudeKm, NumberStyles.Float, CultureInfo.InvariantCulture, out altitude)
                         && altitude >= 40 && altitude <= 60;
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
            if (!valid) GUILayout.Label("Choose an altitude from 40 to 60 km.", wrapped);
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
            }
            GUILayout.Space(5);
            GUILayout.Label("Run launches the active rocket from the pad and verifies a full orbit after capture. Stop releases the controls.", wrapped);
            if (GUILayout.Button("Close")) { HideWindow(); if (button != null) button.SetFalse(false); }
            GUI.DragWindow(new Rect(0, 0, window.width, 25));
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
    /// mission and writes Logs/autotest-result.txt. Without the flag this does nothing.
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
                string[] rest = new string[Math.Max(0, Math.Min(3, args.Length - index - 1))];
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
            string craft = Path.Combine(KSPUtil.ApplicationRootPath, "Ships/VAB/" + CraftInstaller.CraftName + ".craft");
            ConfigNode node = ConfigNode.Load(craft);
            if (node == null) { Result("error: generated craft missing: " + craft); yield break; }
            VesselCrewManifest crew = game.CrewRoster.DefaultCrewForVessel(node, VesselCrewManifest.FromConfigNode(node), true, true);
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
