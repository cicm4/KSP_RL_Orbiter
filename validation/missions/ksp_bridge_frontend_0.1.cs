// KSP 1.12 native UI for the external Orbiter controller.
// Network/process work runs on one background thread. No flight controls are
// written here; the existing Python worker owns guidance, staging and cleanup.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using KSP.UI.Screens;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace OrbiterRL
{
    [KSPAddon(KSPAddon.Startup.MainMenu, true)]
    public sealed class OrbiterAutopilot : MonoBehaviour
    {
        private const string InputLock = "OrbiterAutopilotWindow";
        private static OrbiterAutopilot instance;
        private ApplicationLauncherButton button;
        private Texture2D icon;
        private Rect window = new Rect(0, 100, 360, 410);
        private bool visible;
        private int destination;
        private string altitudeKm = "50";
        private string selectedJob;
        private string configurationError;
        private string activeVessel = "No active vessel";
        private string activeBody = "Unknown";
        private Bridge bridge;
        private GUIStyle wrapped;

        private void Awake()
        {
            if (instance != null) { Destroy(this); return; }
            instance = this;
            DontDestroyOnLoad(gameObject);
            window.x = Math.Max(10, Screen.width - 405);
            GameEvents.onGUIApplicationLauncherReady.Add(AddButton);
            GameEvents.onGUIApplicationLauncherDestroyed.Add(RemoveButton);
            GameEvents.onGameSceneLoadRequested.Add(SceneRequested);
            try
            {
                string folder = Path.Combine(KSPUtil.ApplicationRootPath, "GameData/OrbiterAutopilot");
                bridge = new Bridge(Settings.Load(Path.Combine(folder, "settings.cfg")),
                                    Path.Combine(folder, "backend.log"));
                bridge.Start();
                Debug.Log("[OrbiterAutopilot] Native toolbar service initialized");
            }
            catch (Exception error)
            {
                configurationError = "Orbiter setup is incomplete: " + error.Message;
                Debug.LogError("[OrbiterAutopilot] " + configurationError);
            }
            AddButton();
        }

        private void AddButton()
        {
            if (button != null || !ApplicationLauncher.Ready || ApplicationLauncher.Instance == null) return;
            if (icon == null) icon = CreateIcon();
            button = ApplicationLauncher.Instance.AddModApplication(
                ShowWindow, HideWindow, null, null, null, null,
                ApplicationLauncher.AppScenes.FLIGHT | ApplicationLauncher.AppScenes.MAPVIEW, icon);
            Debug.Log("[OrbiterAutopilot] Flight toolbar button registered");
        }

        private void RemoveButton()
        {
            if (button != null && ApplicationLauncher.Instance != null)
                ApplicationLauncher.Instance.RemoveModApplication(button);
            button = null;
        }

        private void ShowWindow()
        {
            visible = true;
            if (bridge != null) bridge.RequestConnection();
        }

        private void HideWindow()
        {
            visible = false;
            InputLockManager.RemoveControlLock(InputLock);
        }

        private void SceneRequested(GameScenes scene)
        {
            // Reload/revert into FLIGHT also changes the controlling vessel.
            if (bridge != null)
            {
                bridge.FlightScene = false;
                bridge.RequestStop("scene change");
            }
            if (scene != GameScenes.FLIGHT) HideWindow();
        }

        private void Update()
        {
            if (instance != this) return;
            bool flight = HighLogic.LoadedSceneIsFlight;
            if (flight)
            {
                AddButton();
                Vessel vessel = FlightGlobals.ActiveVessel;
                activeVessel = vessel == null ? "No active vessel" : vessel.vesselName;
                activeBody = vessel == null || vessel.mainBody == null ? "Unknown" : vessel.mainBody.bodyName;
            }
            if (!flight) InputLockManager.RemoveControlLock(InputLock);
            if (bridge == null) return;
            bridge.FlightScene = flight;
            if (flight) bridge.RequestConnection();
            foreach (string message in bridge.TakeMessages())
                Debug.Log("[OrbiterAutopilot] " + message);
            View view = bridge.Read();
            if (view.Status != null && view.Status.active && view.Status.mission != null &&
                view.Status.mission.id != selectedJob)
            {
                selectedJob = view.Status.mission.id;
                destination = view.Status.mission.target == "Minmus" ? 1 : 0;
                altitudeKm = (view.Status.mission.target_altitude_m / 1000).ToString("0.#", CultureInfo.InvariantCulture);
            }
        }

        private void OnGUI()
        {
            if (!visible || !HighLogic.LoadedSceneIsFlight) return;
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
            GUILayout.Label("Mun/Minmus · matched transfer rocket", wrapped);
            GUILayout.Label(Short(activeVessel + " · " + activeBody, 100), wrapped);
            if (!String.IsNullOrEmpty(configurationError))
            {
                GUILayout.Label(configurationError, wrapped);
                if (GUILayout.Button("Close")) CloseWindow();
                GUI.DragWindow(new Rect(0, 0, window.width, 25));
                return;
            }
            View view = bridge.Read();
            Status status = view.Status;
            bool active = status != null && status.active;
            bool locked = active || view.AwaitingRun || view.CommandBusy;
            bool fresh = view.Connected && view.StatusAge < 8;
            GUILayout.Space(6);
            GUI.enabled = !locked;
            destination = GUILayout.Toolbar(destination, new string[] { "Mun", "Minmus" });
            GUILayout.BeginHorizontal();
            GUILayout.Label("Orbit altitude", GUILayout.Width(105));
            altitudeKm = GUILayout.TextField(altitudeKm, 6, GUILayout.Width(75));
            GUILayout.Label("km  (40–60)");
            GUILayout.EndHorizontal();
            double altitude;
            bool valid = Double.TryParse(altitudeKm, NumberStyles.Float, CultureInfo.InvariantCulture, out altitude)
                         && !Double.IsNaN(altitude) && !Double.IsInfinity(altitude) && altitude >= 40 && altitude <= 60;
            GUI.enabled = true;
            GUILayout.Space(8);
            GUILayout.BeginHorizontal();
            GUI.enabled = fresh && valid && !locked && status != null && status.can_run;
            if (GUILayout.Button("Run", GUILayout.Height(30)))
                bridge.RequestRun(destination == 0 ? "Mun" : "Minmus", altitude * 1000);
            GUI.enabled = !view.StopPending && (active || view.AwaitingRun);
            if (GUILayout.Button(view.StopPending || (status != null && status.state == "stopping") ? "Stopping…" : "Stop", GUILayout.Height(30)))
                bridge.RequestStop("Stop button");
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            if (!valid) GUILayout.Label("Choose an altitude from 40 to 60 km.", wrapped);
            GUILayout.Space(8);
            string heading = !fresh ? "Connecting / status unavailable" : HumanPhase(status);
            GUILayout.Label(heading, wrapped);
            if (!String.IsNullOrEmpty(view.Message)) GUILayout.Label(Short(view.Message, 220), wrapped);
            if (!String.IsNullOrEmpty(view.Error)) GUILayout.Label(Short(view.Error, 320), wrapped);
            if (fresh && status != null && !active && status.preflight != null && !status.preflight.ready &&
                status.state != "succeeded" && status.preflight.failures != null && status.preflight.failures.Length > 0)
                GUILayout.Label(Short(status.preflight.failures[0], 220), wrapped);
            if (status != null && status.mission != null)
                GUILayout.Label("Mission: " + status.mission.target + " · " +
                    (status.mission.target_altitude_m / 1000).ToString("0.#", CultureInfo.InvariantCulture) + " km", wrapped);
            if (view.StatusAge < Double.PositiveInfinity)
                GUILayout.Label("Last status: " + Age(view.StatusAge) + " ago" + (fresh ? "" : " (stale)"), wrapped);
            if (status != null && status.telemetry != null && status.telemetry.wall_time > 0)
            {
                double age = Math.Max(0, (DateTime.UtcNow - Epoch).TotalSeconds - status.telemetry.wall_time);
                GUILayout.Label("Last flight sample: " + Age(age) + " ago", wrapped);
            }
            GUILayout.Space(5);
            GUILayout.Label("Run launches the active, supported rocket and verifies a full orbit after capture.", wrapped);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Refresh connection")) bridge.RequestConnection();
            if (GUILayout.Button("Close")) CloseWindow();
            GUILayout.EndHorizontal();
            GUI.DragWindow(new Rect(0, 0, window.width, 25));
        }

        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        private static string Age(double seconds) { return seconds < 60 ? Math.Floor(seconds) + " s" : Math.Floor(seconds / 60) + " min"; }
        private static string Short(string text, int limit) { return text == null || text.Length <= limit ? text : text.Substring(0, limit - 1) + "…"; }

        private static string HumanPhase(Status status)
        {
            if (status == null) return "Checking launchpad";
            if (status.state == "succeeded") return Verified(status) ? "Orbit verified" : "Orbit result is incomplete";
            if (status.state == "stopped") return CleanupConfirmed(status) ? "Stopped · control cleanup confirmed" : "Worker stopped · control cleanup unconfirmed";
            if (status.state == "failed") return "Mission stopped with an error";
            if (status.state == "stopping") return "Stopping · waiting for control cleanup";
            if (!status.active) return status.can_run ? "Ready on the launchpad" : "Launchpad check";
            switch (status.phase)
            {
                case "ascent": return "Launch · climbing to parking orbit";
                case "transfer_planning": return "Parking orbit · checking an encounter";
                case "maneuver": return "Transfer · executing a maneuver";
                case "transfer_coast": return "Transfer · coasting to the destination";
                case "capture_coast": return "Capture · approaching the burn";
                case "capture": return "Capture · establishing the target orbit";
                case "orbit_verification": return "Verifying a full unpowered orbit";
                default: return "Preparing the mission";
            }
        }

        private static bool Verified(Status status)
        {
            Result result = status.mission == null ? null : status.mission.result;
            return result != null && result.success && result.end_to_end_from_launch_verified &&
                   result.launch != null && result.launch.success && result.mission != null &&
                   result.mission.independent_coast != null && result.mission.independent_coast.success;
        }

        private static bool CleanupConfirmed(Status status)
        {
            Cleanup cleanup = status.mission == null ? null : status.mission.control_cleanup;
            return cleanup != null && cleanup.verified && (cleanup.errors == null || cleanup.errors.Length == 0);
        }

        private void CloseWindow()
        {
            HideWindow();
            if (button != null) button.SetFalse(false);
        }

        private void OnApplicationQuit() { if (bridge != null) bridge.RequestStop("game exit"); }

        private void OnDestroy()
        {
            if (instance != this) return;
            GameEvents.onGUIApplicationLauncherReady.Remove(AddButton);
            GameEvents.onGUIApplicationLauncherDestroyed.Remove(RemoveButton);
            GameEvents.onGameSceneLoadRequested.Remove(SceneRequested);
            HideWindow();
            RemoveButton();
            if (icon != null) Destroy(icon);
            if (bridge != null) bridge.Shutdown(); // Never kills the Python service.
            instance = null;
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

        private sealed class Settings
        {
            internal string Python, Root;
            internal int Port = 8765, RpcPort = 50000;
            internal bool AutoStart = true;
            internal string Origin { get { return "http://127.0.0.1:" + Port; } }
            internal static Settings Load(string file)
            {
                ConfigNode root = ConfigNode.Load(file);
                ConfigNode node = root == null ? null : root.GetNode("ORBITER_AUTOPILOT");
                if (node == null) throw new InvalidOperationException("Install OrbiterAutopilot/settings.cfg first.");
                Settings settings = new Settings();
                settings.Python = node.GetValue("python");
                settings.Root = node.GetValue("project_root");
                if (String.IsNullOrEmpty(settings.Root) || !Path.IsPathRooted(settings.Root) || !Directory.Exists(settings.Root))
                    throw new InvalidOperationException("The configured project_root must be an existing absolute directory.");
                settings.Root = Path.GetFullPath(settings.Root).TrimEnd(Path.DirectorySeparatorChar);
                if (node.HasValue("port") && !Int32.TryParse(node.GetValue("port"), out settings.Port))
                    throw new InvalidOperationException("The configured HTTP port is invalid.");
                if (node.HasValue("rpc_port") && !Int32.TryParse(node.GetValue("rpc_port"), out settings.RpcPort))
                    throw new InvalidOperationException("The configured kRPC port is invalid.");
                if (settings.Port < 1 || settings.Port > 65535 || settings.RpcPort < 1 || settings.RpcPort > 65535)
                    throw new InvalidOperationException("Configured ports must be between 1 and 65535.");
                if (node.HasValue("auto_start") && !Boolean.TryParse(node.GetValue("auto_start"), out settings.AutoStart))
                    throw new InvalidOperationException("auto_start must be true or false.");
                return settings;
            }
        }

        private sealed class RunTicket
        {
            internal string Target, BaselineJob, Instance;
            internal double Altitude;
            internal DateTime Sent;
        }

        private sealed class StopTicket
        {
            internal string Job, Instance;
            internal RunTicket UnconfirmedRun;
            internal int Attempts;
        }

        private sealed class View
        {
            internal Status Status;
            internal string Error, Message;
            internal bool Connected, CommandBusy, AwaitingRun, StopPending;
            internal double StatusAge;
        }

        private sealed class Bridge
        {
            private readonly Settings settings;
            private readonly string logFile;
            private readonly object sync = new object();
            private readonly AutoResetEvent wake = new AutoResetEvent(false);
            private readonly Queue<string> messages = new Queue<string>();
            private Session session;
            private Status status;
            private DateTime received = DateTime.MinValue;
            private RunTicket pendingRun, attemptedRun;
            private StopTicket pendingStop;
            private bool requested, connected, busy, awaitingRun, postingRun, shutdown, autoStarted;
            private string error, message;
            private string loggedResult;
            internal volatile bool FlightScene;

            internal Bridge(Settings settings, string logFile) { this.settings = settings; this.logFile = logFile; }
            internal void Start() { new Thread(Loop) { IsBackground = true, Name = "OrbiterAutopilot HTTP" }.Start(); }
            internal void RequestConnection()
            {
                bool first;
                lock (sync) { first = !requested; requested = true; }
                if (first) wake.Set();
            }

            internal View Read()
            {
                lock (sync) return new View {
                    Status = status, Error = error, Message = message, Connected = connected,
                    CommandBusy = busy, AwaitingRun = awaitingRun, StopPending = pendingStop != null,
                    StatusAge = received == DateTime.MinValue ? Double.PositiveInfinity : Math.Max(0, (DateTime.UtcNow - received).TotalSeconds)
                };
            }

            internal string[] TakeMessages()
            {
                lock (sync) { string[] result = messages.ToArray(); messages.Clear(); return result; }
            }

            private void Log(string text) { lock (sync) { if (messages.Count < 30) messages.Enqueue(text); } }

            internal void RequestRun(string target, double altitude)
            {
                lock (sync)
                {
                    if (!FlightScene || !connected || busy || awaitingRun || status == null || status.active || !status.can_run ||
                        session == null || (DateTime.UtcNow - received).TotalSeconds >= 8) return;
                    pendingRun = new RunTicket { Target = target, Altitude = altitude, Instance = session.server_instance,
                                                BaselineJob = status.mission == null ? null : status.mission.id };
                    attemptedRun = pendingRun;
                    busy = awaitingRun = requested = true;
                    message = "Starting the mission…"; error = null;
                    Log("Run requested: " + target + " at " + altitude.ToString("0", CultureInfo.InvariantCulture) + " m");
                }
                wake.Set();
            }

            internal void RequestStop(string reason)
            {
                lock (sync)
                {
                    if (pendingStop != null) return;
                    if (pendingRun != null && !postingRun)
                    {
                        pendingRun = null; awaitingRun = busy = false;
                        message = "Run cancelled before it was sent.";
                        return;
                    }
                    if (awaitingRun)
                        pendingStop = new StopTicket { Instance = attemptedRun.Instance, UnconfirmedRun = attemptedRun };
                    else if (status != null && status.active && status.mission != null && session != null)
                        pendingStop = new StopTicket { Job = status.mission.id, Instance = session.server_instance };
                    else return;
                    requested = true;
                    message = "Stop requested; waiting for verified control cleanup.";
                    Log("Stop requested (" + reason + ")");
                }
                wake.Set();
            }

            internal void Shutdown() { RequestStop("addon shutdown"); lock (sync) shutdown = true; wake.Set(); }

            private void Loop()
            {
                while (true)
                {
                    wake.WaitOne(1500);
                    RunTicket run;
                    StopTicket stop;
                    lock (sync)
                    {
                        if (shutdown && pendingStop == null && !postingRun && !awaitingRun) return;
                        if (!requested) continue;
                        if (!FlightScene && pendingStop == null && !awaitingRun &&
                            (status == null || !status.active)) continue;
                        run = pendingRun; stop = pendingStop;
                        if (run != null && stop == null) { pendingRun = null; postingRun = true; }
                    }
                    try
                    {
                        EnsureSession();
                        if (stop != null) ExecuteStop(stop);
                        else if (run != null) ExecuteRun(run);
                        PollStatus();
                    }
                    catch (Exception failure)
                    {
                        lock (sync) { connected = false; error = failure.Message; busy = postingRun; }
                        if (run != null)
                        {
                            // Even a transport failure can happen after Run was accepted.
                            lock (sync)
                            {
                                postingRun = false; busy = false;
                                if (run.Sent == DateTime.MinValue)
                                {
                                    awaitingRun = false;
                                    if (pendingStop != null && pendingStop.UnconfirmedRun == run) pendingStop = null;
                                    message = "Run was not sent. Check the connection before trying again.";
                                }
                                else if (awaitingRun) message = "Run acceptance is unknown; checking status without retrying Run.";
                            }
                        }
                    }
                    lock (sync)
                    {
                        if (shutdown && pendingStop == null && !postingRun && !awaitingRun) return;
                    }
                }
            }

            private void EnsureSession()
            {
                string json;
                try { json = Request("GET", "/api/session", null, 3000); }
                catch (WebException failure)
                {
                    if (failure.Status != WebExceptionStatus.ConnectFailure) throw;
                    if (!settings.AutoStart || autoStarted) throw new InvalidOperationException("The local autopilot service is unavailable. Check backend.log.");
                    StartBackend();
                    throw new InvalidOperationException("Starting the local autopilot service…");
                }
                Session next = ParseSession(json);
                if (next == null || next.application != "orbiter-autopilot" || next.protocol_version != 1 ||
                    next.rpc_port != settings.RpcPort ||
                    String.IsNullOrEmpty(next.server_instance) || String.IsNullOrEmpty(next.csrf_token) ||
                    String.IsNullOrEmpty(next.project_root) ||
                    !SamePath(next.project_root, settings.Root) ||
                    !SamePath(next.launch_model, Path.Combine(settings.Root, "models/transfer_ascent_ppo.zip")) ||
                    !SamePath(next.capture_model, Path.Combine(settings.Root, "models/moon_capture.zip")))
                    throw new InvalidOperationException("Port " + settings.Port + " belongs to an incompatible service or another project. No command was sent.");
                lock (sync) session = next;
            }

            private static bool SamePath(string value, string expected)
            {
                return !String.IsNullOrEmpty(value) && Path.IsPathRooted(value) &&
                    String.Equals(Path.GetFullPath(value).TrimEnd(Path.DirectorySeparatorChar),
                                  Path.GetFullPath(expected).TrimEnd(Path.DirectorySeparatorChar), StringComparison.Ordinal);
            }

            private void StartBackend()
            {
                autoStarted = true;
                if (String.IsNullOrEmpty(settings.Python) || !Path.IsPathRooted(settings.Python) || !File.Exists(settings.Python))
                    throw new InvalidOperationException("The configured Python executable is missing. Reinstall the Orbiter mod configuration.");
                string arguments = "-u -m orbiter.control --project-root " + Quote(settings.Root) +
                    " --port " + settings.Port + " --rpc-port " + settings.RpcPort + " --no-browser --log-file " + Quote(logFile);
                ProcessStartInfo info = new ProcessStartInfo(settings.Python, arguments) {
                    WorkingDirectory = settings.Root, UseShellExecute = false, CreateNoWindow = true
                };
                info.EnvironmentVariables["OMP_NUM_THREADS"] = "1";
                info.EnvironmentVariables["MKL_NUM_THREADS"] = "1";
                info.EnvironmentVariables["OPENBLAS_NUM_THREADS"] = "1";
                using (Process child = Process.Start(info))
                {
                    if (child == null) throw new InvalidOperationException("The local autopilot service could not start.");
                    Log("Started configured Python backend, process " + child.Id);
                }
            }

            private void ExecuteRun(RunTicket run)
            {
                lock (sync)
                {
                    // Stop and the send decision share one linearization point.
                    // A Stop queued during EnsureSession cancels this exact Run;
                    // once Sent is committed, Stop must reconcile its job ID.
                    bool stoppedBeforeSend = pendingStop != null && pendingStop.UnconfirmedRun == run;
                    if (!FlightScene || stoppedBeforeSend)
                    {
                        awaitingRun = busy = postingRun = false;
                        if (pendingStop != null && pendingStop.UnconfirmedRun == run) pendingStop = null;
                        message = stoppedBeforeSend ? "Run cancelled before it was sent." : "Run cancelled after leaving flight.";
                        return;
                    }
                    if (session.server_instance != run.Instance)
                    {
                        awaitingRun = busy = postingRun = false;
                        throw new InvalidOperationException("The service restarted before Run. Review the launchpad status and try again.");
                    }
                    run.Sent = DateTime.UtcNow;
                }
                try
                {
                    string body = "{\"target\":\"" + run.Target + "\",\"target_altitude_m\":" +
                                  run.Altitude.ToString("R", CultureInfo.InvariantCulture) + "}";
                    ApplyStatus(ParseStatus(Request("POST", "/api/run", body, 25000)));
                    ReconcileRun();
                }
                catch (ApiFailure failure)
                {
                    if (failure.Code >= 400 && failure.Code < 500)
                        lock (sync) { awaitingRun = false; message = "Run was rejected."; }
                    else lock (sync) message = "Run acceptance is unknown; checking status before any further command.";
                    throw;
                }
                finally { lock (sync) { postingRun = false; busy = false; } }
            }

            private bool MatchesRun(Mission mission, RunTicket run)
            {
                return mission != null && !String.IsNullOrEmpty(mission.id) && mission.id != run.BaselineJob &&
                       mission.target == run.Target && Math.Abs(mission.target_altitude_m - run.Altitude) < 0.1 &&
                       session.server_instance == run.Instance;
            }

            private void ReconcileRun()
            {
                lock (sync)
                {
                    if (!awaitingRun || attemptedRun == null || status == null) return;
                    if (MatchesRun(status.mission, attemptedRun))
                    {
                        awaitingRun = false;
                        message = status.active ? "Mission accepted. Guidance and staging are active." : TerminalMessage(status);
                        Log("Run accepted, job " + status.mission.id);
                    }
                    else if (!status.active && attemptedRun.Sent != DateTime.MinValue &&
                             (DateTime.UtcNow - attemptedRun.Sent).TotalSeconds > 60)
                    {
                        awaitingRun = false;
                        message = "Run was not confirmed and no active mission is reported. Review the launchpad before trying again.";
                    }
                }
            }

            private void ExecuteStop(StopTicket stop)
            {
                if (session.server_instance != stop.Instance)
                {
                    lock (sync) pendingStop = null;
                    throw new InvalidOperationException("The service restarted; the requested flight identity no longer matches. No Stop was sent.");
                }
                if (String.IsNullOrEmpty(stop.Job))
                {
                    PollStatus();
                    lock (sync)
                    {
                        if (MatchesRun(status.mission, stop.UnconfirmedRun)) stop.Job = status.mission.id;
                        else if (!awaitingRun) { pendingStop = null; return; }
                        else return;
                    }
                }
                stop.Attempts++;
                try
                {
                    string body = "{\"expected_job_id\":\"" + stop.Job + "\",\"server_instance\":\"" + stop.Instance + "\"}";
                    ApplyStatus(ParseStatus(Request("POST", "/api/stop", body, 25000)));
                    lock (sync)
                    {
                        pendingStop = null;
                        message = status.active ? "Stop requested. Waiting for verified control cleanup." : TerminalMessage(status);
                    }
                }
                catch (ApiFailure failure)
                {
                    if (failure.Code == 409) lock (sync) pendingStop = null;
                    throw;
                }
                finally { if (stop.Attempts >= 3) lock (sync) pendingStop = null; }
            }

            private void PollStatus()
            {
                ApplyStatus(ParseStatus(Request("GET", "/api/status", null, 25000)));
                ReconcileRun();
            }

            private static Status ParseStatus(string json)
            {
                // Decode network JSON with KSP's managed decoder. Unity's native
                // field serializer lost the populated mission ID in a live reply;
                // explicit mapping also keeps numbers/strings from becoming proof.
                Hashtable data = JsonObject(json);
                Status value = new Status {
                    state = Text(data, "state", true), phase = Text(data, "phase", true),
                    error = Text(data, "error", false), active = Flag(data, "active", true),
                    can_run = Flag(data, "can_run", true)
                };
                if (Array.IndexOf(new string[] { "idle", "running", "stopping", "succeeded", "failed", "stopped" }, value.state) < 0)
                    throw new InvalidOperationException("The autopilot returned an invalid status. No flight result is confirmed.");
                Hashtable preflight = ObjectField(data, "preflight");
                if (preflight != null) value.preflight = new Preflight { ready = Flag(preflight, "ready", true), failures = TextArray(preflight, "failures", true) };
                Hashtable telemetry = ObjectField(data, "telemetry");
                if (telemetry != null) value.telemetry = new Telemetry { wall_time = Number(telemetry, "wall_time", false) };
                Hashtable mission = ObjectField(data, "mission");
                if (mission != null)
                {
                    value.mission = new Mission {
                        id = Text(mission, "id", true), target = Text(mission, "target", true),
                        target_altitude_m = Number(mission, "target_altitude_m", true)
                    };
                    if (!SafeJob(value.mission.id) || (value.mission.target != "Mun" && value.mission.target != "Minmus") ||
                        value.mission.target_altitude_m < 40000 || value.mission.target_altitude_m > 60000)
                        throw InvalidField("mission identity or target");
                    Hashtable result = ObjectField(mission, "result");
                    if (result != null)
                    {
                        value.mission.result = new Result {
                            success = Flag(result, "success", false),
                            end_to_end_from_launch_verified = Flag(result, "end_to_end_from_launch_verified", false),
                            launch = Proof(ObjectField(result, "launch"))
                        };
                        Hashtable proof = ObjectField(result, "mission");
                        if (proof != null) value.mission.result.mission = new MissionProof { independent_coast = Proof(ObjectField(proof, "independent_coast")) };
                    }
                    Hashtable cleanup = ObjectField(mission, "control_cleanup");
                    if (cleanup != null) value.mission.control_cleanup = new Cleanup {
                        verified = Flag(cleanup, "verified", true), errors = TextArray(cleanup, "errors", true)
                    };
                }
                if (value.active && (value.mission == null || String.IsNullOrEmpty(value.mission.id)))
                    throw new InvalidOperationException("The active mission has no verified job identity. No Stop target is confirmed.");
                if (value.active != (value.state == "running" || value.state == "stopping") || (value.active && value.can_run))
                    throw InvalidField("mission activity");
                return value;
            }

            private static Session ParseSession(string json)
            {
                Hashtable data = JsonObject(json);
                double protocol = Number(data, "protocol_version", true), rpcPort = Number(data, "rpc_port", true);
                if (protocol != Math.Truncate(protocol) || rpcPort != Math.Truncate(rpcPort) || protocol != 1 || rpcPort < 1 || rpcPort > 65535)
                    throw InvalidField("session protocol or RPC port");
                return new Session {
                    application = Text(data, "application", true), protocol_version = (int)protocol, rpc_port = (int)rpcPort,
                    server_instance = Text(data, "server_instance", true), csrf_token = Text(data, "csrf_token", true),
                    project_root = Text(data, "project_root", true), launch_model = Text(data, "launch_model", true),
                    capture_model = Text(data, "capture_model", true)
                };
            }

            private static readonly object jsonSync = new object();
            private static Hashtable JsonObject(string json)
            {
                lock (jsonSync)
                {
                    Hashtable data = EdyCommonTools.MiniJSON.jsonDecode(json) as Hashtable;
                    if (data == null || !EdyCommonTools.MiniJSON.lastDecodeSuccessful()) throw InvalidField("JSON object");
                    return data;
                }
            }
            private static InvalidOperationException InvalidField(string field) { return new InvalidOperationException("The autopilot returned invalid " + field + ". No command or flight proof is confirmed."); }
            private static Hashtable ObjectField(Hashtable data, string field)
            {
                object value = data[field];
                if (value == null) return null;
                Hashtable result = value as Hashtable;
                if (result == null) throw InvalidField(field);
                return result;
            }
            private static string Text(Hashtable data, string field, bool required)
            {
                object value = data[field];
                if (value == null && !required) return null;
                string result = value as string;
                if (result == null || (required && result.Length == 0)) throw InvalidField(field);
                return result;
            }
            private static bool Flag(Hashtable data, string field, bool required)
            {
                if (!data.ContainsKey(field) && !required) return false;
                if (!(data[field] is bool)) throw InvalidField(field);
                return (bool)data[field];
            }
            private static double Number(Hashtable data, string field, bool required)
            {
                if (!data.ContainsKey(field) && !required) return 0;
                object value = data[field];
                if (!(value is double) && !(value is int) && !(value is long) && !(value is decimal)) throw InvalidField(field);
                double result = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                if (Double.IsNaN(result) || Double.IsInfinity(result)) throw InvalidField(field);
                return result;
            }
            private static string[] TextArray(Hashtable data, string field, bool required)
            {
                if (!data.ContainsKey(field) && !required) return new string[0];
                ArrayList values = data[field] as ArrayList;
                if (values == null) throw InvalidField(field);
                string[] result = new string[values.Count];
                for (int i = 0; i < values.Count; i++)
                {
                    result[i] = values[i] as string;
                    if (result[i] == null) throw InvalidField(field);
                }
                return result;
            }
            private static Success Proof(Hashtable data) { return data == null ? null : new Success { success = Flag(data, "success", false) }; }
            private static bool SafeJob(string value)
            {
                if (String.IsNullOrEmpty(value) || value.Length > 128) return false;
                foreach (char character in value)
                    if (!((character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') ||
                        (character >= '0' && character <= '9') || character == '-' || character == '_')) return false;
                return true;
            }

            private static string TerminalMessage(Status value)
            {
                if (value.state == "succeeded") return Verified(value) ? "Orbit verified. A full unpowered orbit completed." : "The worker ended, but complete orbit proof is missing.";
                if (value.state == "stopped") return CleanupConfirmed(value) ? "Stopped. The backend verified control cleanup." : "The worker stopped, but control cleanup is unconfirmed.";
                if (value.state == "failed") return "Mission ended with an error. Review the reported failure.";
                return null;
            }

            private void ApplyStatus(Status next)
            {
                lock (sync)
                {
                    status = next; connected = true; received = DateTime.UtcNow; error = next.error;
                    string terminal = TerminalMessage(next);
                    if (terminal != null && !next.active && (!awaitingRun || (attemptedRun != null && MatchesRun(next.mission, attemptedRun))))
                        message = terminal;
                    string outcome = next.mission == null ? null : next.mission.id + ":" + next.state;
                    if (outcome != loggedResult && (next.state == "stopped" || next.state == "succeeded" || next.state == "failed"))
                    {
                        loggedResult = outcome;
                        Log(next.state == "stopped" && CleanupConfirmed(next) ? "Flight stopped; backend verified control cleanup" :
                            next.state == "succeeded" && Verified(next) ? "Launch-to-orbit mission verified" : "Mission ended without a verified orbit");
                    }
                }
            }

            private string Request(string method, string path, string body, int timeout)
            {
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(settings.Origin + path);
                request.Method = method; request.Timeout = timeout; request.ReadWriteTimeout = timeout;
                request.Proxy = null; request.AllowAutoRedirect = false; request.KeepAlive = false;
                // A single wall deadline also bounds a response body that stalls
                // or drips data; ReadWriteTimeout alone only bounds each read.
                using (System.Threading.Timer deadline = new System.Threading.Timer(
                    delegate { try { request.Abort(); } catch (ObjectDisposedException) { } }, null, timeout, Timeout.Infinite))
                {
                if (body != null)
                {
                    request.ContentType = "application/json";
                    request.Headers["Origin"] = settings.Origin;
                    request.Headers["X-Orbiter-CSRF"] = session.csrf_token;
                    byte[] bytes = Encoding.UTF8.GetBytes(body);
                    request.ContentLength = bytes.Length;
                    using (Stream stream = request.GetRequestStream()) stream.Write(bytes, 0, bytes.Length);
                }
                try
                {
                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                    {
                        string text = ReadResponse(response);
                        if ((int)response.StatusCode < 200 || (int)response.StatusCode >= 300)
                            throw new ApiFailure("The autopilot service returned HTTP " + (int)response.StatusCode, (int)response.StatusCode);
                        return text;
                    }
                }
                catch (WebException failure)
                {
                    HttpWebResponse response = failure.Response as HttpWebResponse;
                    if (response == null) throw;
                    using (response)
                    {
                        string detail = Text(JsonObject(ReadResponse(response)), "error", false);
                        throw new ApiFailure(String.IsNullOrEmpty(detail) ?
                            "The autopilot service rejected the request." : detail, (int)response.StatusCode);
                    }
                }
                }
            }

            private static string ReadResponse(HttpWebResponse response)
            {
                const int limit = 2 * 1024 * 1024;
                if (response.ContentLength > limit) throw new InvalidOperationException("Autopilot response exceeds the supported size.");
                using (Stream stream = response.GetResponseStream())
                using (MemoryStream buffer = new MemoryStream())
                {
                    byte[] chunk = new byte[4096]; int count;
                    while ((count = stream.Read(chunk, 0, chunk.Length)) > 0)
                    {
                        if (buffer.Length + count > limit) throw new InvalidOperationException("Autopilot response exceeds the supported size.");
                        buffer.Write(chunk, 0, count);
                    }
                    return Encoding.UTF8.GetString(buffer.ToArray());
                }
            }

            private static string Quote(string argument)
            {
                StringBuilder quoted = new StringBuilder("\""); int slashes = 0;
                foreach (char character in argument)
                {
                    if (character == '\\') { slashes++; continue; }
                    quoted.Append('\\', character == '"' ? slashes * 2 + 1 : slashes);
                    quoted.Append(character); slashes = 0;
                }
                quoted.Append('\\', slashes * 2); quoted.Append('"');
                return quoted.ToString();
            }
        }

        private sealed class ApiFailure : Exception
        {
            internal readonly int Code;
            internal ApiFailure(string message, int code) : base(message) { Code = code; }
        }

        // Explicitly mapped from the versioned HTTP API; never Unity-serialized.
#pragma warning disable 0649
        [Serializable] private sealed class Session { public string application, server_instance, project_root, csrf_token, launch_model, capture_model; public int protocol_version, rpc_port; }
        [Serializable] private sealed class Status { public string state, phase, error; public bool active, can_run; public Preflight preflight; public Mission mission; public Telemetry telemetry; }
        [Serializable] private sealed class Preflight { public bool ready; public string[] failures; }
        [Serializable] private sealed class Telemetry { public double wall_time; }
        [Serializable] private sealed class Mission { public string id, target; public double target_altitude_m; public Result result; public Cleanup control_cleanup; }
        [Serializable] private sealed class Cleanup { public bool verified; public string[] errors; }
        [Serializable] private sealed class Result { public bool success, end_to_end_from_launch_verified; public Success launch; public MissionProof mission; }
        [Serializable] private sealed class MissionProof { public Success independent_coast; }
        [Serializable] private sealed class Success { public bool success; }
#pragma warning restore 0649
    }
}
