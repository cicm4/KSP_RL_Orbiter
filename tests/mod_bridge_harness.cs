// Runs the production Bridge with a delayed local HTTP server; no Unity loop.
using System;
using System.Reflection;
using System.Threading;
using OrbiterRL;

public static class ModBridgeHarness
{
    private const BindingFlags Members = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
    private static object Call(object target, string method, params object[] arguments) { return target.GetType().GetMethod(method, Members).Invoke(target, arguments); }
    private static object Field(object target, string name) { return target.GetType().GetField(name, Members).GetValue(target); }
    private static void Set(object target, string name, object value) { target.GetType().GetField(name, Members).SetValue(target, value); }

    public static int Main(string[] arguments)
    {
        Type addon = typeof(OrbiterAutopilot);
        Type bridgeType = addon.GetNestedType("Bridge", BindingFlags.NonPublic);
        if (arguments[0] == "PARSE")
        {
            try
            {
                object parsed = bridgeType.GetMethod("ParseStatus", BindingFlags.NonPublic | BindingFlags.Static).Invoke(null,
                    new object[] { System.IO.File.ReadAllText(arguments[1]) });
                object mission = Field(parsed, "mission");
                bool verified = (bool)addon.GetMethod("Verified", BindingFlags.NonPublic | BindingFlags.Static).Invoke(null, new object[] { parsed });
                Type settingsTypeForParse = addon.GetNestedType("Settings", BindingFlags.NonPublic);
                object parsedBridge = Activator.CreateInstance(bridgeType, Members, null,
                    new object[] { Activator.CreateInstance(settingsTypeForParse, true), "/unused/backend.log" }, null);
                Set(parsedBridge, "message", "Mission accepted. Guidance and staging are active.");
                Call(parsedBridge, "ApplyStatus", parsed);
                Console.WriteLine("JOB=" + (mission == null ? "" : Field(mission, "id")));
                Console.WriteLine("VERIFIED=" + verified);
                Console.WriteLine("MESSAGE=" + Field(Call(parsedBridge, "Read"), "Message"));
                return 0;
            }
            catch (TargetInvocationException failure) { Console.Error.WriteLine(failure.InnerException.Message); return 2; }
        }
        Type settingsType = addon.GetNestedType("Settings", BindingFlags.NonPublic);
        object settings = Activator.CreateInstance(settingsType, true);
        Set(settings, "Root", arguments[1]); Set(settings, "Port", Int32.Parse(arguments[0]));
        Set(settings, "RpcPort", 50000); Set(settings, "AutoStart", false);
        object bridge = Activator.CreateInstance(bridgeType, Members, null, new object[] { settings, "/unused/backend.log" }, null);
        Set(bridge, "FlightScene", true);
        Call(bridge, "Start"); Call(bridge, "RequestConnection");
        DateTime deadline = DateTime.UtcNow.AddSeconds(8);
        while (!(bool)Field(Call(bridge, "Read"), "Connected"))
        {
            if (DateTime.UtcNow > deadline) throw new Exception("Initial bridge connection failed");
            Thread.Sleep(10);
        }
        Console.WriteLine("READY"); Console.Out.Flush();
        if (Console.ReadLine() != "RUN") throw new Exception("Expected RUN synchronization");
        Call(bridge, "RequestRun", "Mun", 50000.0);
        Console.WriteLine("RUN_QUEUED"); Console.Out.Flush();
        string action = Console.ReadLine();
        bool sceneInvalidated = true;
        if (action == "SCENE")
        {
            object component = Activator.CreateInstance(addon);
            Set(component, "bridge", bridge);
            Call(component, "SceneRequested", GameScenes.SPACECENTER);
            sceneInvalidated = !(bool)Field(bridge, "FlightScene");
        }
        else if (action == "STOP") Call(bridge, "RequestStop", "headless regression");
        else throw new Exception("Expected STOP or SCENE synchronization");
        Console.WriteLine("STOP_QUEUED"); Console.Out.Flush();
        deadline = DateTime.UtcNow.AddSeconds(8);
        while (true)
        {
            object view = Call(bridge, "Read");
            if (!(bool)Field(view, "AwaitingRun") && !(bool)Field(view, "CommandBusy") && !(bool)Field(view, "StopPending")) break;
            if (DateTime.UtcNow > deadline) throw new Exception("Bridge did not settle after cancellation");
            Thread.Sleep(10);
        }
        object finalView = Call(bridge, "Read");
        object finalStatus = Field(finalView, "Status");
        if (finalStatus != null && Convert.ToString(Field(finalStatus, "state")) == "idle" &&
            Convert.ToString(Field(finalView, "Message")).IndexOf("cancelled", StringComparison.OrdinalIgnoreCase) < 0)
            throw new Exception("Idle polling discarded the before-send cancellation message");
        if (finalStatus != null && Convert.ToString(Field(finalStatus, "state")) == "stopped" &&
            Convert.ToString(Field(finalView, "Message")).IndexOf("waiting", StringComparison.OrdinalIgnoreCase) >= 0)
            throw new Exception("Verified terminal Stop still displays a waiting-for-cleanup message");
        Call(bridge, "Shutdown");
        if (!sceneInvalidated) throw new Exception("Scene exit did not invalidate FlightScene immediately");
        Console.WriteLine("DONE"); Console.Out.Flush();
        return 0;
    }
}
