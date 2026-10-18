// Managed stand-ins for compiling the real HTTP bridge without loading KSP.
// No physics, flight-control or native Unity API is implemented by this fixture.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using System.Web.Script.Serialization;

public sealed class KSPAddon : Attribute { public enum Startup { MainMenu } public KSPAddon(Startup start, bool once) { } }
public enum GameScenes { FLIGHT, SPACECENTER }
public enum ControlTypes { ALLBUTCAMERAS }
public static class InputLockManager { public static void RemoveControlLock(string name) { } public static void SetControlLock(ControlTypes types, string name) { } }
public sealed class Signal { public void Add(Action action) { } public void Remove(Action action) { } }
public sealed class SceneSignal { public void Add(Action<GameScenes> action) { } public void Remove(Action<GameScenes> action) { } }
public static class GameEvents { public static Signal onGUIApplicationLauncherReady = new Signal(), onGUIApplicationLauncherDestroyed = new Signal(); public static SceneSignal onGameSceneLoadRequested = new SceneSignal(); }
public static class HighLogic { public static bool LoadedSceneIsFlight = true; public static UnityEngine.GUISkin Skin = new UnityEngine.GUISkin(); }
public static class KSPUtil { public static string ApplicationRootPath = "/unused"; }
public sealed class CelestialBody { public string bodyName; }
public sealed class Vessel { public string vesselName; public CelestialBody mainBody; }
public static class FlightGlobals { public static Vessel ActiveVessel; }
public sealed class ConfigNode { public static ConfigNode Load(string path) { return null; } public ConfigNode GetNode(string name) { return null; } public string GetValue(string key) { return null; } public bool HasValue(string key) { return false; } }

namespace KSP.UI.Screens
{
    public sealed class ApplicationLauncherButton { public void SetFalse(bool invoke) { } }
    public sealed class ApplicationLauncher
    {
        [Flags] public enum AppScenes { FLIGHT = 1, MAPVIEW = 2 }
        public static bool Ready = true;
        public static ApplicationLauncher Instance = new ApplicationLauncher();
        public ApplicationLauncherButton AddModApplication(Action a, Action b, Action c, Action d, Action e, Action f, AppScenes scenes, UnityEngine.Texture2D icon) { return new ApplicationLauncherButton(); }
        public void RemoveModApplication(ApplicationLauncherButton button) { }
    }
}

namespace UnityEngine
{
    public class Object { public static void Destroy(Object value) { } public static void DontDestroyOnLoad(Object value) { } }
    public sealed class GameObject : Object { }
    public class MonoBehaviour : Object { public GameObject gameObject = new GameObject(); public int GetInstanceID() { return 1; } }
    public sealed class GUIStyle { public bool wordWrap; public GUIStyle() { } public GUIStyle(GUIStyle copy) { } }
    public sealed class GUISkin { public GUIStyle label = new GUIStyle(); }
    public sealed class GUILayoutOption { }
    public struct Vector2 { public float x, y; }
    public struct Rect { public float x, y, width, height; public Rect(float a, float b, float c, float d) { x = a; y = b; width = c; height = d; } public bool Contains(Vector2 point) { return false; } }
    public static class Screen { public static int width = 1280, height = 856; }
    public sealed class Event { public static Event current = new Event(); public Vector2 mousePosition; }
    public static class GUI { public static GUISkin skin = new GUISkin(); public static bool enabled; public static void DragWindow(Rect rect) { } }
    public static class GUILayout
    {
        public static Rect Window(int id, Rect rect, Action<int> draw, string title, params GUILayoutOption[] options) { return rect; }
        public static GUILayoutOption Width(float width) { return new GUILayoutOption(); }
        public static GUILayoutOption Height(float height) { return new GUILayoutOption(); }
        public static void Label(string text, params GUILayoutOption[] options) { }
        public static void Label(string text, GUIStyle style) { }
        public static void Space(float size) { }
        public static void BeginHorizontal() { }
        public static void EndHorizontal() { }
        public static int Toolbar(int selected, string[] labels) { return selected; }
        public static string TextField(string text, int length, params GUILayoutOption[] options) { return text; }
        public static bool Button(string text, params GUILayoutOption[] options) { return false; }
    }
    public static class Mathf { public static float Clamp(float value, float low, float high) { return Math.Max(low, Math.Min(high, value)); } public static float Sqrt(float value) { return (float)Math.Sqrt(value); } }
    public enum TextureFormat { RGBA32 }
    public struct Color32 { public Color32(byte r, byte g, byte b, byte a) { } }
    public sealed class Texture2D : Object { public Texture2D(int width, int height, TextureFormat format, bool mipmap) { } public void SetPixels32(Color32[] colors) { } public void Apply() { } }
    public static class Debug { public static void Log(object value) { } public static void LogError(object value) { } }

}

#if !KSP_JSON
namespace EdyCommonTools
{
    // The production field mapper is exercised in every test. A second fixture
    // uses the actual KSP decoder when its local assembly is available.
    public static class MiniJSON
    {
        public static object jsonDecode(string json) { return Build(new JavaScriptSerializer().DeserializeObject(json)); }
        public static bool lastDecodeSuccessful() { return true; }
        private static object Build(object value)
        {
            IDictionary<string, object> fields = value as IDictionary<string, object>;
            if (fields != null)
            {
                Hashtable result = new Hashtable();
                foreach (KeyValuePair<string, object> item in fields) result.Add(item.Key, Build(item.Value));
                return result;
            }
            object[] items = value as object[];
            if (items != null) { ArrayList result = new ArrayList(); foreach (object item in items) result.Add(Build(item)); return result; }
            return value;
        }
    }
}
#endif
