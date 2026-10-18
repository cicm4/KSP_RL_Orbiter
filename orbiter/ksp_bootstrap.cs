// Optional KSP 1.12 addon for unattended entry into the isolated validation save.
// Build with: python -m orbiter.bootstrap --ksp-root /path/to/KSP
// Explicit launch opt-in: KSP -orbiter-bootstrap
// API usage: https://github.com/allista/AutoLoadGame/blob/master/AutoLoadGame.cs
using System;
using System.Collections;
using System.IO;
using UnityEngine;

namespace OrbiterRL
{
    [KSPAddon(KSPAddon.Startup.MainMenu, true)]
    public sealed class OrbiterBootstrap : MonoBehaviour
    {
        private IEnumerator Start()
        {
            // Installing this addon alone never changes the selected save.
            if (Array.IndexOf(Environment.GetCommandLineArgs(), "-orbiter-bootstrap") < 0)
                yield break;

            // Allow the MainMenu and other startup addons to finish initializing.
            for (int frame = 0; frame < 120; frame++)
                yield return null;

            const string folder = "OrbiterRL";
            string path = Path.Combine(KSPUtil.ApplicationRootPath,
                                       "saves/OrbiterRL/persistent.sfs");
            if (!File.Exists(path))
            {
                Debug.LogError("[OrbiterBootstrap] Dedicated save missing: " + path);
                yield break;
            }
            try
            {
                // The dedicated save must match the current KSP version. Refuse
                // incompatible saves instead of silently rewriting/upgrading one.
                Game game = GamePersistence.LoadGame("persistent", folder, true, false);
                if (game == null)
                    throw new InvalidOperationException("Save incompatible or unreadable");
                game.startScene = GameScenes.SPACECENTER;
                HighLogic.SaveFolder = folder;
                HighLogic.CurrentGame = game;
                Debug.Log("[OrbiterBootstrap] Starting isolated save OrbiterRL");
                game.Start();
            }
            catch (Exception error)
            {
                Debug.LogError("[OrbiterBootstrap] Load failed: " + error);
            }
        }
    }
}
