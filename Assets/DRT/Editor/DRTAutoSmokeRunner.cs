#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Unity.Barracuda;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DRT.Editor
{
    [InitializeOnLoad]
    public static class DRTAutoSmokeRunner
    {
        private const double SmokeDurationSeconds = 75.0;
        private const string WaitingKey = "DRT_SMOKE_WAITING";
        private const string RunningKey = "DRT_SMOKE_RUNNING";
        private const string StartTimeKey = "DRT_SMOKE_START_TIME";
        private const string DurationKey = "DRT_SMOKE_DURATION";
        private const string TravelModeKey = "DRT_SMOKE_TRAVEL_MODE";
        private const string PhysicalDriveModeKey = "DRT_SMOKE_PHYSICAL_DRIVE_MODE";
        private const string PPODrivePolicyKey = "DRT_SMOKE_PPO_DRIVE_POLICY";
        private const string PolicyKey = "DRT_SMOKE_POLICY";
        private const string PPOSpeedLimitEnabledKey = "DRT_SMOKE_PPO_SPEED_LIMIT_ENABLED";
        private const string PPOSpeedLimitMetersPerSecondKey = "DRT_SMOKE_PPO_SPEED_LIMIT_MPS";
        private const string PPOTrainingRouteEnabledKey = "DRT_SMOKE_PPO_TRAINING_ROUTE_ENABLED";
        private const string PPOTrainingRouteStartStopKey = "DRT_SMOKE_PPO_TRAINING_ROUTE_START_STOP";
        private const string PPOTrainingRouteEndStopKey = "DRT_SMOKE_PPO_TRAINING_ROUTE_END_STOP";
        private const string OnnxModelPathKey = "DRT_SMOKE_ONNX_MODEL_PATH";
        private const string PPOOnnxModelPathKey = "DRT_SMOKE_PPO_ONNX_MODEL_PATH";
        private const string ScenePathKey = "DRT_SMOKE_SCENE_PATH";
        private const string LabelKey = "DRT_SMOKE_LABEL";
        private const string ConfigAppliedKey = "DRT_SMOKE_CONFIG_APPLIED";
        private const string TargetCompletedRunsKey = "DRT_SMOKE_TARGET_COMPLETED_RUNS";
        private const string StartWallTimeTicksKey = "DRT_SMOKE_START_WALL_TIME_TICKS";
        private const string LastExportCountKey = "DRT_SMOKE_LAST_EXPORT_COUNT";
        private const string ExitEditorOnStopKey = "DRT_SMOKE_EXIT_EDITOR_ON_STOP";
        private const string CommandLineConsumedKey = "DRT_SMOKE_COMMAND_LINE_CONSUMED";

        private static string SmokeFlagPath =>
            Path.Combine(Path.GetDirectoryName(Application.dataPath), "Temp", "DRT_AutoSmoke.flag");

        static DRTAutoSmokeRunner()
        {
            if (File.Exists(SmokeFlagPath) && EditorApplication.isPlaying)
            {
                EditorApplication.delayCall += StopInterruptedSmokeRun;
            }

            EditorApplication.delayCall += TryStartFromFlag;
            EditorApplication.delayCall += TryStartFromCommandLine;
            EditorApplication.update += PollForSmokeFlag;
            EditorApplication.playModeStateChanged += PlayModeStateChanged;
        }

        private static void StopInterruptedSmokeRun()
        {
            if (!EditorApplication.isPlaying)
            {
                return;
            }

            FinishActiveEpisode("Smoke interrupted by script reload.");
            WriteRunSummaryIfNeeded();
            DeleteSmokeFlag();
            SessionState.SetBool(RunningKey, false);
            Debug.Log("[DRT_SMOKE] Stopping interrupted Play Mode smoke test.");
            EditorApplication.ExitPlaymode();
        }

        private static void PollForSmokeFlag()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return;
            }

            TryStartFromFlag();
        }

        private static void TryStartFromFlag()
        {
            if (!File.Exists(SmokeFlagPath))
            {
                return;
            }

            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return;
            }

            ReadSmokeConfig(SmokeFlagPath);
            DeleteSmokeFlag();
            StartConfiguredSmokeRun();
        }

        private static void TryStartFromCommandLine()
        {
            if (SessionState.GetBool(CommandLineConsumedKey, false))
            {
                return;
            }

            string configPath = GetCommandLineValue("-drtSmokeConfig");
            if (string.IsNullOrWhiteSpace(configPath))
            {
                return;
            }

            SessionState.SetBool(CommandLineConsumedKey, true);

            if (!File.Exists(configPath))
            {
                Debug.LogError($"[DRT_SMOKE] Command-line smoke config was not found. path={configPath}");
                if (Application.isBatchMode)
                {
                    EditorApplication.Exit(2);
                }
                return;
            }

            ReadSmokeConfig(configPath);
            StartConfiguredSmokeRun();
        }

        private static void StartConfiguredSmokeRun()
        {
            OpenConfiguredSceneIfNeeded();
            ApplyEditorConfigBeforePlay();
            SessionState.SetBool(WaitingKey, true);
            SessionState.SetBool(RunningKey, false);
            SessionState.SetBool(ConfigAppliedKey, false);
            Debug.Log("[DRT_SMOKE] Starting Play Mode smoke test.");
            EditorApplication.EnterPlaymode();
        }

        private static void PlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.EnteredPlayMode && SessionState.GetBool(WaitingKey, false))
            {
                SessionState.SetBool(WaitingKey, false);
                SessionState.SetBool(RunningKey, true);
                SessionState.SetFloat(StartTimeKey, (float)EditorApplication.timeSinceStartup);
                SessionState.SetString(StartWallTimeTicksKey, DateTime.Now.Ticks.ToString());
                SessionState.SetInt(LastExportCountKey, -1);
                SessionState.SetBool(ConfigAppliedKey, false);
                EditorApplication.update += UpdateSmoke;
                ApplyRuntimeConfigIfNeeded();
                Debug.Log($"[DRT_SMOKE] Entered Play Mode. label={SessionState.GetString(LabelKey, "-")}");
            }

            if (state == PlayModeStateChange.EnteredEditMode && File.Exists(SmokeFlagPath))
            {
                DeleteSmokeFlag();
                SessionState.SetBool(WaitingKey, false);
                SessionState.SetBool(RunningKey, false);
                Debug.Log("[DRT_SMOKE] Smoke flag removed.");
            }

            if (state == PlayModeStateChange.EnteredEditMode && SessionState.GetBool(RunningKey, false))
            {
                EditorApplication.update -= UpdateSmoke;
                WriteRunSummaryIfNeeded();
                SessionState.SetBool(RunningKey, false);
                Debug.Log("[DRT_SMOKE] Smoke run stopped from Play Mode exit.");
            }

            if (state == PlayModeStateChange.EnteredEditMode &&
                SessionState.GetBool(ExitEditorOnStopKey, false) &&
                !SessionState.GetBool(WaitingKey, false) &&
                !SessionState.GetBool(RunningKey, false))
            {
                Debug.Log("[DRT_SMOKE] Exiting Unity editor after smoke run.");
                EditorApplication.delayCall += () => EditorApplication.Exit(0);
            }
        }

        private static void UpdateSmoke()
        {
            ApplyRuntimeConfigIfNeeded();

            if (!SessionState.GetBool(RunningKey, false))
            {
                return;
            }

            if (TargetCompletedRunsReached())
            {
                StopSmokeRun("Target completed run count reached.", false);
                return;
            }

            double elapsed = EditorApplication.timeSinceStartup - SessionState.GetFloat(StartTimeKey, 0f);
            float durationSeconds = Mathf.Max(1f, SessionState.GetFloat(DurationKey, (float)SmokeDurationSeconds));
            if (elapsed < durationSeconds)
            {
                return;
            }

            StopSmokeRun("Smoke timeout.", true);
        }

        private static void ReadSmokeConfig(string configPath)
        {
            SessionState.SetFloat(DurationKey, (float)SmokeDurationSeconds);
            SessionState.SetString(TravelModeKey, string.Empty);
            SessionState.SetString(PhysicalDriveModeKey, string.Empty);
            SessionState.SetString(PPODrivePolicyKey, string.Empty);
            SessionState.SetString(PolicyKey, string.Empty);
            SessionState.SetString(PPOSpeedLimitEnabledKey, string.Empty);
            SessionState.SetString(PPOSpeedLimitMetersPerSecondKey, string.Empty);
            SessionState.SetString(PPOTrainingRouteEnabledKey, string.Empty);
            SessionState.SetString(PPOTrainingRouteStartStopKey, string.Empty);
            SessionState.SetString(PPOTrainingRouteEndStopKey, string.Empty);
            SessionState.SetString(OnnxModelPathKey, string.Empty);
            SessionState.SetString(PPOOnnxModelPathKey, string.Empty);
            SessionState.SetString(ScenePathKey, string.Empty);
            SessionState.SetString(LabelKey, "-");
            SessionState.SetInt(TargetCompletedRunsKey, 0);
            SessionState.SetBool(ExitEditorOnStopKey, false);

            string[] lines = File.ReadAllLines(configPath);
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                int separatorIndex = line.IndexOf('=');
                if (separatorIndex <= 0)
                {
                    continue;
                }

                string key = line.Substring(0, separatorIndex).Trim();
                string value = line.Substring(separatorIndex + 1).Trim();
                if (key.Equals("durationSeconds", StringComparison.OrdinalIgnoreCase) &&
                    float.TryParse(value, out float durationSeconds))
                {
                    SessionState.SetFloat(DurationKey, durationSeconds);
                }
                else if (key.Equals("travelExecutionMode", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(TravelModeKey, value);
                }
                else if (key.Equals("physicalDriveMode", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PhysicalDriveModeKey, value);
                }
                else if (key.Equals("ppoDrivePolicy", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PPODrivePolicyKey, value);
                }
                else if (key.Equals("nextStopPolicy", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PolicyKey, value);
                }
                else if (key.Equals("ppoSpeedLimitEnabled", StringComparison.OrdinalIgnoreCase) ||
                         key.Equals("usePPOSpeedLimit", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PPOSpeedLimitEnabledKey, value);
                }
                else if (key.Equals("ppoSpeedLimitMetersPerSecond", StringComparison.OrdinalIgnoreCase) ||
                         key.Equals("ppoSpeedLimitMps", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PPOSpeedLimitMetersPerSecondKey, value);
                }
                else if (key.Equals("ppoTrainingRouteEnabled", StringComparison.OrdinalIgnoreCase) ||
                         key.Equals("usePPOTrainingRouteEpisode", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PPOTrainingRouteEnabledKey, value);
                }
                else if (key.Equals("ppoTrainingRouteStartStop", StringComparison.OrdinalIgnoreCase) ||
                         key.Equals("ppoTrainingStartStopId", StringComparison.OrdinalIgnoreCase) ||
                         key.Equals("trainingStartStopId", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PPOTrainingRouteStartStopKey, value);
                }
                else if (key.Equals("ppoTrainingRouteEndStop", StringComparison.OrdinalIgnoreCase) ||
                         key.Equals("ppoTrainingEndStopId", StringComparison.OrdinalIgnoreCase) ||
                         key.Equals("trainingEndStopId", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PPOTrainingRouteEndStopKey, value);
                }
                else if (key.Equals("onnxModelPath", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(OnnxModelPathKey, value);
                }
                else if (key.Equals("ppoOnnxModelPath", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(PPOOnnxModelPathKey, value);
                }
                else if (key.Equals("scenePath", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(ScenePathKey, value);
                }
                else if (key.Equals("label", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetString(LabelKey, value);
                }
                else if (key.Equals("targetCompletedRuns", StringComparison.OrdinalIgnoreCase) &&
                         int.TryParse(value, out int targetCompletedRuns))
                {
                    SessionState.SetInt(TargetCompletedRunsKey, Mathf.Max(0, targetCompletedRuns));
                }
                else if (key.Equals("exitEditorOnStop", StringComparison.OrdinalIgnoreCase))
                {
                    SessionState.SetBool(
                        ExitEditorOnStopKey,
                        value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
                        value.Equals("true", StringComparison.OrdinalIgnoreCase));
                }
            }
        }

        private static string GetCommandLineValue(string key)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i].Equals(key, StringComparison.OrdinalIgnoreCase))
                {
                    return args[i + 1];
                }
            }

            return string.Empty;
        }

        private static void DeleteSmokeFlag()
        {
            if (File.Exists(SmokeFlagPath))
            {
                File.Delete(SmokeFlagPath);
            }
        }

        private static void ApplyEditorConfigBeforePlay()
        {
            if (EditorApplication.isPlaying)
            {
                return;
            }

            var busController = UnityEngine.Object.FindObjectOfType<DRTBusController>();
            var nextStopSelector = UnityEngine.Object.FindObjectOfType<DRTNextStopSelector>();
            if (busController == null || nextStopSelector == null)
            {
                return;
            }

            ApplyConfig(busController, nextStopSelector);
        }

        private static void ApplyRuntimeConfigIfNeeded()
        {
            if (!EditorApplication.isPlaying ||
                SessionState.GetBool(ConfigAppliedKey, false) ||
                !SessionState.GetBool(RunningKey, false))
            {
                return;
            }

            var busController = UnityEngine.Object.FindObjectOfType<DRTBusController>();
            var nextStopSelector = UnityEngine.Object.FindObjectOfType<DRTNextStopSelector>();
            if (busController == null || nextStopSelector == null)
            {
                return;
            }

            ApplyConfig(busController, nextStopSelector);
            SessionState.SetBool(ConfigAppliedKey, true);
            Debug.Log(
                $"[DRT_SMOKE] Applied runtime config. label={SessionState.GetString(LabelKey, "-")}, " +
                $"mode={busController.TravelExecutionModeName}, physicalDriver={busController.PhysicalDriveModeName}, " +
                $"ppoPolicy={busController.PPODrivePolicyName}, policy={nextStopSelector.NextStopPolicyName}, " +
                $"ppoRoute={busController.PPOTrainingRouteName}, " +
                $"ppoSpeedLimit={(busController.UsePPOSpeedLimit ? $"{busController.PPOSpeedLimitMetersPerSecond:0.00}m/s" : "off")}");
        }

        private static void ApplyConfig(
            DRTBusController busController,
            DRTNextStopSelector nextStopSelector)
        {
            string travelModeText = SessionState.GetString(TravelModeKey, string.Empty);
            if (Enum.TryParse(travelModeText, true, out DRTTravelExecutionMode travelMode))
            {
                SetPrivateField(busController, "travelExecutionMode", travelMode);
            }

            string physicalDriveModeText = SessionState.GetString(PhysicalDriveModeKey, string.Empty);
            if (Enum.TryParse(physicalDriveModeText, true, out DRTPhysicalDriveMode physicalDriveMode))
            {
                SetPrivateField(busController, "physicalDriveMode", physicalDriveMode);
                SetPrivateField(
                    busController,
                    "useGleyVehicleControlInPhysicalDrive",
                    physicalDriveMode == DRTPhysicalDriveMode.Gley ||
                    physicalDriveMode == DRTPhysicalDriveMode.NoisyGley);
            }

            string ppoDrivePolicyText = SessionState.GetString(PPODrivePolicyKey, string.Empty);
            if (Enum.TryParse(ppoDrivePolicyText, true, out DRTPPODrivePolicy ppoDrivePolicy))
            {
                SetPrivateField(busController, "ppoDrivePolicy", ppoDrivePolicy);
            }

            string policyText = SessionState.GetString(PolicyKey, string.Empty);
            if (Enum.TryParse(policyText, true, out DRTNextStopPolicy policy))
            {
                SetPrivateField(nextStopSelector, "nextStopPolicy", policy);
            }

            string ppoSpeedLimitEnabledText = SessionState.GetString(PPOSpeedLimitEnabledKey, string.Empty);
            if (TryParseConfigBool(ppoSpeedLimitEnabledText, out bool ppoSpeedLimitEnabled))
            {
                SetPrivateField(busController, "usePPOSpeedLimit", ppoSpeedLimitEnabled);
            }

            string ppoSpeedLimitText = SessionState.GetString(PPOSpeedLimitMetersPerSecondKey, string.Empty);
            if (float.TryParse(ppoSpeedLimitText, NumberStyles.Float, CultureInfo.InvariantCulture, out float ppoSpeedLimitMetersPerSecond))
            {
                SetPrivateField(busController, "ppoSpeedLimitMetersPerSecond", Mathf.Max(0.5f, ppoSpeedLimitMetersPerSecond));
            }

            string routeEnabledText = SessionState.GetString(PPOTrainingRouteEnabledKey, string.Empty);
            if (TryParseConfigBool(routeEnabledText, out bool routeEnabled))
            {
                SetPrivateField(busController, "usePPOTrainingRouteEpisode", routeEnabled);
            }

            string routeStartStopText = SessionState.GetString(PPOTrainingRouteStartStopKey, string.Empty);
            if (int.TryParse(routeStartStopText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int routeStartStopId))
            {
                SetPrivateField(busController, "ppoTrainingRouteStartStopId", Mathf.Max(1, routeStartStopId));
            }

            string routeEndStopText = SessionState.GetString(PPOTrainingRouteEndStopKey, string.Empty);
            if (int.TryParse(routeEndStopText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int routeEndStopId))
            {
                SetPrivateField(busController, "ppoTrainingRouteEndStopId", Mathf.Max(1, routeEndStopId));
            }

            ApplyConfiguredOnnxModel(nextStopSelector);
            ApplyConfiguredPPOOnnxModel(busController);

            nextStopSelector.Configure(busController);
        }

        private static bool TryParseConfigBool(string value, out bool result)
        {
            result = false;
            if (string.IsNullOrWhiteSpace(value))
            {
                return false;
            }

            if (value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("yes", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("on", StringComparison.OrdinalIgnoreCase))
            {
                result = true;
                return true;
            }

            if (value.Equals("0", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("false", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("no", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("off", StringComparison.OrdinalIgnoreCase))
            {
                result = false;
                return true;
            }

            return bool.TryParse(value, out result);
        }

        private static void OpenConfiguredSceneIfNeeded()
        {
            string scenePath = SessionState.GetString(ScenePathKey, string.Empty);
            if (string.IsNullOrWhiteSpace(scenePath))
            {
                return;
            }

            string normalizedScenePath = NormalizeProjectAssetPath(scenePath);
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            if (string.IsNullOrWhiteSpace(normalizedScenePath) ||
                !File.Exists(Path.Combine(projectRoot, normalizedScenePath)))
            {
                Debug.LogError($"[DRT_SMOKE] Scene path was not found. path={scenePath}");
                return;
            }

            EditorSceneManager.OpenScene(normalizedScenePath, OpenSceneMode.Single);
        }

        private static void ApplyConfiguredOnnxModel(DRTNextStopSelector nextStopSelector)
        {
            string modelPath = SessionState.GetString(OnnxModelPathKey, string.Empty);
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                return;
            }

            string assetPath = NormalizeProjectAssetPath(modelPath);
            if (string.IsNullOrWhiteSpace(assetPath))
            {
                Debug.LogError($"[DRT_SMOKE] ONNX model path is outside this project. path={modelPath}");
                return;
            }

            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
            var model = AssetDatabase.LoadAssetAtPath<NNModel>(assetPath);
            if (model == null)
            {
                Debug.LogError($"[DRT_SMOKE] Failed to load ONNX NNModel at {assetPath}");
                return;
            }

            SetPrivateField(nextStopSelector, "onnxInferenceModel", model);
        }

        private static void ApplyConfiguredPPOOnnxModel(DRTBusController busController)
        {
            string modelPath = SessionState.GetString(PPOOnnxModelPathKey, string.Empty);
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                return;
            }

            string assetPath = NormalizeProjectAssetPath(modelPath);
            if (string.IsNullOrWhiteSpace(assetPath))
            {
                Debug.LogError($"[DRT_SMOKE] PPO ONNX model path is outside this project. path={modelPath}");
                return;
            }

            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
            var model = AssetDatabase.LoadAssetAtPath<NNModel>(assetPath);
            if (model == null)
            {
                Debug.LogError($"[DRT_SMOKE] Failed to load PPO ONNX NNModel at {assetPath}");
                return;
            }

            SetPrivateField(busController, "ppoOnnxInferenceModel", model);
        }

        private static string NormalizeProjectAssetPath(string modelPath)
        {
            string normalized = modelPath.Trim().Replace('\\', '/');
            if (normalized.StartsWith("Assets/", StringComparison.OrdinalIgnoreCase))
            {
                return normalized;
            }

            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..")).Replace('\\', '/');
            if (Path.IsPathRooted(modelPath))
            {
                string absolutePath = Path.GetFullPath(modelPath).Replace('\\', '/');
                if (!absolutePath.StartsWith(projectRoot + "/", StringComparison.OrdinalIgnoreCase))
                {
                    return string.Empty;
                }

                return absolutePath.Substring(projectRoot.Length + 1);
            }

            string candidateAbsolutePath = Path.GetFullPath(Path.Combine(projectRoot, normalized)).Replace('\\', '/');
            if (!candidateAbsolutePath.StartsWith(projectRoot + "/", StringComparison.OrdinalIgnoreCase))
            {
                return string.Empty;
            }

            return candidateAbsolutePath.Substring(projectRoot.Length + 1);
        }

        private static void StopSmokeRun(string reason, bool finishActiveEpisode)
        {
            SessionState.SetBool(RunningKey, false);
            EditorApplication.update -= UpdateSmoke;
            if (finishActiveEpisode)
            {
                FinishActiveEpisode(reason);
            }

            WriteRunSummaryIfNeeded();
            Debug.Log($"[DRT_SMOKE] Stopping Play Mode smoke test. reason={reason}");
            EditorApplication.ExitPlaymode();
        }

        private static bool TargetCompletedRunsReached()
        {
            int targetCompletedRuns = SessionState.GetInt(TargetCompletedRunsKey, 0);
            if (targetCompletedRuns <= 0)
            {
                return false;
            }

            int completedRuns = CountCompletedRunExportsSinceStart();
            int lastExportCount = SessionState.GetInt(LastExportCountKey, -1);
            if (completedRuns != lastExportCount)
            {
                SessionState.SetInt(LastExportCountKey, completedRuns);
                Debug.Log(
                    $"[DRT_SMOKE] Completed export progress. " +
                    $"label={SessionState.GetString(LabelKey, "-")}, count={completedRuns}/{targetCompletedRuns}");
            }

            return completedRuns >= targetCompletedRuns;
        }

        private static int CountCompletedRunExportsSinceStart()
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string exportDirectory = Path.Combine(projectRoot, "DRT_Episode_Exports");
            if (!Directory.Exists(exportDirectory))
            {
                return 0;
            }

            string policyToken = GetPolicyExportToken(SessionState.GetString(PolicyKey, string.Empty));
            string travelToken = GetTravelModeExportToken(SessionState.GetString(TravelModeKey, string.Empty));
            string pattern = $"drt_{travelToken}_{policyToken}_*_episode.csv";
            DateTime startTime = GetRunStartWallTime().AddSeconds(-1);
            int count = 0;

            foreach (string episodePath in Directory.GetFiles(exportDirectory, pattern, SearchOption.AllDirectories))
            {
                try
                {
                    if (File.GetLastWriteTime(episodePath) < startTime)
                    {
                        continue;
                    }

                    string text = File.ReadAllText(episodePath);
                    if (text.Contains("summary,completed_all_requests,1"))
                    {
                        count++;
                    }
                }
                catch (IOException)
                {
                    // The exporter may still be writing this CSV; count it on the next poll.
                }
            }

            return count;
        }

        private static DateTime GetRunStartWallTime()
        {
            string ticksText = SessionState.GetString(StartWallTimeTicksKey, string.Empty);
            if (long.TryParse(ticksText, out long ticks))
            {
                return new DateTime(ticks);
            }

            return DateTime.Now;
        }

        private static void WriteRunSummaryIfNeeded()
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string exportDirectory = Path.Combine(projectRoot, "DRT_Episode_Exports");
            if (!Directory.Exists(exportDirectory))
            {
                return;
            }

            string policyToken = GetPolicyExportToken(SessionState.GetString(PolicyKey, string.Empty));
            string travelToken = GetTravelModeExportToken(SessionState.GetString(TravelModeKey, string.Empty));
            string pattern = $"drt_{travelToken}_{policyToken}_*_episode.csv";
            DateTime startTime = GetRunStartWallTime().AddSeconds(-1);
            var records = new List<EpisodeCsvRecord>();

            foreach (string episodePath in Directory.GetFiles(exportDirectory, pattern, SearchOption.AllDirectories))
            {
                if (File.GetLastWriteTime(episodePath) < startTime)
                {
                    continue;
                }

                if (TryReadEpisodeCsv(episodePath, out EpisodeCsvRecord record))
                {
                    records.Add(record);
                }
            }

            if (records.Count == 0)
            {
                return;
            }

            string runDirectory = records
                .OrderByDescending(record => File.GetLastWriteTime(record.Path))
                .Select(record => Path.GetDirectoryName(record.Path))
                .FirstOrDefault(directory => !string.IsNullOrWhiteSpace(directory));
            if (string.IsNullOrWhiteSpace(runDirectory))
            {
                return;
            }

            var completedRecords = records
                .Where(record => record.IsCompletedAllRequests)
                .OrderBy(record => record.EpisodeIndex)
                .ToList();
            string summaryPath = Path.Combine(runDirectory, $"drt_{travelToken}_{policyToken}_run_summary.csv");
            File.WriteAllText(summaryPath, BuildRunSummaryCsv(records, completedRecords), Encoding.UTF8);
            Debug.Log(
                $"[DRT_SMOKE] Run summary exported. path={summaryPath}, " +
                $"included={completedRecords.Count}, excluded={records.Count - completedRecords.Count}");
        }

        private static bool TryReadEpisodeCsv(string path, out EpisodeCsvRecord record)
        {
            record = new EpisodeCsvRecord { Path = path };

            try
            {
                string[] lines = File.ReadAllLines(path, Encoding.UTF8);
                string[] header = Array.Empty<string>();

                foreach (string line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    List<string> cells = ParseCsvLine(line);
                    if (cells.Count == 0)
                    {
                        continue;
                    }

                    if (cells[0] == "section")
                    {
                        header = cells.ToArray();
                        continue;
                    }

                    if (header.Length >= 3 && header[0] == "section" && header[1] == "key" && cells.Count >= 3)
                    {
                        if (cells[0] == "summary")
                        {
                            record.Summary[cells[1]] = cells[2];
                        }

                        continue;
                    }

                    if (header.Length > 1 && header[0] == "section" && cells[0] == "route_leg")
                    {
                        var row = new Dictionary<string, string>();
                        for (int i = 0; i < header.Length; i++)
                        {
                            row[header[i]] = i < cells.Count ? cells[i] : string.Empty;
                        }

                        record.RouteLegs.Add(row);
                    }
                }

                record.EpisodeIndex = ReadInt(record.Summary, "episode_index", -1);
                return record.Summary.Count > 0;
            }
            catch (IOException)
            {
                return false;
            }
        }

        private static string BuildRunSummaryCsv(List<EpisodeCsvRecord> allRecords, List<EpisodeCsvRecord> completedRecords)
        {
            var builder = new StringBuilder();
            builder.AppendLine("section,key,value");
            AppendSummaryRow(builder, "metadata", "generated_at", DateTime.Now.ToString("o", CultureInfo.InvariantCulture));
            AppendSummaryRow(builder, "metadata", "source_episode_count", allRecords.Count.ToString(CultureInfo.InvariantCulture));
            AppendSummaryRow(builder, "metadata", "included_completed_episode_count", completedRecords.Count.ToString(CultureInfo.InvariantCulture));
            AppendSummaryRow(builder, "metadata", "excluded_incomplete_episode_count", (allRecords.Count - completedRecords.Count).ToString(CultureInfo.InvariantCulture));

            if (completedRecords.Count == 0)
            {
                AppendSummaryRow(builder, "averages", "status", "average unavailable");
                AppendSummaryRow(builder, "averages", "reason", "평균 낼 수 없음");
                return builder.ToString();
            }

            AppendAverage(builder, completedRecords, "episode_time_seconds");
            AppendAverage(builder, completedRecords, "episode_distance_meters");
            AppendAverage(builder, completedRecords, "service_rate");
            AppendAverage(builder, completedRecords, "completed_passengers");
            AppendAverage(builder, completedRecords, "total_passengers");
            AppendAverage(builder, completedRecords, "average_wait_seconds");
            AppendAverage(builder, completedRecords, "average_ride_seconds");
            AppendSummaryRow(builder, "averages", "mean_route_leg_count", FormatFloat((float)completedRecords.Average(record => record.RouteLegs.Count)));

            AppendRouteUsageSummary(builder, completedRecords);
            builder.AppendLine();
            builder.AppendLine("section,episode_index,episode_file,episode_time_seconds,episode_distance_meters,service_rate,completed_passengers,average_wait_seconds,average_ride_seconds,route_leg_count,route_sequence");

            foreach (EpisodeCsvRecord record in completedRecords)
            {
                builder
                    .Append("episode").Append(',')
                    .Append(record.EpisodeIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(CsvEscape(Path.GetFileName(record.Path))).Append(',')
                    .Append(CsvEscape(GetSummaryValue(record, "episode_time_seconds"))).Append(',')
                    .Append(CsvEscape(GetSummaryValue(record, "episode_distance_meters"))).Append(',')
                    .Append(CsvEscape(GetSummaryValue(record, "service_rate"))).Append(',')
                    .Append(CsvEscape(GetSummaryValue(record, "completed_passengers"))).Append(',')
                    .Append(CsvEscape(GetSummaryValue(record, "average_wait_seconds"))).Append(',')
                    .Append(CsvEscape(GetSummaryValue(record, "average_ride_seconds"))).Append(',')
                    .Append(record.RouteLegs.Count.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(CsvEscape(BuildRouteSequence(record)))
                    .AppendLine();
            }

            return builder.ToString();
        }

        private static void AppendRouteUsageSummary(StringBuilder builder, List<EpisodeCsvRecord> completedRecords)
        {
            var sequenceCounts = completedRecords
                .Select(BuildRouteSequence)
                .Where(sequence => !string.IsNullOrWhiteSpace(sequence))
                .GroupBy(sequence => sequence)
                .Select(group => new RouteUsage(group.Key, group.Count()))
                .OrderByDescending(item => item.Count)
                .ThenBy(item => item.Sequence)
                .ToList();

            AppendTopUsage(builder, "route_summary", "most_used_route_sequence", "route_sequence_count", "route_sequence_share", sequenceCounts, completedRecords.Count);

            var pairCounts = completedRecords
                .SelectMany(record => record.RouteLegs)
                .Select(BuildRoutePair)
                .Where(pair => !string.IsNullOrWhiteSpace(pair))
                .GroupBy(pair => pair)
                .Select(group => new RouteUsage(group.Key, group.Count()))
                .OrderByDescending(item => item.Count)
                .ThenBy(item => item.Sequence)
                .ToList();
            int totalPairCount = pairCounts.Sum(item => item.Count);
            AppendTopUsage(builder, "route_summary", "most_used_route_leg_pair", "route_leg_pair_count", "route_leg_pair_share", pairCounts, totalPairCount);
        }

        private static void AppendTopUsage(
            StringBuilder builder,
            string section,
            string valueKey,
            string countKey,
            string shareKey,
            List<RouteUsage> items,
            int denominator)
        {
            if (items.Count == 0 || denominator <= 0)
            {
                AppendSummaryRow(builder, section, valueKey, "평균 낼 수 없음");
                return;
            }

            int topCount = items[0].Count;
            int tiedCount = items.Count(item => item.Count == topCount);
            if (tiedCount > 1)
            {
                AppendSummaryRow(builder, section, valueKey, "평균 낼 수 없음");
                AppendSummaryRow(builder, section, countKey, topCount.ToString(CultureInfo.InvariantCulture));
                AppendSummaryRow(builder, section, "tie_count", tiedCount.ToString(CultureInfo.InvariantCulture));
                return;
            }

            AppendSummaryRow(builder, section, valueKey, items[0].Sequence);
            AppendSummaryRow(builder, section, countKey, topCount.ToString(CultureInfo.InvariantCulture));
            AppendSummaryRow(builder, section, shareKey, FormatFloat((float)topCount / denominator));
        }

        private static string BuildRouteSequence(EpisodeCsvRecord record)
        {
            return string.Join("|", record.RouteLegs.Select(BuildRoutePair).Where(pair => !string.IsNullOrWhiteSpace(pair)));
        }

        private static string BuildRoutePair(Dictionary<string, string> leg)
        {
            string from = GetValue(leg, "from_stop_id");
            string to = GetValue(leg, "to_stop_id");
            return string.IsNullOrWhiteSpace(from) || string.IsNullOrWhiteSpace(to) ? string.Empty : $"{from}->{to}";
        }

        private static void AppendAverage(StringBuilder builder, List<EpisodeCsvRecord> records, string key)
        {
            var values = records
                .Select(record => ReadFloat(record.Summary, key))
                .Where(value => value.HasValue)
                .Select(value => value.Value)
                .ToList();

            if (values.Count != records.Count || values.Count == 0)
            {
                AppendSummaryRow(builder, "averages", $"mean_{key}", "평균 낼 수 없음");
                return;
            }

            AppendSummaryRow(builder, "averages", $"mean_{key}", FormatFloat(values.Average()));
        }

        private static void AppendSummaryRow(StringBuilder builder, string section, string key, string value)
        {
            builder
                .Append(CsvEscape(section))
                .Append(',')
                .Append(CsvEscape(key))
                .Append(',')
                .Append(CsvEscape(value))
                .AppendLine();
        }

        private static List<string> ParseCsvLine(string line)
        {
            var result = new List<string>();
            var cell = new StringBuilder();
            bool inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                char current = line[i];
                if (inQuotes)
                {
                    if (current == '"' && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        cell.Append('"');
                        i++;
                    }
                    else if (current == '"')
                    {
                        inQuotes = false;
                    }
                    else
                    {
                        cell.Append(current);
                    }
                }
                else if (current == ',')
                {
                    result.Add(cell.ToString());
                    cell.Length = 0;
                }
                else if (current == '"')
                {
                    inQuotes = true;
                }
                else
                {
                    cell.Append(current);
                }
            }

            result.Add(cell.ToString());
            return result;
        }

        private static int ReadInt(Dictionary<string, string> values, string key, int fallback)
        {
            return int.TryParse(GetValue(values, key), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
                ? value
                : fallback;
        }

        private static float? ReadFloat(Dictionary<string, string> values, string key)
        {
            return float.TryParse(GetValue(values, key), NumberStyles.Float, CultureInfo.InvariantCulture, out float value)
                ? value
                : (float?)null;
        }

        private static string GetSummaryValue(EpisodeCsvRecord record, string key)
        {
            return GetValue(record.Summary, key);
        }

        private static string GetValue(Dictionary<string, string> values, string key)
        {
            return values.TryGetValue(key, out string value) ? value : string.Empty;
        }

        private static string FormatFloat(float value)
        {
            return value.ToString("0.###", CultureInfo.InvariantCulture);
        }

        private static string CsvEscape(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return string.Empty;
            }

            bool mustQuote = value.Contains(",") || value.Contains("\"") || value.Contains("\r") || value.Contains("\n");
            if (!mustQuote)
            {
                return value;
            }

            return $"\"{value.Replace("\"", "\"\"")}\"";
        }

        private sealed class EpisodeCsvRecord
        {
            public string Path;
            public int EpisodeIndex;
            public readonly Dictionary<string, string> Summary = new Dictionary<string, string>();
            public readonly List<Dictionary<string, string>> RouteLegs = new List<Dictionary<string, string>>();

            public bool IsCompletedAllRequests
            {
                get
                {
                    string value = GetValue(Summary, "completed_all_requests").Trim();
                    return value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase);
                }
            }
        }

        private sealed class RouteUsage
        {
            public readonly string Sequence;
            public readonly int Count;

            public RouteUsage(string sequence, int count)
            {
                Sequence = sequence;
                Count = count;
            }
        }

        private static string GetPolicyExportToken(string policyText)
        {
            if (Enum.TryParse(policyText, true, out DRTNextStopPolicy policy))
            {
                switch (policy)
                {
                    case DRTNextStopPolicy.ONNXInference:
                        return "inference";
                    case DRTNextStopPolicy.VanillaSequential:
                        return "vanilla";
                    case DRTNextStopPolicy.GreedyNearestFeasible:
                        return "greedy";
                    case DRTNextStopPolicy.Fifo:
                        return "fifo";
                    case DRTNextStopPolicy.AllStationRunner:
                        return "all_station";
                }
            }

            return "train";
        }

        private static string GetTravelModeExportToken(string travelModeText)
        {
            if (!Enum.TryParse(travelModeText, true, out DRTTravelExecutionMode travelMode))
            {
                return "matrix";
            }

            if (travelMode == DRTTravelExecutionMode.Train)
            {
                return "train";
            }

            if (travelMode == DRTTravelExecutionMode.PhysicalDrive)
            {
                string physicalDriveModeText = SessionState.GetString(PhysicalDriveModeKey, string.Empty);
                if (Enum.TryParse(physicalDriveModeText, true, out DRTPhysicalDriveMode physicalDriveMode))
                {
                    switch (physicalDriveMode)
                    {
                        case DRTPhysicalDriveMode.PPOAutonomous:
                            return "physical_ppo";
                        case DRTPhysicalDriveMode.PPOPurePursuit:
                            return "physical_ppo_purepursuit";
                        case DRTPhysicalDriveMode.NoisyGley:
                            return "physical_noisy_gley";
                    }
                }

                return "physical_gley";
            }

            return "matrix";
        }

        private static void FinishActiveEpisode(string reason)
        {
            var busController = UnityEngine.Object.FindObjectOfType<DRTBusController>();
            if (busController == null)
            {
                return;
            }

            MethodInfo finishEpisodeMethod = typeof(DRTBusController).GetMethod(
                "FinishEpisode",
                BindingFlags.Instance | BindingFlags.NonPublic);
            finishEpisodeMethod?.Invoke(busController, new object[] { reason });
        }

        private static void SetPrivateField(object target, string fieldName, object value)
        {
            FieldInfo field = GetPrivateField(target.GetType(), fieldName);
            field?.SetValue(target, value);
        }

        private static FieldInfo GetPrivateField(Type type, string fieldName)
        {
            return type.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        }
    }
}
#endif
