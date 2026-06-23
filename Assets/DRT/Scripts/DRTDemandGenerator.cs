using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace DRT
{
    public enum DRTDemandScenarioSelectionMode
    {
        FixedScenarioCsv,
        RandomPerEpisode
    }

    public class DRTDemandGenerator : MonoBehaviour
    {
        [HideInInspector, SerializeField] private DRTPassengerManager passengerManager;

        [Header("Scenario")]
        [HideInInspector, SerializeField] private bool generateOnStart;
        [HideInInspector, SerializeField] private bool clearExistingRequests = true;
        [SerializeField, InspectorName("Scenario Selection Mode")] private DRTDemandScenarioSelectionMode scenarioSelectionMode = DRTDemandScenarioSelectionMode.FixedScenarioCsv;
        [SerializeField, InspectorName("Fixed Scenario CSV")] private TextAsset scenarioCsvAsset;
        [SerializeField, InspectorName("Random Scenario CSVs")] private List<TextAsset> randomScenarioCsvAssets = new List<TextAsset>();
        [HideInInspector, SerializeField] private bool spawnRequestsAtRequestTime = true;

        [HideInInspector, SerializeField] private int stopCount = 8;
        [HideInInspector, SerializeField] private float episodeLengthSeconds = 3000f;

        [HideInInspector, SerializeField] private bool logSpawnedRequests;

        private readonly List<DRTDemandReplayEntry> replaySchedule = new List<DRTDemandReplayEntry>();
        private readonly Dictionary<TextAsset, CachedDemandScenario> scenarioCache = new Dictionary<TextAsset, CachedDemandScenario>();
        private readonly List<TextAsset> randomScenarioCandidates = new List<TextAsset>();
        private int nextReplayIndex;
        private int loadedPassengerCount;
        private string loadedScenarioDescription = "-";

        public bool HasGenerated { get; private set; }
        public int LoadedPassengerCount => loadedPassengerCount;
        public int SpawnedPassengerCount { get; private set; }
        public int PendingDemandCount => CountPendingPassengers();
        public bool HasPendingDemand => HasGenerated && spawnRequestsAtRequestTime && nextReplayIndex < replaySchedule.Count;
        public string LoadedScenarioDescription => loadedScenarioDescription;
        public string ExportScenarioId => loadedPassengerCount > 0
            ? loadedPassengerCount.ToString(CultureInfo.InvariantCulture)
            : "0";

        public void Configure(DRTPassengerManager newPassengerManager, int newStopCount)
        {
            Configure(newPassengerManager, newStopCount, episodeLengthSeconds);
        }

        public void Configure(DRTPassengerManager newPassengerManager, int newStopCount, float newEpisodeLengthSeconds)
        {
            passengerManager = newPassengerManager;
            stopCount = Mathf.Max(2, newStopCount);
            episodeLengthSeconds = Mathf.Max(1f, newEpisodeLengthSeconds);
        }

        public void ConfigureDiagnostics(bool newLogSpawnedRequests)
        {
            logSpawnedRequests = newLogSpawnedRequests;
        }

        private void Start()
        {
            if (generateOnStart && !HasGenerated)
            {
                GenerateDemand();
            }
        }

        [ContextMenu("Generate DRT Demand")]
        public void GenerateDemand()
        {
            GenerateDemand(false);
        }

        public void GenerateDemand(bool suppressLog)
        {
            if (!EnsurePassengerManager())
            {
                return;
            }

            if (clearExistingRequests)
            {
                passengerManager.ClearRequests();
            }

            replaySchedule.Clear();
            nextReplayIndex = 0;
            loadedPassengerCount = 0;
            SpawnedPassengerCount = 0;
            loadedScenarioDescription = "-";

            if (!TryBuildReplaySchedule(suppressLog, out List<DRTDemandReplayEntry> sourceSchedule, out string error))
            {
                HasGenerated = false;
                Debug.LogError($"[DEMANDGENERATOR] Cannot load demand scenario. {error}");
                return;
            }

            for (int i = 0; i < sourceSchedule.Count; i++)
            {
                var entry = sourceSchedule[i];
                if (!IsValidEntry(entry, suppressLog))
                {
                    continue;
                }

                replaySchedule.Add(entry);
                loadedPassengerCount += Mathf.Max(1, entry.passengerCount);
            }

            HasGenerated = true;
            if (spawnRequestsAtRequestTime)
            {
                SpawnDueRequests(0f, suppressLog);
            }
            else
            {
                SpawnAllRequests(suppressLog);
            }

            if (!suppressLog)
            {
                Debug.Log(
                    $"[DEMANDGENERATOR] Loaded scenario={loadedScenarioDescription}, " +
                    $"rows={replaySchedule.Count}, passengers={loadedPassengerCount}, " +
                    $"spawnByTime={spawnRequestsAtRequestTime}, active={passengerManager.Requests.Count}, " +
                    $"pending={PendingDemandCount}, first={FormatFirstScenarioEntry()}");
            }
        }

        public void ResetDemand()
        {
            ResetDemand(false);
        }

        public void ResetDemand(bool suppressLog)
        {
            HasGenerated = false;
            GenerateDemand(suppressLog);
        }

        public int SpawnDueRequests(float currentEpisodeTime)
        {
            return SpawnDueRequests(currentEpisodeTime, false);
        }

        public int SpawnDueRequests(float currentEpisodeTime, bool suppressLog)
        {
            if (!HasGenerated)
            {
                GenerateDemand(suppressLog);
            }

            if (!HasGenerated || !spawnRequestsAtRequestTime || !EnsurePassengerManager())
            {
                return 0;
            }

            int spawned = 0;
            float safeEpisodeTime = Mathf.Max(0f, currentEpisodeTime);
            while (nextReplayIndex < replaySchedule.Count &&
                   replaySchedule[nextReplayIndex].requestTimeSeconds <= safeEpisodeTime + 0.0001f)
            {
                spawned += SpawnEntry(replaySchedule[nextReplayIndex], true);
                nextReplayIndex++;
            }

            if (spawned > 0 && logSpawnedRequests && !suppressLog)
            {
                Debug.Log(
                    $"[DEMANDGENERATOR] Spawned due requests={spawned}, " +
                    $"time={safeEpisodeTime:0.0}s, active={passengerManager.Requests.Count}, pending={PendingDemandCount}");
            }

            return spawned;
        }

        private bool TryBuildReplaySchedule(bool suppressLog, out List<DRTDemandReplayEntry> result, out string error)
        {
            result = null;
            error = null;

            TextAsset selectedScenario = SelectScenarioCsvAsset(out error);
            if (selectedScenario == null)
            {
                return false;
            }

            loadedScenarioDescription = $"asset:{selectedScenario.name}";
            return TryGetCachedCsvSchedule(selectedScenario, loadedScenarioDescription, suppressLog, out result, out error);
        }

        private TextAsset SelectScenarioCsvAsset(out string error)
        {
            error = null;
            if (scenarioSelectionMode == DRTDemandScenarioSelectionMode.FixedScenarioCsv)
            {
                if (scenarioCsvAsset != null)
                {
                    return scenarioCsvAsset;
                }

                error = "Fixed Scenario CSV TextAsset is not assigned.";
                return null;
            }

            randomScenarioCandidates.Clear();
            if (randomScenarioCsvAssets != null)
            {
                for (int i = 0; i < randomScenarioCsvAssets.Count; i++)
                {
                    if (randomScenarioCsvAssets[i] != null)
                    {
                        randomScenarioCandidates.Add(randomScenarioCsvAssets[i]);
                    }
                }
            }

            if (randomScenarioCandidates.Count == 0)
            {
                error = "Random Scenario CSVs list has no assigned TextAssets.";
                return null;
            }

            return randomScenarioCandidates[UnityEngine.Random.Range(0, randomScenarioCandidates.Count)];
        }

        private bool TryGetCachedCsvSchedule(
            TextAsset scenario,
            string sourceDescription,
            bool suppressLog,
            out List<DRTDemandReplayEntry> result,
            out string error)
        {
            result = null;
            error = null;

            if (scenario == null)
            {
                error = "Scenario CSV TextAsset is not assigned.";
                return false;
            }

            if (!scenarioCache.TryGetValue(scenario, out CachedDemandScenario cachedScenario))
            {
                if (!TryParseCsvSchedule(
                        scenario.text,
                        sourceDescription,
                        suppressLog,
                        out List<DRTDemandReplayEntry> parsedSchedule,
                        out error))
                {
                    return false;
                }

                parsedSchedule.Sort(CompareReplayEntries);
                cachedScenario = new CachedDemandScenario
                {
                    sourceDescription = sourceDescription,
                    replaySchedule = parsedSchedule
                };
                scenarioCache[scenario] = cachedScenario;
            }

            loadedScenarioDescription = cachedScenario.sourceDescription;
            result = cachedScenario.replaySchedule;
            return true;
        }

        private bool TryParseCsvSchedule(
            string csvText,
            string sourceDescription,
            bool suppressLog,
            out List<DRTDemandReplayEntry> result,
            out string error)
        {
            result = new List<DRTDemandReplayEntry>();
            error = null;

            if (string.IsNullOrWhiteSpace(csvText))
            {
                error = "CSV text is empty.";
                return false;
            }

            string[] rawLines = csvText.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var lines = new List<string>();
            for (int i = 0; i < rawLines.Length; i++)
            {
                string line = rawLines[i].Trim();
                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                lines.Add(line);
            }

            if (lines.Count < 2)
            {
                error = "CSV must contain a header row and at least one passenger row.";
                return false;
            }

            string[] headers = SplitCsvLine(lines[0]);
            var columns = new Dictionary<string, int>();
            for (int i = 0; i < headers.Length; i++)
            {
                string normalizedHeader = NormalizeHeader(headers[i]);
                if (!string.IsNullOrEmpty(normalizedHeader))
                {
                    columns[normalizedHeader] = i;
                }
            }

            int idColumn = FindColumn(columns, "id", "passengerid", "passenger_id");
            int fromColumn = FindColumn(columns, "from", "fron", "origin", "originstopid", "origin_stop_id");
            int toColumn = FindColumn(columns, "to", "destination", "destinationstopid", "destination_stop_id");
            int reqColumn = FindColumn(columns, "req", "request", "requesttime", "requesttimeseconds", "request_time_seconds", "time");
            int statusColumn = FindColumn(columns, "status", "state");

            if (idColumn < 0 || fromColumn < 0 || toColumn < 0 || reqColumn < 0)
            {
                error = "CSV header must include id, from, to, and req columns. status is optional.";
                return false;
            }

            var seenPassengerIds = new HashSet<int>();
            for (int lineIndex = 1; lineIndex < lines.Count; lineIndex++)
            {
                string[] fields = SplitCsvLine(lines[lineIndex]);
                if (IsEmptyRow(fields))
                {
                    continue;
                }

                int rowNumber = lineIndex + 1;
                if (!TryReadInt(fields, idColumn, rowNumber, "id", out int passengerId, out error) ||
                    !TryReadInt(fields, fromColumn, rowNumber, "from", out int originStopId, out error) ||
                    !TryReadInt(fields, toColumn, rowNumber, "to", out int destinationStopId, out error) ||
                    !TryReadFloat(fields, reqColumn, rowNumber, "req", out float requestTimeSeconds, out error))
                {
                    return false;
                }

                if (passengerId > 0 && !seenPassengerIds.Add(passengerId))
                {
                    error = $"CSV row {rowNumber} duplicates passenger id {passengerId}.";
                    return false;
                }

                result.Add(new DRTDemandReplayEntry
                {
                    passengerId = passengerId,
                    requestTimeSeconds = Mathf.Max(0f, requestTimeSeconds),
                    originStopId = originStopId,
                    destinationStopId = destinationStopId,
                    passengerCount = 1,
                    initialStatus = ReadInitialStatus(fields, statusColumn, rowNumber, suppressLog),
                    sourceDescription = $"{sourceDescription}:row{rowNumber}"
                });
            }

            if (result.Count == 0)
            {
                error = "CSV contains no passenger rows.";
                return false;
            }

            return true;
        }

        private void SpawnAllRequests(bool suppressLog)
        {
            int spawned = 0;
            for (int i = 0; i < replaySchedule.Count; i++)
            {
                spawned += SpawnEntry(replaySchedule[i], false);
            }

            nextReplayIndex = replaySchedule.Count;

            if (spawned > 0 && logSpawnedRequests && !suppressLog)
            {
                Debug.Log($"[DEMANDGENERATOR] Spawned all scenario requests={spawned}.");
            }
        }

        private int SpawnEntry(DRTDemandReplayEntry entry, bool preserveInitialStatus)
        {
            if (entry == null || !EnsurePassengerManager())
            {
                return 0;
            }

            int count = Mathf.Max(1, entry.passengerCount);
            int spawned = 0;
            DRTPassengerStatus initialStatus = preserveInitialStatus
                ? entry.initialStatus
                : DRTPassengerStatus.Scheduled;

            for (int passenger = 0; passenger < count; passenger++)
            {
                int passengerId = entry.passengerId > 0 && count == 1 ? entry.passengerId : 0;
                var request = new DRTPassengerRequest(
                    passengerId,
                    entry.originStopId,
                    entry.destinationStopId,
                    entry.requestTimeSeconds,
                    initialStatus);

                passengerManager.AddRequest(request);
                spawned++;
                SpawnedPassengerCount++;
            }

            return spawned;
        }

        private bool IsValidEntry(DRTDemandReplayEntry entry, bool suppressLog)
        {
            if (entry == null)
            {
                return false;
            }

            if (entry.originStopId == entry.destinationStopId)
            {
                if (!suppressLog)
                {
                    Debug.LogWarning(
                        $"[DEMANDGENERATOR] Demand ignored ({entry.sourceDescription}). " +
                        $"Origin and destination are both Stop {entry.originStopId}.");
                }

                return false;
            }

            if (entry.originStopId < 1 || entry.destinationStopId < 1)
            {
                if (!suppressLog)
                {
                    Debug.LogWarning(
                        $"[DEMANDGENERATOR] Demand ignored ({entry.sourceDescription}). Stop IDs must start at 1.");
                }

                return false;
            }

            if (entry.passengerCount < 1)
            {
                return false;
            }

            return true;
        }

        private bool EnsurePassengerManager()
        {
            if (passengerManager == null)
            {
                passengerManager = FindObjectOfType<DRTPassengerManager>();
            }

            if (passengerManager != null)
            {
                return true;
            }

            Debug.LogError("[DRT] Cannot generate demand. PassengerManager is missing.");
            return false;
        }

        private int CountPendingPassengers()
        {
            if (!HasGenerated || !spawnRequestsAtRequestTime)
            {
                return 0;
            }

            int count = 0;
            for (int i = nextReplayIndex; i < replaySchedule.Count; i++)
            {
                count += Mathf.Max(1, replaySchedule[i].passengerCount);
            }

            return count;
        }

        private string FormatFirstScenarioEntry()
        {
            if (replaySchedule.Count == 0)
            {
                return "-";
            }

            var request = replaySchedule[0];
            string passengerId = request.passengerId > 0 ? request.passengerId.ToString(CultureInfo.InvariantCulture) : "auto";
            return $"#{passengerId}:{request.originStopId}->{request.destinationStopId}@{request.requestTimeSeconds:0}s";
        }

        private static DRTPassengerStatus ReadInitialStatus(string[] fields, int statusColumn, int rowNumber, bool suppressLog)
        {
            if (statusColumn < 0 || !TryGetField(fields, statusColumn, out string statusText) || string.IsNullOrWhiteSpace(statusText))
            {
                return DRTPassengerStatus.Scheduled;
            }

            if (!Enum.TryParse(statusText.Trim(), true, out DRTPassengerStatus parsedStatus))
            {
                if (!suppressLog)
                {
                    Debug.LogWarning($"[DEMANDGENERATOR] CSV row {rowNumber} has unknown status '{statusText}'. Spawning as Scheduled.");
                }

                return DRTPassengerStatus.Scheduled;
            }

            return NormalizeInitialStatus(parsedStatus);
        }

        private static DRTPassengerStatus NormalizeInitialStatus(DRTPassengerStatus status)
        {
            return status == DRTPassengerStatus.Waiting
                ? DRTPassengerStatus.Waiting
                : DRTPassengerStatus.Scheduled;
        }

        private static int CompareReplayEntries(DRTDemandReplayEntry a, DRTDemandReplayEntry b)
        {
            if (ReferenceEquals(a, b))
            {
                return 0;
            }

            if (a == null)
            {
                return 1;
            }

            if (b == null)
            {
                return -1;
            }

            int timeCompare = a.requestTimeSeconds.CompareTo(b.requestTimeSeconds);
            if (timeCompare != 0)
            {
                return timeCompare;
            }

            return a.passengerId.CompareTo(b.passengerId);
        }

        private static bool TryReadInt(
            string[] fields,
            int column,
            int rowNumber,
            string columnName,
            out int value,
            out string error)
        {
            value = 0;
            error = null;
            if (!TryGetField(fields, column, out string text) ||
                !int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
            {
                error = $"CSV row {rowNumber} column '{columnName}' is not a valid integer.";
                return false;
            }

            return true;
        }

        private static bool TryReadFloat(
            string[] fields,
            int column,
            int rowNumber,
            string columnName,
            out float value,
            out string error)
        {
            value = 0f;
            error = null;
            if (!TryGetField(fields, column, out string text) ||
                !float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            {
                error = $"CSV row {rowNumber} column '{columnName}' is not a valid number.";
                return false;
            }

            return true;
        }

        private static bool TryGetField(string[] fields, int column, out string value)
        {
            value = null;
            if (fields == null || column < 0 || column >= fields.Length)
            {
                return false;
            }

            value = fields[column].Trim();
            return true;
        }

        private static int FindColumn(Dictionary<string, int> columns, params string[] aliases)
        {
            for (int i = 0; i < aliases.Length; i++)
            {
                string normalizedAlias = NormalizeHeader(aliases[i]);
                if (columns.TryGetValue(normalizedAlias, out int column))
                {
                    return column;
                }
            }

            return -1;
        }

        private static string NormalizeHeader(string header)
        {
            if (string.IsNullOrWhiteSpace(header))
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            for (int i = 0; i < header.Length; i++)
            {
                char c = header[i];
                if (char.IsLetterOrDigit(c))
                {
                    builder.Append(char.ToLowerInvariant(c));
                }
            }

            return builder.ToString();
        }

        private static string[] SplitCsvLine(string line)
        {
            var fields = new List<string>();
            var builder = new StringBuilder();
            bool inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"')
                {
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        builder.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = !inQuotes;
                    }

                    continue;
                }

                if (c == ',' && !inQuotes)
                {
                    fields.Add(builder.ToString());
                    builder.Length = 0;
                    continue;
                }

                builder.Append(c);
            }

            fields.Add(builder.ToString());
            return fields.ToArray();
        }

        private static bool IsEmptyRow(string[] fields)
        {
            if (fields == null || fields.Length == 0)
            {
                return true;
            }

            for (int i = 0; i < fields.Length; i++)
            {
                if (!string.IsNullOrWhiteSpace(fields[i]))
                {
                    return false;
                }
            }

            return true;
        }

        private void OnValidate()
        {
            stopCount = Mathf.Max(2, stopCount);
            episodeLengthSeconds = Mathf.Max(1f, episodeLengthSeconds);
            scenarioCache.Clear();
            randomScenarioCandidates.Clear();
        }

        private sealed class CachedDemandScenario
        {
            public string sourceDescription;
            public List<DRTDemandReplayEntry> replaySchedule;
        }

        private sealed class DRTDemandReplayEntry
        {
            public int passengerId;
            public float requestTimeSeconds;
            public int originStopId;
            public int destinationStopId;
            public int passengerCount;
            public DRTPassengerStatus initialStatus;
            public string sourceDescription;
        }
    }
}
