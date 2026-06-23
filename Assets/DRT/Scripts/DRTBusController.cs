using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Gley.TrafficSystem;
using Gley.UrbanSystem;
using Unity.Barracuda;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.Serialization;

#if UNITY_EDITOR
using UnityEditor;
#endif

using VehicleTypes = Gley.TrafficSystem.User.VehicleTypes;

namespace DRT
{
    [AddComponentMenu("DRT/DRT Bus Controller")]
    public class DRTBusController : MonoBehaviour
    {
        private const string PPODriveAgentObjectName = "DRTDrivePPOAgent";
        private const string PPOPurePursuitAgentObjectName = "DRTPurePursuitPPOAgent";
        private const int MaxBackgroundTrafficDensity = 100;
        private const float MaxNoisyGleyLateralNoise = 3f;
        private const float MaxNoisyGleyNoiseStrength = 3f;

        [HideInInspector, SerializeField] private Transform busStopsRoot;
        [HideInInspector, SerializeField] private DRTPassengerManager passengerManager;
        [HideInInspector, SerializeField] private DRTDemandGenerator demandGenerator;
        [HideInInspector, SerializeField] private DRTNextStopSelector nextStopSelector;
        [HideInInspector, SerializeField] private Transform controlledPlayerVehicle;

        [Header("Vehicle")]
        [SerializeField, InspectorName("Vehicle")] private int vehicleIndex = 0;
        [SerializeField, InspectorName("Start Stop")] private int startStopId = 1;
        [SerializeField, InspectorName("Vehicle Type")] private VehicleTypes controlledVehicleType = VehicleTypes.Car;
        [SerializeField, InspectorName("Dwell (s)")] private float dwellSeconds = 5f;
        [SerializeField, InspectorName("Arrival Dist")] private float arrivalDistanceMeters = 3f;
        [HideInInspector, SerializeField] private float stopWaypointSnapDistanceMeters = 5f;
        [HideInInspector, SerializeField] private float arrivalWaitTimeoutSeconds = 12f;
        [SerializeField, InspectorName("Speed x")] private float controlledVehicleSpeedMultiplier = 1.5f;
        [Tooltip("Only used when Physical Drive uses the Gley vehicle driver. 1.0 keeps Gley speed unchanged; 1.12 means roughly +12%.")]
        [HideInInspector, SerializeField] private float gleyControlledVehicleSpeedMultiplier = 1.12f;
        [HideInInspector, SerializeField] private float playerWaypointReachDistanceMeters = 6f;
        [Header("Noisy Gley")]
        [SerializeField, Range(0f, MaxNoisyGleyLateralNoise), InspectorName("Steering Noise")] private float noisyGleyLateralNoise = 0.35f;
        [SerializeField, Range(0f, 0.5f), InspectorName("Speed Noise")] private float noisyGleySpeedNoise = 0.12f;
        [SerializeField, Range(0f, MaxNoisyGleyNoiseStrength), InspectorName("Noise Strength")] private float noisyGleyNoiseStrength = 1f;
        [SerializeField, Range(0.01f, 5f), InspectorName("Noise Frequency")] private float noisyGleyNoiseFrequency = 1f;
        [SerializeField, Range(0f, 1f), InspectorName("Noise Irregularity")] private float noisyGleyNoiseIrregularity = 0.65f;
        [Header("Vehicle Follow")]
        [SerializeField, InspectorName("Follow Bus")] private bool autoFollowControlledVehicle = true;
        [SerializeField, InspectorName("Physical Driver")] private DRTPhysicalDriveMode physicalDriveMode = DRTPhysicalDriveMode.Gley;
        [SerializeField, InspectorName("Gley Driver")] private bool useGleyVehicleControlInPhysicalDrive = true;
        [SerializeField, InspectorName("PPO Policy")] private DRTPPODrivePolicy ppoDrivePolicy = DRTPPODrivePolicy.MLAgentsTraining;
        [Tooltip("Used only when Physical Driver is PPO Autonomous and PPO Policy is ONNX Inference.")]
        [SerializeField, InspectorName("PPO ONNX Model")] private NNModel ppoOnnxInferenceModel;
        [HideInInspector, SerializeField] private InferenceDevice ppoOnnxInferenceDevice = InferenceDevice.Default;
        [SerializeField, InspectorName("Use PPO Speed Limit")] private bool usePPOSpeedLimit = true;
        [SerializeField, InspectorName("PPO Speed Limit (m/s)")] private float ppoSpeedLimitMetersPerSecond = 5f;

        [Header("PPO PurePursuit Parameters")]
        [SerializeField, InspectorName("Min Ld (m)")] private float ppoPurePursuitMinLookaheadMeters = 0.1f;
        [SerializeField, InspectorName("Max Ld (m)")] private float ppoPurePursuitMaxLookaheadMeters = 25f;
        [SerializeField, Range(0f, 1f), InspectorName("Zero Action Ld Norm")] private float ppoPurePursuitZeroActionLookaheadNormalized = 0f;
        [SerializeField, Range(0f, 1f), InspectorName("Min Speed Ratio")] private float ppoPurePursuitMinTargetSpeedRatio = 0f;
        [SerializeField, Range(0f, 1f), InspectorName("Max Speed Ratio")] private float ppoPurePursuitMaxTargetSpeedRatio = 1f;
        [SerializeField, Range(0f, 1f), InspectorName("Zero Action Speed Norm")] private float ppoPurePursuitZeroActionSpeedNormalized = 1f;
        [SerializeField, InspectorName("Throttle Smooth")] private float ppoPurePursuitThrottleInputSmoothing = 6f;
        [SerializeField, InspectorName("Steering Smooth")] private float ppoPurePursuitSteeringInputSmoothing = 12f;
        [SerializeField, Range(0.01f, 1f), InspectorName("Curvature Smooth Beta")] private float ppoPurePursuitCurvatureSmoothingBeta = 0.75f;
        [SerializeField, InspectorName("Speed Reward")] private float ppoPurePursuitSpeedRewardPerSecond = 0.005f;
        [SerializeField, InspectorName("Progress Reward")] private float ppoPurePursuitProgressRewardPerMeter = 0.1f;
        [SerializeField, InspectorName("Destination Progress Reward")] private float ppoPurePursuitDestinationProgressRewardPerMeter = 0f;
        [SerializeField, InspectorName("Destination Reward")] private float ppoPurePursuitDestinationReward = 10f;
        [SerializeField, InspectorName("Ld Change Penalty")] private float ppoPurePursuitLookaheadChangePenaltyPerMeter = 0f;
        [FormerlySerializedAs("ppoPurePursuitLateralErrorPenaltyPerSecond")]
        [SerializeField, InspectorName("Lateral Penalty / m")] private float ppoPurePursuitLateralErrorPenaltyPerMeter = 1.5f;
        [FormerlySerializedAs("ppoPurePursuitLocalLateralVelocityPenaltyPerSecond")]
        [SerializeField, InspectorName("Local Lat Vel Penalty / m")] private float ppoPurePursuitLocalLateralVelocityPenaltyPerMeter = 1f;
        [SerializeField, InspectorName("Local Lat Vel Kappa Gain")] private float ppoPurePursuitLocalLateralVelocityCurvatureGain = 3f;
        [FormerlySerializedAs("ppoPurePursuitHeadingErrorPenaltyPerSecond")]
        [SerializeField, InspectorName("Heading Penalty / m")] private float ppoPurePursuitHeadingErrorPenaltyPerMeter = 0.05f;
        [SerializeField, InspectorName("Overspeed Penalty")] private float ppoPurePursuitOverspeedPenaltyPerSecond = 0.3f;
        [SerializeField, InspectorName("CrossTrack Norm (m)")] private float ppoPurePursuitMaxCrossTrackErrorMeters = 6f;
        [SerializeField, InspectorName("CrossTrack End (m)")] private float ppoPurePursuitHardCrossTrackLimitMeters = 4f;
        [SerializeField, InspectorName("CrossTrack End Penalty")] private float ppoPurePursuitAssignedRouteExitPenaltyMagnitude = 5f;
        [SerializeField, InspectorName("No Move Timeout (s)")] private float ppoPurePursuitNoMovementTimeoutRealSeconds = 30f;
        [SerializeField, InspectorName("Min Move (m)")] private float ppoPurePursuitMinimumMovementMeters = 0.25f;

        [Header("Travel Execution")]
        [Tooltip("Controls how the selected next stop is reached. Matrix Teleport uses the travel-time matrix; Physical Drive uses the configured vehicle driver; Train follows waypoint paths kinematically for recording.")]
        [InspectorName("Mode")]
        [SerializeField] private DRTTravelExecutionMode travelExecutionMode = DRTTravelExecutionMode.MatrixTeleport;
        [SerializeField, InspectorName("Matrix Resource")] private string travelTimeMatrixResourceName = "drt_stop_travel_time_matrix";
        [Tooltip("Only used to estimate matrix-mode distance in exports. Matrix travel time is read directly from the CSV.")]
        [HideInInspector, SerializeField] private float matrixNominalSpeedMetersPerSecond = 15f;
        [Tooltip("Suppresses routine Unity logs while Matrix Teleport is active. Applies to training, ONNX inference, and vanilla policies.")]
        [HideInInspector, SerializeField] private bool suppressUnityLogsDuringMatrixTraining = true;

        [Header("Episode CSV Export")]
        [FormerlySerializedAs("exportInferenceCsvOnEpisodeEnd")]
        [SerializeField, InspectorName("Export CSV")] private bool exportEpisodeCsvOnEpisodeEnd = true;
        [HideInInspector, SerializeField] private float vehicleTraceSampleIntervalSeconds = 1f;

        [HideInInspector, SerializeField] private bool logMatrixTravel = true;
        [HideInInspector, SerializeField] private bool logEpisodeSummary = true;
        [HideInInspector, SerializeField] private bool logReward = true;
        [HideInInspector, SerializeField] private bool logDecision = true;
        [HideInInspector, SerializeField] private bool logPolicyAction = true;
        [HideInInspector, SerializeField] private bool logSpawnedRequests;
        [HideInInspector, SerializeField] private bool logStopProcessing = true;
        [HideInInspector, SerializeField] private bool logMovementDiagnostics;
        [HideInInspector, SerializeField] private float movementDiagnosticsIntervalSeconds = 2f;

        [Header("Path Visualization")]
        [SerializeField, InspectorName("Show Path")] private bool showAssignedPath = true;
        [HideInInspector, SerializeField] private bool showAssignedPathInGame = true;
        [HideInInspector, SerializeField] private bool showAssignedPathGizmos = true;
        [HideInInspector, SerializeField] private Color assignedPathColor = new Color(0f, 0.85f, 1f, 0.9f);
        [HideInInspector, SerializeField] private Color assignedPathWaypointColor = new Color(1f, 0.85f, 0f, 0.95f);
        [HideInInspector, SerializeField] private float assignedPathLineWidth = 0.7f;
        [HideInInspector, SerializeField] private float assignedPathWaypointRadius = 1.2f;
        [HideInInspector, SerializeField] private float assignedPathVerticalOffset = 0.35f;

        [Header("Background Traffic")]
        [SerializeField, InspectorName("Enable")] private bool backgroundTrafficEnabledOnStart = true;
        [SerializeField, Range(0, MaxBackgroundTrafficDensity), InspectorName("Density")] private int enabledTrafficDensity = 30;
        [SerializeField, InspectorName("Safe Gap (m)")] private float backgroundTrafficSafeGapMeters = 6f;
        [SerializeField, InspectorName("Min Trigger (m)")] private float backgroundTrafficMinTriggerLengthMeters = 10f;
        [SerializeField, InspectorName("Max Trigger (m)")] private float backgroundTrafficMaxTriggerLengthMeters = 30f;
        [SerializeField, InspectorName("Brake Time (s)")] private float backgroundTrafficBrakeTimeSeconds = 3f;
        [SerializeField, InspectorName("NPC Speed Limit (m/s)")] private float backgroundTrafficSpeedLimitMetersPerSecond = 8f;

        [Header("Episode")]
        [SerializeField, InspectorName("Length (s)")] private float episodeLengthSeconds = 3000f;
        [SerializeField, InspectorName("Sim Speed")] private float simulationSecondsPerRealSecond = 1f;
        [SerializeField, InspectorName("Stop When Done")] private bool stopWhenAllRequestsCompleted = true;

        [Header("PPO Training Route")]
        [SerializeField, InspectorName("Use Route Episode")] private bool usePPOTrainingRouteEpisode;
        [SerializeField, InspectorName("Use All-Station Route Cycle")] private bool usePPOTrainingAllStationRouteCycle;
        [SerializeField, InspectorName("Route Start Stop")] private int ppoTrainingRouteStartStopId = 1;
        [SerializeField, InspectorName("Route End Stop")] private int ppoTrainingRouteEndStopId = 4;

        [HideInInspector, SerializeField] private bool failEpisodeOnVehicleFault = true;
        [HideInInspector, SerializeField] private float failurePenalty = -1f;
        [HideInInspector, SerializeField] private float noMovementTimeoutRealSeconds = 30f;
        [Tooltip("A longer timeout used only when Gley reports a normal stop caused by traffic lights, give-way, or obstacles. Set 0 to never fail on traffic waits.")]
        [HideInInspector, SerializeField] private float trafficBlockTimeoutRealSeconds = 180f;
        [HideInInspector, SerializeField] private float minimumVehicleMovementMeters = 0.25f;
        [HideInInspector, SerializeField] private float maxRoadWaypointDistanceMeters = 30f;
        [HideInInspector, SerializeField] private float fallYThreshold = -10f;

        private readonly List<DRTStop> stops = new List<DRTStop>();
        private readonly DRTStopTravelTimeMatrix travelTimeMatrix = new DRTStopTravelTimeMatrix();
        private IDRTVehicleDriver vehicleDriver;
        private DRTPlayerVehicleDriver playerVehicleDriver;
        private DRTGleyVehicleDriver gleyVehicleDriver;
        private DRTNoisyGleyVehicleDriver noisyGleyVehicleDriver;
        private DRTNoisyPlayerVehicleDriver noisyPlayerVehicleDriver;
        private DRTPPOVehicleDriver ppoVehicleDriver;
        private DRTPPOPurePursuitVehicleDriver ppoPurePursuitVehicleDriver;
        private DRTTrainVehicleDriver trainVehicleDriver;
        private float episodeTimeSeconds;
        private bool initialized;
        private bool driving;
        private bool episodeFinished;
        private bool waitingForArrivalProximity;
        private bool cameraFollowApplied;
        private Transform cameraFollowTarget;
        private bool backgroundTrafficEnabled;
        private bool backgroundTrafficStateInitialized;
        private bool appliedBackgroundTrafficEnabled;
        private int appliedBackgroundTrafficDensity = -1;
        private float appliedBackgroundTrafficSafeGapMeters = -1f;
        private float appliedBackgroundTrafficMinTriggerLengthMeters = -1f;
        private float appliedBackgroundTrafficMaxTriggerLengthMeters = -1f;
        private float appliedBackgroundTrafficBrakeTimeSeconds = -1f;
        private float appliedBackgroundTrafficSpeedLimitMetersPerSecond = -1f;
        private int activeGleyVehicleIndex = -1;
        private int lastLoggedActiveGleyVehicleIndex = -1;
        private DRTPhysicalDriveMode lastLoggedActiveGleyMode;
        private int currentStopId;
        private int targetStopId;
        private int ppoTrainingAllStationRouteCycleIndex;
        private int activePPOTrainingRouteEpisodeIndex = -1;
        private int activePPOTrainingRouteStartStopId;
        private int activePPOTrainingRouteEndStopId;
        private Coroutine dwellRoutine;
        private Coroutine decisionRoutine;
        private readonly List<Vector3> assignedPathPoints = new List<Vector3>();
        private LineRenderer assignedPathLineRenderer;
        private float assignedPathDistanceMeters;
        private readonly List<DRTRouteLegRecord> routeLegRecords = new List<DRTRouteLegRecord>();
        private readonly List<DRTVehicleTraceRecord> vehicleTraceRecords = new List<DRTVehicleTraceRecord>();
        private DRTRouteLegRecord activeRouteLeg;
        private bool episodeCsvExported;
        private float nextMovementDiagnosticTime;
        private Vector3 lastVehicleMovementPosition;
        private float lastVehicleMovementRealtime;
        private bool hasVehicleMovementSample;
        private float trafficBlockStartRealtime;
        private bool hasTrafficBlockSample;
        private string trafficBlockReason;
        private float episodeTravelDistanceMeters;
        private Vector3 lastTravelDistanceSamplePosition;
        private bool hasTravelDistanceSample;
        private bool travelTimeMatrixLoadAttempted;
        private int episodeIndex;
        private DateTime episodeExportRunTimestamp;
        private bool hasEpisodeExportRunTimestamp;
        private bool allStationMatrixCsvExported;

        public float EpisodeTimeSeconds => episodeTimeSeconds;
        public bool IsInitialized => initialized;
        public bool IsDriving => driving;
        public bool IsEpisodeFinished => episodeFinished;
        public bool IsWaitingForArrivalProximity => waitingForArrivalProximity;
        public int CurrentStopId => currentStopId;
        public int TargetStopId => targetStopId;
        public int VehicleIndex => UsesGleyVehicleControl ? GetActiveGleyVehicleIndex() : vehicleIndex;
        public string ControlledVehicleName => vehicleDriver != null ? vehicleDriver.VehicleName : controlledPlayerVehicle != null ? controlledPlayerVehicle.name : "-";
        public string ControlledDriverName => vehicleDriver != null ? vehicleDriver.GetType().Name : GetConfiguredDriverName();
        public Transform ControlledVehicleTransform => GetControlledVehicleTransform();
        public Vector3 ControlledVehicleBodyPosition => GetControlledVehicleBodyPosition();
        public int AssignedPathPointCount => assignedPathPoints.Count;
        public float AssignedPathDistanceMeters => assignedPathDistanceMeters;
        public bool IsVehicleTemporarilyBlocked => vehicleDriver != null && vehicleDriver.IsTemporarilyBlocked;
        public string TemporaryBlockReason => vehicleDriver != null ? vehicleDriver.TemporaryBlockReason : string.Empty;
        public float ArrivalDistanceMeters => arrivalDistanceMeters;
        public float VehicleSpeedMS => GetVehicleSpeedMS();
        public float TargetDistanceMeters => GetTargetDistanceMeters();
        public float EpisodeLengthSeconds => episodeLengthSeconds;
        public float SimulationSecondsPerRealSecond => simulationSecondsPerRealSecond;
        public float EpisodeTravelDistanceMeters => episodeTravelDistanceMeters;
        public string TargetStopObjectName => TryGetStop(targetStopId, out DRTStop stop) ? stop.name : "-";
        public bool BackgroundTrafficEnabled => backgroundTrafficEnabled;
        public int BackgroundTrafficDensity => enabledTrafficDensity;
        public int ActiveBackgroundVehicleCount => CountActiveBackgroundTrafficVehicles();
        public DRTTravelExecutionMode TravelExecutionMode => travelExecutionMode;
        public string TravelExecutionModeName => travelExecutionMode.ToString();
        public DRTPhysicalDriveMode PhysicalDriveMode => physicalDriveMode;
        public string PhysicalDriveModeName => physicalDriveMode.ToString();
        public DRTPPODrivePolicy PPODrivePolicy => ppoDrivePolicy;
        public string PPODrivePolicyName => ppoDrivePolicy.ToString();
        public bool UsePPOSpeedLimit => usePPOSpeedLimit;
        public float PPOSpeedLimitMetersPerSecond => ppoSpeedLimitMetersPerSecond;
        public DRTNextStopPolicy NextStopPolicy => nextStopSelector != null ? nextStopSelector.NextStopPolicy : DRTNextStopPolicy.MLAgentsTraining;
        public string NextStopPolicyName => nextStopSelector != null ? nextStopSelector.NextStopPolicyName : "-";
        public bool UsesMatrixTeleport => travelExecutionMode == DRTTravelExecutionMode.MatrixTeleport;
        public bool UsesTrainDrive => travelExecutionMode == DRTTravelExecutionMode.Train;
        public bool UsesAllStationRunner => nextStopSelector != null && nextStopSelector.UsesAllStationRunner;
        public bool UsesGleyVehicleControl => travelExecutionMode == DRTTravelExecutionMode.PhysicalDrive &&
                                              (physicalDriveMode == DRTPhysicalDriveMode.Gley ||
                                               (physicalDriveMode == DRTPhysicalDriveMode.NoisyGley &&
                                                vehicleDriver is DRTNoisyGleyVehicleDriver));
        public bool UsesPPOVehicleControl => travelExecutionMode == DRTTravelExecutionMode.PhysicalDrive &&
                                             (physicalDriveMode == DRTPhysicalDriveMode.PPOAutonomous ||
                                              physicalDriveMode == DRTPhysicalDriveMode.PPOPurePursuit);
        public bool UsesPPOPurePursuitVehicleControl => travelExecutionMode == DRTTravelExecutionMode.PhysicalDrive &&
                                                        physicalDriveMode == DRTPhysicalDriveMode.PPOPurePursuit;
        private bool IsGleyBasedPhysicalDriveMode => physicalDriveMode == DRTPhysicalDriveMode.Gley ||
                                                     physicalDriveMode == DRTPhysicalDriveMode.NoisyGley;
        public bool UsesPPOVehicleTraining => UsesPPOVehicleControl &&
                                              ppoDrivePolicy == DRTPPODrivePolicy.MLAgentsTraining;
        public bool UsesPPOTrainingRouteEpisode => IsPPOTrainingRouteEpisodeActive();
        public bool UsesPPOTrainingAllStationRouteCycle => usePPOTrainingAllStationRouteCycle;
        public int PPOTrainingRouteStartStopId => GetActivePPOTrainingRouteStartStopId();
        public int PPOTrainingRouteEndStopId => GetActivePPOTrainingRouteEndStopId();
        public string PPOTrainingRouteName => UsesPPOTrainingRouteEpisode
            ? $"{(usePPOTrainingAllStationRouteCycle ? "AllStationCycle " : string.Empty)}Stop {PPOTrainingRouteStartStopId} -> Stop {PPOTrainingRouteEndStopId}"
            : "off";
        public bool SuppressUnityLogsDuringMatrixTeleport => UsesMatrixTeleport &&
                                                             suppressUnityLogsDuringMatrixTraining;
        public bool SuppressUnityLogsDuringMatrixTraining => SuppressUnityLogsDuringMatrixTeleport;
        public IReadOnlyList<DRTStop> Stops => stops;

        public void Configure(
            Transform newBusStopsRoot,
            DRTPassengerManager newPassengerManager,
            DRTDemandGenerator newDemandGenerator,
            DRTNextStopSelector newNextStopSelector,
            int newVehicleIndex = 0,
            int newStartStopId = 1)
        {
            busStopsRoot = newBusStopsRoot;
            passengerManager = newPassengerManager;
            demandGenerator = newDemandGenerator;
            nextStopSelector = newNextStopSelector;
            vehicleIndex = Mathf.Max(0, newVehicleIndex);
            startStopId = Mathf.Max(1, newStartStopId);
            LoadStops();
            WirePassengerManager();
            WireDemandGenerator();
            WireNextStopSelector();
            ResolveControlledVehicle(false);
        }

        [ContextMenu("Run All Station Travel Time Calibration")]
        public void RunAllStationTravelTimeCalibration()
        {
            Debug.LogWarning("[BUSCONTROLLER] Select Next Stop Policy = All Station Runner, Travel Execution Mode = Physical Drive, then run Play Mode.");
        }

        private void Awake()
        {
            EnsureEpisodeExportRunTimestamp();
            ResolveReferences();
            LoadStops(false);
            WirePassengerManager();
            WireDemandGenerator();
            WireNextStopSelector();
            ResolveControlledVehicle(false);
        }

        private void Start()
        {
            if (demandGenerator != null && !demandGenerator.HasGenerated)
            {
                demandGenerator.GenerateDemand();
            }
        }

        public void ResetEpisodeFromAgent()
        {
            ResolveReferences();
            LoadStops(false);
            WireNextStopSelector();
            StopEpisodeCoroutines();

            episodeIndex++;
            PreparePPOTrainingRouteSelectionForNewEpisode();
            episodeTimeSeconds = 0f;
            initialized = false;
            driving = false;
            episodeFinished = false;
            waitingForArrivalProximity = false;
            currentStopId = ResolveEpisodeStartStopId();
            targetStopId = 0;
            ClearAssignedPathVisualization();
            ResetEpisodeExportState();
            nextMovementDiagnosticTime = 0f;
            hasVehicleMovementSample = false;
            hasTrafficBlockSample = false;
            trafficBlockReason = string.Empty;
            episodeTravelDistanceMeters = 0f;
            hasTravelDistanceSample = false;
            travelTimeMatrixLoadAttempted = false;
            lastVehicleMovementRealtime = Time.realtimeSinceStartup;
            lastVehicleMovementPosition = Vector3.zero;

            WirePassengerManager();
            WireDemandGenerator();

            demandGenerator?.ResetDemand(SuppressUnityLogsDuringMatrixTeleport);

            if (API.IsInitialized())
            {
                EnsureBackgroundTrafficStateInitialized();
                ApplyBackgroundTrafficState();
                ResetControlledVehicleForEpisode();
            }

            int requestCount = passengerManager != null ? passengerManager.Requests.Count : 0;
            LogInfo($"[BUSCONTROLLER] Episode reset. index={episodeIndex}, requests={requestCount}, startStop={currentStopId}");
        }

        private void PreparePPOTrainingRouteSelectionForNewEpisode()
        {
            if (!usePPOTrainingAllStationRouteCycle)
            {
                return;
            }

            activePPOTrainingRouteEpisodeIndex = -1;
            activePPOTrainingRouteStartStopId = 0;
            activePPOTrainingRouteEndStopId = 0;
        }

        private void Update()
        {
            if (episodeFinished)
            {
                return;
            }

            if (!initialized)
            {
                TryInitializeDriving();
                return;
            }

            if (!UsesMatrixTeleport)
            {
                episodeTimeSeconds += Time.deltaTime * simulationSecondsPerRealSecond;
            }

            AdvanceDemandToCurrentTime();
            passengerManager?.UpdateRequestStates(episodeTimeSeconds);
            TrackPhysicalTravelDistanceIfNeeded();
            RecordVehicleTraceIfNeeded();
            LogMovementDiagnosticsIfNeeded();
            SyncBackgroundTrafficInspectorStateIfNeeded();
            RefreshCameraFollowTargetIfNeeded();
            if (!UsesMatrixTeleport)
            {
                MonitorVehicleFailureIfNeeded();
            }

            if (episodeFinished)
            {
                return;
            }

            if (!UsesAllStationRunner && episodeTimeSeconds >= episodeLengthSeconds)
            {
                FinishEpisode("Episode time ended.");
            }
        }

        private void OnDisable()
        {
            vehicleDriver?.ReleaseControl();
            ClearAssignedPathVisualization();
        }

        [ContextMenu("Reload Stops")]
        public void LoadStops()
        {
            LoadStops(true);
        }

        private void LoadStops(bool logIfMissing)
        {
            stops.Clear();

            if (busStopsRoot == null)
            {
                if (logIfMissing)
                {
                    Debug.LogError("[DRT] BusStops root is missing.");
                }
                return;
            }

            for (int i = 0; i < busStopsRoot.childCount; i++)
            {
                Transform child = busStopsRoot.GetChild(i);
                var stop = child.GetComponent<DRTStop>();

                if (TryParseStopIdFromName(child.name, out int parsedStopId))
                {
                    if (stop == null)
                    {
                        stop = child.gameObject.AddComponent<DRTStop>();
                    }

                    stop.SetStopId(parsedStopId);
                }
                else if (stop == null)
                {
                    continue;
                }
                else if (stop.StopId < 1)
                {
                    stop.SetStopId(i + 1);
                }

                stops.Add(stop);
            }

            stops.Sort((a, b) => a.StopId.CompareTo(b.StopId));

            var duplicateIds = stops
                .GroupBy(stop => stop.StopId)
                .Where(group => group.Count() > 1)
                .Select(group => $"{group.Key} ({string.Join(", ", group.Select(stop => stop.name))})")
                .ToList();

            if (duplicateIds.Count > 0)
            {
                Debug.LogWarning($"[DRT] Duplicate stop IDs detected: {string.Join(", ", duplicateIds)}");
            }

            if (logIfMissing)
            {
                Debug.Log($"[BUSCONTROLLER] Loaded stops count={stops.Count}, root={busStopsRoot.name}");
                LogStopMap();
            }
        }

        private void ResolveReferences()
        {
            if (passengerManager == null)
            {
                passengerManager = FindObjectOfType<DRTPassengerManager>();
            }

            if (demandGenerator == null)
            {
                demandGenerator = FindObjectOfType<DRTDemandGenerator>();
            }

            if (nextStopSelector == null)
            {
                nextStopSelector = FindObjectOfType<DRTNextStopSelector>();
            }
        }

        private void WireNextStopSelector()
        {
            if (nextStopSelector != null)
            {
                nextStopSelector.Configure(this);
                nextStopSelector.ConfigureDiagnostics(logReward, logDecision, logPolicyAction);
            }
        }

        private int ResolveEpisodeStartStopId()
        {
            if (TryGetPPOTrainingRouteStartStop(out int configuredRouteStartStopId))
            {
                return configuredRouteStartStopId;
            }

            int resolvedStartStopId = startStopId;
            if (!UsesAllStationRunner || nextStopSelector == null)
            {
                return resolvedStartStopId;
            }

            HashSet<string> completedPairKeys = LoadCompletedAllStationPairKeysFromSampleExports();
            nextStopSelector.ConfigureAllStationResume(completedPairKeys, startStopId);

            if (nextStopSelector.TryGetAllStationResumeStart(
                    stops,
                    startStopId,
                    out int resumeStartStopId,
                    out int resumeTargetStopId,
                    out int skippedEdgeCount,
                    out int totalEdgeCount))
            {
                resolvedStartStopId = resumeStartStopId;
                LogInfo(
                    $"[BUSCONTROLLER] AllStation resume enabled. " +
                    $"completedPairs={completedPairKeys.Count}, skippedPrefix={skippedEdgeCount}/{totalEdgeCount}, " +
                    $"start={resumeStartStopId}, next={resumeTargetStopId}");
            }
            else
            {
                LogInfo(
                    $"[BUSCONTROLLER] AllStation resume found no remaining route edge. " +
                    $"completedPairs={completedPairKeys.Count}, start={resumeStartStopId}, skippedPrefix={skippedEdgeCount}/{totalEdgeCount}");
                resolvedStartStopId = resumeStartStopId > 0 ? resumeStartStopId : startStopId;
            }

            return resolvedStartStopId;
        }

        private HashSet<string> LoadCompletedAllStationPairKeysFromSampleExports()
        {
            var completedPairKeys = new HashSet<string>();
            string exportRoot = GetAllStationExportRootPath();
            if (string.IsNullOrWhiteSpace(exportRoot) || !Directory.Exists(exportRoot))
            {
                return completedPairKeys;
            }

            string[] sampleFiles;
            try
            {
                sampleFiles = Directory.GetFiles(exportRoot, "*_matrix_samples.csv", SearchOption.AllDirectories);
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[BUSCONTROLLER] Failed to scan AllStation matrix sample exports. path={exportRoot}, error={exception.Message}");
                return completedPairKeys;
            }

            for (int i = 0; i < sampleFiles.Length; i++)
            {
                AddCompletedPairsFromSampleCsv(sampleFiles[i], completedPairKeys);
            }

            return completedPairKeys;
        }

        private void AddCompletedPairsFromSampleCsv(string sampleCsvPath, HashSet<string> completedPairKeys)
        {
            if (string.IsNullOrWhiteSpace(sampleCsvPath) || completedPairKeys == null)
            {
                return;
            }

            string[] lines;
            try
            {
                lines = File.ReadAllLines(sampleCsvPath);
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[BUSCONTROLLER] Failed to read AllStation matrix sample CSV. path={sampleCsvPath}, error={exception.Message}");
                return;
            }

            for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++)
            {
                string line = lines[lineIndex];
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                string[] columns = line.Split(',');
                if (columns.Length < 3 ||
                    !int.TryParse(columns[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int fromStopId) ||
                    !int.TryParse(columns[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int toStopId) ||
                    !int.TryParse(columns[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int sampleCount) ||
                    sampleCount <= 0)
                {
                    continue;
                }

                completedPairKeys.Add(BuildStopPairKey(fromStopId, toStopId));
            }
        }

        private void LoadHistoricalAllStationSampleData(
            string excludedSampleCsvPath,
            Dictionary<string, List<float>> samplesByPair)
        {
            if (samplesByPair == null)
            {
                return;
            }

            string exportRoot = GetAllStationExportRootPath();
            if (string.IsNullOrWhiteSpace(exportRoot) || !Directory.Exists(exportRoot))
            {
                return;
            }

            string[] sampleFiles;
            try
            {
                sampleFiles = Directory.GetFiles(exportRoot, "*_matrix_samples.csv", SearchOption.AllDirectories)
                    .OrderBy(path => File.GetLastWriteTimeUtc(path))
                    .ToArray();
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[BUSCONTROLLER] Failed to scan AllStation matrix sample exports. path={exportRoot}, error={exception.Message}");
                return;
            }

            for (int i = 0; i < sampleFiles.Length; i++)
            {
                if (IsSamePath(sampleFiles[i], excludedSampleCsvPath))
                {
                    continue;
                }

                AddHistoricalAllStationSampleDataFromCsv(sampleFiles[i], samplesByPair);
            }
        }

        private void AddHistoricalAllStationSampleDataFromCsv(
            string sampleCsvPath,
            Dictionary<string, List<float>> samplesByPair)
        {
            if (string.IsNullOrWhiteSpace(sampleCsvPath) ||
                samplesByPair == null)
            {
                return;
            }

            string[] lines;
            try
            {
                lines = File.ReadAllLines(sampleCsvPath);
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[BUSCONTROLLER] Failed to read AllStation matrix sample CSV. path={sampleCsvPath}, error={exception.Message}");
                return;
            }

            for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++)
            {
                string line = lines[lineIndex];
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                string[] columns = line.Split(',');
                if (columns.Length < 4 ||
                    !int.TryParse(columns[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int fromStopId) ||
                    !int.TryParse(columns[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int toStopId) ||
                    !int.TryParse(columns[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int sampleCount))
                {
                    continue;
                }

                string key = BuildStopPairKey(fromStopId, toStopId);
                float.TryParse(columns[3].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float matrixSeconds);

                if (sampleCount <= 0)
                {
                    continue;
                }

                bool addedSample = false;
                if (columns.Length >= 5)
                {
                    string sampleText = columns[4].Trim().Trim('"');
                    if (!string.IsNullOrWhiteSpace(sampleText))
                    {
                        string[] sampleValues = sampleText.Split('|');
                        for (int sampleIndex = 0; sampleIndex < sampleValues.Length; sampleIndex++)
                        {
                            if (float.TryParse(sampleValues[sampleIndex].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float sampleSeconds) &&
                                sampleSeconds > 0f)
                            {
                                AddSampleValue(samplesByPair, key, sampleSeconds);
                                addedSample = true;
                            }
                        }
                    }
                }

                if (!addedSample && matrixSeconds > 0f)
                {
                    AddSampleValue(samplesByPair, key, matrixSeconds);
                }
            }
        }

        private static void AddSampleValue(Dictionary<string, List<float>> samplesByPair, string key, float seconds)
        {
            if (samplesByPair == null || string.IsNullOrWhiteSpace(key) || seconds <= 0f)
            {
                return;
            }

            if (!samplesByPair.TryGetValue(key, out List<float> samples))
            {
                samples = new List<float>();
                samplesByPair[key] = samples;
            }

            samples.Add(seconds);
        }

        private static bool IsSamePath(string left, string right)
        {
            if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
            {
                return false;
            }

            try
            {
                return string.Equals(
                    System.IO.Path.GetFullPath(left),
                    System.IO.Path.GetFullPath(right),
                    StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return string.Equals(left, right, StringComparison.OrdinalIgnoreCase);
            }
        }

        private static string GetAllStationExportRootPath()
        {
            string projectRoot = System.IO.Path.GetDirectoryName(Application.dataPath);
            return string.IsNullOrWhiteSpace(projectRoot)
                ? null
                : System.IO.Path.Combine(projectRoot, "DRT_Episode_Exports", "all_station");
        }

        private void WireDemandGenerator()
        {
            if (demandGenerator == null)
            {
                return;
            }

            int configuredStopCount = stops.Count;
            if (configuredStopCount <= 0 && busStopsRoot != null)
            {
                configuredStopCount = busStopsRoot.childCount;
            }

            demandGenerator.Configure(
                passengerManager,
                Mathf.Max(2, configuredStopCount),
                episodeLengthSeconds);
            demandGenerator.ConfigureDiagnostics(logSpawnedRequests);
        }

        private void WirePassengerManager()
        {
            passengerManager?.ConfigureDiagnostics(logStopProcessing);
        }

        private bool ResolveControlledVehicle(bool logIfMissing)
        {
            if (travelExecutionMode == DRTTravelExecutionMode.Train)
            {
                return ResolveControlledTrainVehicle(logIfMissing);
            }

            if (travelExecutionMode == DRTTravelExecutionMode.PhysicalDrive)
            {
                switch (physicalDriveMode)
                {
                    case DRTPhysicalDriveMode.PPOAutonomous:
                        return ResolveControlledPPOVehicle(logIfMissing);
                    case DRTPhysicalDriveMode.PPOPurePursuit:
                        return ResolveControlledPPOPurePursuitVehicle(logIfMissing);
                    case DRTPhysicalDriveMode.NoisyGley:
                        return ResolveControlledNoisyGleyVehicle(logIfMissing);
                    default:
                        return ResolveControlledGleyVehicle(logIfMissing);
                }
            }

            return ResolveControlledPlayerVehicle(logIfMissing);
        }

        private string GetConfiguredDriverName()
        {
            if (travelExecutionMode == DRTTravelExecutionMode.Train)
            {
                return nameof(DRTTrainVehicleDriver);
            }

            if (travelExecutionMode != DRTTravelExecutionMode.PhysicalDrive)
            {
                return nameof(DRTPlayerVehicleDriver);
            }

            switch (physicalDriveMode)
            {
                case DRTPhysicalDriveMode.PPOAutonomous:
                    return nameof(DRTPPOVehicleDriver);
                case DRTPhysicalDriveMode.PPOPurePursuit:
                    return nameof(DRTPPOPurePursuitVehicleDriver);
                case DRTPhysicalDriveMode.NoisyGley:
                    return nameof(DRTNoisyGleyVehicleDriver);
                default:
                    return nameof(DRTGleyVehicleDriver);
            }
        }

        private bool ResolveControlledGleyVehicle(bool logIfMissing)
        {
            DisableNoisyGleyVehicleDriver();
            DisableNoisyPlayerVehicleDriver();
            DisablePPOVehicleDriver();
            DisablePPOPurePursuitVehicleDriver();
            DisableTrainVehicleDriver();
            int effectiveVehicleIndex = ResolveGleyVehicleIndex(logIfMissing);

            if (gleyVehicleDriver == null)
            {
                gleyVehicleDriver = GetComponent<DRTGleyVehicleDriver>();
                if (gleyVehicleDriver == null)
                {
                    gleyVehicleDriver = gameObject.AddComponent<DRTGleyVehicleDriver>();
                }
            }

            gleyVehicleDriver.Configure(
                effectiveVehicleIndex,
                controlledVehicleType,
                gleyControlledVehicleSpeedMultiplier,
                usePPOSpeedLimit,
                ppoSpeedLimitMetersPerSecond);
            if (gleyVehicleDriver.VehicleTransform == null)
            {
                if (logIfMissing)
                {
                    Debug.LogWarning($"[BUSCONTROLLER] Gley controlled vehicle not found. vehicleIndex={effectiveVehicleIndex}");
                }

                return false;
            }

            activeGleyVehicleIndex = effectiveVehicleIndex;
            vehicleDriver = gleyVehicleDriver;
            controlledPlayerVehicle = gleyVehicleDriver.VehicleTransform;
            LogActiveGleyVehicleIfChanged();
            return true;
        }

        private bool ResolveControlledNoisyGleyVehicle(bool logIfMissing)
        {
            DisablePPOVehicleDriver();
            DisablePPOPurePursuitVehicleDriver();
            DisableTrainVehicleDriver();
            int effectiveVehicleIndex = ResolveGleyVehicleIndex(logIfMissing);

            DisableNoisyPlayerVehicleDriver();

            if (noisyGleyVehicleDriver == null)
            {
                noisyGleyVehicleDriver = GetComponent<DRTNoisyGleyVehicleDriver>();
                if (noisyGleyVehicleDriver == null)
                {
                    noisyGleyVehicleDriver = gameObject.AddComponent<DRTNoisyGleyVehicleDriver>();
                }
            }

            noisyGleyVehicleDriver.Configure(
                effectiveVehicleIndex,
                controlledVehicleType,
                gleyControlledVehicleSpeedMultiplier,
                noisyGleyLateralNoise,
                noisyGleySpeedNoise,
                noisyGleyNoiseFrequency,
                noisyGleyNoiseStrength,
                noisyGleyNoiseIrregularity,
                usePPOSpeedLimit,
                ppoSpeedLimitMetersPerSecond);

            if (noisyGleyVehicleDriver.VehicleTransform == null)
            {
                if (logIfMissing)
                {
                    Debug.LogWarning($"[BUSCONTROLLER] Noisy Gley controlled vehicle not found. vehicleIndex={effectiveVehicleIndex}");
                }

                return false;
            }

            activeGleyVehicleIndex = effectiveVehicleIndex;
            vehicleDriver = noisyGleyVehicleDriver;
            controlledPlayerVehicle = noisyGleyVehicleDriver.VehicleTransform;
            LogActiveGleyVehicleIfChanged();
            return true;
        }

        private bool ResolveControlledNoisyPlayerVehicle(bool logIfMissing)
        {
            activeGleyVehicleIndex = -1;
            DisableNoisyGleyVehicleDriver();
            gleyVehicleDriver?.ReleaseControl();

            if (!ResolveControlledPlayerTransform(logIfMissing))
            {
                return false;
            }

            if (noisyPlayerVehicleDriver == null ||
                noisyPlayerVehicleDriver.transform != controlledPlayerVehicle)
            {
                noisyPlayerVehicleDriver = controlledPlayerVehicle.GetComponent<DRTNoisyPlayerVehicleDriver>();
                if (noisyPlayerVehicleDriver == null)
                {
                    noisyPlayerVehicleDriver = controlledPlayerVehicle.gameObject.AddComponent<DRTNoisyPlayerVehicleDriver>();
                }
            }

            noisyPlayerVehicleDriver.Configure(
                controlledVehicleType,
                controlledVehicleSpeedMultiplier,
                playerWaypointReachDistanceMeters,
                arrivalDistanceMeters,
                noisyGleyLateralNoise,
                noisyGleySpeedNoise,
                noisyGleyNoiseFrequency,
                noisyGleyNoiseStrength,
                noisyGleyNoiseIrregularity,
                usePPOSpeedLimit,
                ppoSpeedLimitMetersPerSecond);

            vehicleDriver = noisyPlayerVehicleDriver;
            LogInfo(
                $"[BUSCONTROLLER] Noisy Gley is driving scene PlayerCar '{controlledPlayerVehicle.name}' " +
                "through DRTNoisyPlayerVehicleDriver.");
            return true;
        }

        private bool ResolveControlledPlayerVehicle(bool logIfMissing)
        {
            activeGleyVehicleIndex = -1;
            DisableNoisyGleyVehicleDriver();
            DisableNoisyPlayerVehicleDriver();
            DisablePPOVehicleDriver();
            DisablePPOPurePursuitVehicleDriver();
            DisableTrainVehicleDriver();

            if (!ResolveControlledPlayerTransform(logIfMissing))
            {
                return false;
            }

            playerVehicleDriver = controlledPlayerVehicle.GetComponent<DRTPlayerVehicleDriver>();
            if (playerVehicleDriver == null)
            {
                playerVehicleDriver = controlledPlayerVehicle.gameObject.AddComponent<DRTPlayerVehicleDriver>();
            }

            playerVehicleDriver.Configure(
                controlledVehicleType,
                controlledVehicleSpeedMultiplier,
                playerWaypointReachDistanceMeters,
                arrivalDistanceMeters);

            vehicleDriver = playerVehicleDriver;
            return true;
        }

        private bool ResolveControlledTrainVehicle(bool logIfMissing)
        {
            activeGleyVehicleIndex = -1;
            DisableNoisyGleyVehicleDriver();
            DisableNoisyPlayerVehicleDriver();
            gleyVehicleDriver?.StopAndHold(true);
            gleyVehicleDriver?.ReleaseControl();
            DisablePPOVehicleDriver();
            DisablePPOPurePursuitVehicleDriver();

            if (!ResolveControlledPlayerTransform(logIfMissing))
            {
                return false;
            }

            playerVehicleDriver = controlledPlayerVehicle.GetComponent<DRTPlayerVehicleDriver>();
            playerVehicleDriver?.ReleaseControl();

            trainVehicleDriver = controlledPlayerVehicle.GetComponent<DRTTrainVehicleDriver>();
            if (trainVehicleDriver == null)
            {
                trainVehicleDriver = controlledPlayerVehicle.gameObject.AddComponent<DRTTrainVehicleDriver>();
            }

            trainVehicleDriver.enabled = true;
            trainVehicleDriver.Configure(
                controlledVehicleType,
                controlledVehicleSpeedMultiplier,
                playerWaypointReachDistanceMeters,
                arrivalDistanceMeters);

            vehicleDriver = trainVehicleDriver;
            return true;
        }

        private bool ResolveControlledPPOVehicle(bool logIfMissing)
        {
            activeGleyVehicleIndex = -1;
            DisableNoisyGleyVehicleDriver();
            DisableNoisyPlayerVehicleDriver();
            gleyVehicleDriver?.ReleaseControl();
            DisablePPOPurePursuitVehicleDriver();
            DisableTrainVehicleDriver();

            if (!ResolveControlledPlayerTransform(logIfMissing))
            {
                return false;
            }

            DisableLegacyRootPPOVehicleDriver();

            Transform ppoAgentTransform = ResolvePPOAgentTransform(controlledPlayerVehicle, PPODriveAgentObjectName);
            if (ppoAgentTransform == null)
            {
                if (logIfMissing)
                {
                    Debug.LogWarning("[BUSCONTROLLER] PPO agent transform could not be created.");
                }

                return false;
            }

            ppoVehicleDriver = ppoAgentTransform.GetComponent<DRTPPOVehicleDriver>();
            if (ppoVehicleDriver == null)
            {
                ppoVehicleDriver = ppoAgentTransform.gameObject.AddComponent<DRTPPOVehicleDriver>();
            }

            ppoVehicleDriver.enabled = true;
            ppoVehicleDriver.Configure(
                controlledPlayerVehicle,
                controlledVehicleType,
                controlledVehicleSpeedMultiplier,
                playerWaypointReachDistanceMeters,
                arrivalDistanceMeters,
                ppoDrivePolicy,
                ppoOnnxInferenceModel,
                ppoOnnxInferenceDevice);
            ppoVehicleDriver.ConfigureSpeedLimit(usePPOSpeedLimit, ppoSpeedLimitMetersPerSecond);

            var collisionRelay = controlledPlayerVehicle.GetComponent<DRTPPOVehicleCollisionRelay>();
            if (collisionRelay == null)
            {
                collisionRelay = controlledPlayerVehicle.gameObject.AddComponent<DRTPPOVehicleCollisionRelay>();
            }

            collisionRelay.Configure(ppoVehicleDriver);

            vehicleDriver = ppoVehicleDriver;
            return true;
        }

        private bool ResolveControlledPPOPurePursuitVehicle(bool logIfMissing)
        {
            activeGleyVehicleIndex = -1;
            DisableNoisyGleyVehicleDriver();
            DisableNoisyPlayerVehicleDriver();
            gleyVehicleDriver?.ReleaseControl();
            DisablePPOVehicleDriver();
            DisableTrainVehicleDriver();

            if (!ResolveControlledPlayerTransform(logIfMissing))
            {
                return false;
            }

            Transform ppoAgentTransform = ResolvePPOAgentTransform(controlledPlayerVehicle, PPOPurePursuitAgentObjectName);
            if (ppoAgentTransform == null)
            {
                if (logIfMissing)
                {
                    Debug.LogWarning("[BUSCONTROLLER] PPO pure pursuit agent transform could not be created.");
                }

                return false;
            }

            ppoPurePursuitVehicleDriver = ppoAgentTransform.GetComponent<DRTPPOPurePursuitVehicleDriver>();
            if (ppoPurePursuitVehicleDriver == null)
            {
                ppoPurePursuitVehicleDriver = ppoAgentTransform.gameObject.AddComponent<DRTPPOPurePursuitVehicleDriver>();
            }

            ppoPurePursuitVehicleDriver.enabled = true;
            ppoPurePursuitVehicleDriver.Configure(
                controlledPlayerVehicle,
                controlledVehicleType,
                controlledVehicleSpeedMultiplier,
                playerWaypointReachDistanceMeters,
                arrivalDistanceMeters,
                ppoDrivePolicy,
                ppoOnnxInferenceModel,
                ppoOnnxInferenceDevice);
            ppoPurePursuitVehicleDriver.ConfigureSpeedLimit(usePPOSpeedLimit, ppoSpeedLimitMetersPerSecond);
            ApplyPPOPurePursuitParameters(ppoPurePursuitVehicleDriver);

            var collisionRelay = controlledPlayerVehicle.GetComponent<DRTPPOPurePursuitVehicleCollisionRelay>();
            if (collisionRelay == null)
            {
                collisionRelay = controlledPlayerVehicle.gameObject.AddComponent<DRTPPOPurePursuitVehicleCollisionRelay>();
            }

            collisionRelay.Configure(ppoPurePursuitVehicleDriver);

            vehicleDriver = ppoPurePursuitVehicleDriver;
            return true;
        }

        private void ApplyPPOPurePursuitParameters(DRTPPOPurePursuitVehicleDriver driver)
        {
            if (driver == null)
            {
                return;
            }

            driver.ConfigurePurePursuitParameters(
                ppoPurePursuitMinLookaheadMeters,
                ppoPurePursuitMaxLookaheadMeters,
                ppoPurePursuitZeroActionLookaheadNormalized,
                ppoPurePursuitMinTargetSpeedRatio,
                ppoPurePursuitMaxTargetSpeedRatio,
                ppoPurePursuitZeroActionSpeedNormalized,
                ppoPurePursuitThrottleInputSmoothing,
                ppoPurePursuitSteeringInputSmoothing,
                ppoPurePursuitCurvatureSmoothingBeta,
                ppoPurePursuitSpeedRewardPerSecond,
                ppoPurePursuitProgressRewardPerMeter,
                ppoPurePursuitDestinationProgressRewardPerMeter,
                ppoPurePursuitDestinationReward,
                ppoPurePursuitLookaheadChangePenaltyPerMeter,
                ppoPurePursuitLateralErrorPenaltyPerMeter,
                ppoPurePursuitLocalLateralVelocityPenaltyPerMeter,
                ppoPurePursuitLocalLateralVelocityCurvatureGain,
                ppoPurePursuitHeadingErrorPenaltyPerMeter,
                ppoPurePursuitOverspeedPenaltyPerSecond,
                ppoPurePursuitMaxCrossTrackErrorMeters,
                ppoPurePursuitHardCrossTrackLimitMeters,
                ppoPurePursuitAssignedRouteExitPenaltyMagnitude,
                ppoPurePursuitNoMovementTimeoutRealSeconds,
                ppoPurePursuitMinimumMovementMeters);
        }

        private Transform ResolvePPOAgentTransform(Transform vehicleRoot, string agentObjectName)
        {
            if (vehicleRoot == null)
            {
                return null;
            }

            Transform agentTransform = vehicleRoot.Find(agentObjectName);
            if (agentTransform == null)
            {
                var agentObject = new GameObject(agentObjectName);
                agentTransform = agentObject.transform;
                agentTransform.SetParent(vehicleRoot, false);
            }

            agentTransform.localPosition = Vector3.zero;
            agentTransform.localRotation = Quaternion.identity;
            agentTransform.localScale = Vector3.one;

            if (agentTransform.GetComponent<BehaviorParameters>() == null)
            {
                agentTransform.gameObject.AddComponent<BehaviorParameters>();
            }

            return agentTransform;
        }

        private void DisableLegacyRootPPOVehicleDriver()
        {
            if (controlledPlayerVehicle == null)
            {
                return;
            }

            var rootDriver = controlledPlayerVehicle.GetComponent<DRTPPOVehicleDriver>();
            if (rootDriver == null || rootDriver == ppoVehicleDriver)
            {
                return;
            }

            rootDriver.ReleaseControl();
            rootDriver.enabled = false;
        }

        private void DisablePPOVehicleDriver()
        {
            if (ppoVehicleDriver == null)
            {
                return;
            }

            ppoVehicleDriver.ReleaseControl();
            ppoVehicleDriver.enabled = false;
        }

        private void DisableNoisyGleyVehicleDriver()
        {
            if (noisyGleyVehicleDriver == null)
            {
                noisyGleyVehicleDriver = GetComponent<DRTNoisyGleyVehicleDriver>();
            }

            if (noisyGleyVehicleDriver == null)
            {
                return;
            }

            noisyGleyVehicleDriver.ReleaseControl();
            noisyGleyVehicleDriver.enabled = false;
        }

        private void DisableNoisyPlayerVehicleDriver()
        {
            if (noisyPlayerVehicleDriver == null && controlledPlayerVehicle != null)
            {
                noisyPlayerVehicleDriver = controlledPlayerVehicle.GetComponent<DRTNoisyPlayerVehicleDriver>();
            }

            if (noisyPlayerVehicleDriver == null)
            {
                return;
            }

            noisyPlayerVehicleDriver.ReleaseControl();
            noisyPlayerVehicleDriver.enabled = false;
        }

        private void DisablePPOPurePursuitVehicleDriver()
        {
            if (ppoPurePursuitVehicleDriver == null && controlledPlayerVehicle != null)
            {
                Transform agentTransform = controlledPlayerVehicle.Find(PPOPurePursuitAgentObjectName);
                if (agentTransform != null)
                {
                    ppoPurePursuitVehicleDriver = agentTransform.GetComponent<DRTPPOPurePursuitVehicleDriver>();
                }
            }

            if (ppoPurePursuitVehicleDriver == null)
            {
                return;
            }

            ppoPurePursuitVehicleDriver.ReleaseControl();
            ppoPurePursuitVehicleDriver.enabled = false;
        }

        private void DisableTrainVehicleDriver()
        {
            if (trainVehicleDriver == null && controlledPlayerVehicle != null)
            {
                trainVehicleDriver = controlledPlayerVehicle.GetComponent<DRTTrainVehicleDriver>();
            }

            if (trainVehicleDriver == null)
            {
                return;
            }

            trainVehicleDriver.ReleaseControl();
            trainVehicleDriver.enabled = false;
        }

        private bool ResolveControlledPlayerTransform(bool logIfMissing)
        {
            if (TryResolveScenePlayerTransform(out Transform scenePlayer))
            {
                controlledPlayerVehicle = scenePlayer;
                return true;
            }

            if (controlledPlayerVehicle != null)
            {
                return true;
            }

            if (logIfMissing)
            {
                Debug.LogWarning("[BUSCONTROLLER] Player vehicle not found. Assign TrafficComponent.player or name the player object 'Player'.");
            }

            return false;
        }

        private bool TryResolveScenePlayerTransform(out Transform playerTransform)
        {
            playerTransform = null;

            var trafficComponent = FindObjectOfType<TrafficComponent>();
            if (trafficComponent != null && trafficComponent.player != null)
            {
                playerTransform = trafficComponent.player;
                return true;
            }

            GameObject playerObject = GameObject.Find("Player");
            if (playerObject == null)
            {
                return false;
            }

            playerTransform = playerObject.transform;
            return true;
        }

        private int ResolveGleyVehicleIndex(bool logIfMissing)
        {
            bool hasScenePlayer = TryResolveScenePlayerTransform(out Transform scenePlayer);
            if (API.IsInitialized() &&
                hasScenePlayer &&
                TryGetGleyVehicleIndex(scenePlayer, out int scenePlayerVehicleIndex))
            {
                return scenePlayerVehicleIndex;
            }

            int fallbackVehicleIndex = Mathf.Max(0, vehicleIndex);
            if (logIfMissing && API.IsInitialized())
            {
                if (hasScenePlayer)
                {
                    Debug.LogWarning(
                        $"[BUSCONTROLLER] Scene player '{scenePlayer.name}' is not a Gley traffic vehicle. " +
                        $"Falling back to configured vehicleIndex={fallbackVehicleIndex} for {physicalDriveMode}.");
                }

                VehicleComponent fallbackVehicle = API.GetVehicleComponent(fallbackVehicleIndex);
                if (fallbackVehicle == null)
                {
                    Debug.LogWarning($"[BUSCONTROLLER] Configured Gley vehicle not found. vehicleIndex={fallbackVehicleIndex}");
                }
            }

            return fallbackVehicleIndex;
        }

        private int GetActiveGleyVehicleIndex()
        {
            return activeGleyVehicleIndex >= 0
                ? activeGleyVehicleIndex
                : Mathf.Max(0, vehicleIndex);
        }

        private void LogActiveGleyVehicleIfChanged()
        {
            if (activeGleyVehicleIndex < 0 ||
                (lastLoggedActiveGleyVehicleIndex == activeGleyVehicleIndex &&
                 lastLoggedActiveGleyMode == physicalDriveMode))
            {
                return;
            }

            lastLoggedActiveGleyVehicleIndex = activeGleyVehicleIndex;
            lastLoggedActiveGleyMode = physicalDriveMode;
            LogInfo(
                $"[BUSCONTROLLER] {physicalDriveMode} active vehicleIndex={activeGleyVehicleIndex}, " +
                $"vehicle={ControlledVehicleName}");
        }

        private static bool TryGetGleyVehicleIndex(Transform candidateRoot, out int resolvedVehicleIndex)
        {
            resolvedVehicleIndex = -1;
            if (candidateRoot == null || !API.IsInitialized())
            {
                return false;
            }

            if (TryGetGleyVehicleIndex(candidateRoot.gameObject, out resolvedVehicleIndex))
            {
                return true;
            }

            VehicleComponent parentVehicle = candidateRoot.GetComponentInParent<VehicleComponent>();
            if (parentVehicle != null &&
                TryGetGleyVehicleIndex(parentVehicle.gameObject, out resolvedVehicleIndex))
            {
                return true;
            }

            VehicleComponent[] childVehicles = candidateRoot.GetComponentsInChildren<VehicleComponent>(true);
            for (int i = 0; i < childVehicles.Length; i++)
            {
                if (childVehicles[i] != null &&
                    TryGetGleyVehicleIndex(childVehicles[i].gameObject, out resolvedVehicleIndex))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool TryGetGleyVehicleIndex(GameObject candidateVehicle, out int resolvedVehicleIndex)
        {
            resolvedVehicleIndex = -1;
            if (candidateVehicle == null || !API.IsInitialized())
            {
                return false;
            }

            int vehicleListIndex = API.GetVehicleIndex(candidateVehicle);
            if (vehicleListIndex < 0 || API.GetVehicleComponent(vehicleListIndex) == null)
            {
                return false;
            }

            resolvedVehicleIndex = vehicleListIndex;
            return true;
        }

        private Transform GetControlledVehicleTransform()
        {
            if (vehicleDriver != null && vehicleDriver.VehicleTransform != null)
            {
                return vehicleDriver.VehicleTransform;
            }

            return controlledPlayerVehicle;
        }

        private void TryInitializeDriving()
        {
            if (passengerManager == null || nextStopSelector == null || stops.Count == 0)
            {
                return;
            }

            if (!API.IsInitialized())
            {
                return;
            }

            if (!ResolveControlledVehicle(true))
            {
                return;
            }

            initialized = true;
            currentStopId = ResolveEpisodeStartStopId();
            targetStopId = 0;
            EnsureBackgroundTrafficStateInitialized();
            ApplyBackgroundTrafficState();
            ResetControlledVehicleForEpisode();
            ResetTravelDistanceSample(GetControlledVehicleBodyPosition());
            RecordVehicleTrace("episode_start");
            if (UsesMatrixTeleport && !EnsureTravelTimeMatrix())
            {
                initialized = false;
                return;
            }

            ApplyCameraFollow(GetControlledVehicleTransform());
            LogInfo(
                $"[BUSCONTROLLER] Initialized mode={TravelExecutionModeName}, " +
                $"physicalDriver={PhysicalDriveModeName}, ppoPolicy={PPODrivePolicyName}, " +
                $"policy={NextStopPolicyName}, " +
                $"ppoRoute={PPOTrainingRouteName}, " +
                $"driver={ControlledDriverName}, vehicle={ControlledVehicleName}, " +
                $"gleyControl={UsesGleyVehicleControl}, gleySpeedMultiplier={gleyControlledVehicleSpeedMultiplier:0.00}, " +
                $"noisyGley=steer:{noisyGleyLateralNoise:0.000}/speed:{noisyGleySpeedNoise:0.000}/freq:{noisyGleyNoiseFrequency:0.00}/strength:{noisyGleyNoiseStrength:0.00}/irregular:{noisyGleyNoiseIrregularity:0.00}, " +
                $"firstTargetHint={currentStopId}");
            SendToNextStop();
        }

        public void EnableBackgroundTraffic()
        {
            SetBackgroundTrafficEnabled(true);
        }

        public void DisableBackgroundTraffic()
        {
            SetBackgroundTrafficEnabled(false);
        }

        public void ToggleBackgroundTraffic()
        {
            SetBackgroundTrafficEnabled(!backgroundTrafficEnabled);
        }

        public void SetBackgroundTrafficEnabled(bool enabled)
        {
            backgroundTrafficEnabledOnStart = enabled;
            backgroundTrafficEnabled = enabled;
            backgroundTrafficStateInitialized = true;
            ApplyBackgroundTrafficState();
        }

        private void BeginDwellAtTarget()
        {
            driving = false;
            vehicleDriver?.StopAndHold(true);

            if (dwellRoutine != null)
            {
                StopCoroutine(dwellRoutine);
            }

            dwellRoutine = StartCoroutine(WaitForProximityThenDwell(targetStopId));
        }

        private void StopEpisodeCoroutines()
        {
            if (dwellRoutine != null)
            {
                StopCoroutine(dwellRoutine);
                dwellRoutine = null;
            }

            if (decisionRoutine != null)
            {
                StopCoroutine(decisionRoutine);
                decisionRoutine = null;
            }

            nextStopSelector?.CancelDecision();
            vehicleDriver?.StopAndHold(false);
        }

        private void ResetControlledVehicleForEpisode()
        {
            if (!ResolveControlledVehicle(true))
            {
                return;
            }

            if (TryGetStartWaypoint(out TrafficWaypoint startWaypoint))
            {
                Transform controlledTransform = GetControlledVehicleTransform();
                Quaternion fallbackRotation = controlledTransform != null ? controlledTransform.rotation : transform.rotation;
                Quaternion rotation = GetWaypointForwardRotation(startWaypoint, fallbackRotation);
                vehicleDriver.TeleportTo(startWaypoint.Position, rotation, startWaypoint.ListIndex);
                ApplyCameraFollow(GetControlledVehicleTransform());
                ResetLegSafetyState(GetControlledVehicleBodyPosition());
                return;
            }

            vehicleDriver.StopAndHold(true);
            ApplyCameraFollow(GetControlledVehicleTransform());
            ResetLegSafetyState(GetControlledVehicleBodyPosition());
        }

        public bool PrepareAllStationCalibration(bool disableBackgroundTraffic, out string error)
        {
            error = null;

            ResolveReferences();
            LoadStops(false);
            StopEpisodeCoroutines();
            initialized = false;
            episodeFinished = true;
            driving = false;
            waitingForArrivalProximity = false;
            targetStopId = 0;
            currentStopId = ResolveEpisodeStartStopId();
            ClearAssignedPathVisualization();

            if (stops.Count < 2)
            {
                error = "At least two DRT stops are required.";
                return false;
            }

            if (!API.IsInitialized())
            {
                error = "Gley Traffic API is not initialized.";
                return false;
            }

            if (!ResolveControlledGleyVehicle(true))
            {
                error = $"Gley controlled vehicle not found. vehicleIndex={vehicleIndex}";
                return false;
            }

            EnsureBackgroundTrafficStateInitialized();
            if (disableBackgroundTraffic)
            {
                SetBackgroundTrafficEnabled(false);
            }
            else
            {
                ApplyBackgroundTrafficState();
            }

            vehicleDriver.StopAndHold(true);
            ApplyCameraFollow(GetControlledVehicleTransform());
            ResetLegSafetyState(GetControlledVehicleBodyPosition());
            ResetTravelDistanceSample(GetControlledVehicleBodyPosition());
            return true;
        }

        public bool TeleportCalibrationVehicleToStop(int stopId, out string error)
        {
            error = null;

            if (!API.IsInitialized())
            {
                error = "Gley Traffic API is not initialized.";
                return false;
            }

            if (!TryGetStop(stopId, out DRTStop stop))
            {
                error = $"Stop {stopId} was not found.";
                return false;
            }

            if (!ResolveControlledGleyVehicle(true))
            {
                error = $"Gley controlled vehicle not found. vehicleIndex={vehicleIndex}";
                return false;
            }

            Vector3 servicePoint = GetStopServicePoint(stop);
            Transform controlledTransform = GetControlledVehicleTransform();
            Quaternion rotation = controlledTransform != null ? controlledTransform.rotation : Quaternion.identity;
            TrafficWaypoint closestWaypoint = API.GetClosestWaypoint(servicePoint);
            if (closestWaypoint != null)
            {
                rotation = GetWaypointForwardRotation(closestWaypoint, rotation);
            }

            currentStopId = stopId;
            targetStopId = 0;
            driving = false;
            waitingForArrivalProximity = false;
            vehicleDriver.TeleportTo(servicePoint, rotation, closestWaypoint != null ? closestWaypoint.ListIndex : -1);
            vehicleDriver.StopAndHold(true);
            ApplyCameraFollow(GetControlledVehicleTransform());
            ResetLegSafetyState(GetControlledVehicleBodyPosition());
            ResetTravelDistanceSample(GetControlledVehicleBodyPosition());
            return true;
        }

        public bool AssignCalibrationRoute(
            int fromStopId,
            int toStopId,
            out int pathWaypointCount,
            out float plannedPathDistanceMeters,
            out string error)
        {
            pathWaypointCount = 0;
            plannedPathDistanceMeters = 0f;
            error = null;

            if (fromStopId == toStopId)
            {
                error = "Origin and destination stops are the same.";
                return false;
            }

            if (!API.IsInitialized())
            {
                error = "Gley Traffic API is not initialized.";
                return false;
            }

            if (!TryGetStop(fromStopId, out _) || !TryGetStop(toStopId, out DRTStop destinationStop))
            {
                error = $"Stop lookup failed. from={fromStopId}, to={toStopId}";
                return false;
            }

            if (!ResolveControlledGleyVehicle(true))
            {
                error = $"Gley controlled vehicle not found. vehicleIndex={vehicleIndex}";
                return false;
            }

            currentStopId = fromStopId;
            targetStopId = toStopId;
            waitingForArrivalProximity = false;

            Vector3 servicePoint = GetStopServicePoint(destinationStop);
            Vector3 routeStartPoint = GetGleyRouteStartPoint(out _);
            List<int> path = API.GetPath(routeStartPoint, servicePoint, controlledVehicleType);
            if (path == null || path.Count == 0)
            {
                error = $"Gley path not found. from={fromStopId}, to={toStopId}";
                return false;
            }

            if (!vehicleDriver.SetPath(path, servicePoint))
            {
                error = $"Controlled vehicle path assignment failed. from={fromStopId}, to={toStopId}";
                return false;
            }

            driving = true;
            UpdateAssignedPathVisualization(routeStartPoint, path, servicePoint);
            ResetLegSafetyState(GetControlledVehicleBodyPosition());
            ResetTravelDistanceSample(GetControlledVehicleBodyPosition());
            pathWaypointCount = path.Count;
            plannedPathDistanceMeters = assignedPathDistanceMeters;
            return true;
        }

        public float GetCalibrationDistanceToStopMeters(int stopId)
        {
            return GetDistanceToStopMeters(stopId);
        }

        public void StopCalibrationVehicle(bool zeroVelocity)
        {
            driving = false;
            waitingForArrivalProximity = false;
            targetStopId = 0;
            vehicleDriver?.StopAndHold(zeroVelocity);
        }

        private bool TryGetStartWaypoint(out TrafficWaypoint waypoint)
        {
            waypoint = null;

            if (!API.IsInitialized())
            {
                return false;
            }

            DRTStop startStop = null;
            int startCandidateStopId = currentStopId > 0 ? currentStopId : startStopId;
            if (!TryGetStop(startCandidateStopId, out startStop) && stops.Count > 0)
            {
                startStop = stops[0];
            }

            if (startStop == null)
            {
                return false;
            }

            Vector3 startPosition = GetStopServicePoint(startStop);
            waypoint = API.GetClosestWaypoint(startPosition);
            if (waypoint == null)
            {
                Debug.LogWarning($"[BUSCONTROLLER] Could not find a traffic waypoint near start stop {startStop.StopId}.");
            }

            return waypoint != null;
        }

        private Quaternion GetWaypointForwardRotation(TrafficWaypoint waypoint, Quaternion fallback)
        {
            if (waypoint == null || waypoint.Neighbors == null || waypoint.Neighbors.Length == 0)
            {
                return fallback;
            }

            var nextWaypoint = API.GetWaypointFromIndex(waypoint.Neighbors[0]);
            if (nextWaypoint == null)
            {
                return fallback;
            }

            Vector3 forward = nextWaypoint.Position - waypoint.Position;
            forward.y = 0f;
            return forward.sqrMagnitude > 0.001f
                ? Quaternion.LookRotation(forward.normalized, Vector3.up)
                : fallback;
        }

        private IEnumerator WaitForProximityThenDwell(int reachedStopId)
        {
            waitingForArrivalProximity = true;
            float waitStartTime = Time.time;

            while (!episodeFinished)
            {
                float distance = GetDistanceToStopMeters(reachedStopId);
                if (distance <= arrivalDistanceMeters)
                {
                    break;
                }

                if (Time.time - waitStartTime >= arrivalWaitTimeoutSeconds)
                {
                    Debug.LogWarning(
                        $"[BUSCONTROLLER] Controlled vehicle arrived for Stop {reachedStopId}, " +
                        $"but vehicle-stop distance is {FormatMeters(distance)} > {arrivalDistanceMeters:0.00}m. " +
                        "Boarding/dropoff skipped. Move the BusStop closer to the road waypoint or increase the arrival threshold.");
                    waitingForArrivalProximity = false;
                    CompleteActiveRouteLeg(reachedStopId, null);
                    RecordVehicleTrace("arrival_timeout");
                    currentStopId = reachedStopId;
                    yield return new WaitForSeconds(dwellSeconds);
                    SendToNextStop();
                    yield break;
                }

                yield return null;
            }

            waitingForArrivalProximity = false;

            if (episodeFinished)
            {
                yield break;
            }

            if (ProcessStopArrivalAndMaybeFinish(reachedStopId))
            {
                yield break;
            }

            yield return new WaitForSeconds(dwellSeconds);
            SendToNextStop();
        }

        private void SendToNextStop()
        {
            if (decisionRoutine != null)
            {
                StopCoroutine(decisionRoutine);
            }

            decisionRoutine = StartCoroutine(SelectAndSendToNextStop());
        }

        private IEnumerator SelectAndSendToNextStop()
        {
            if (episodeFinished)
            {
                yield break;
            }

            if (UsesMatrixTeleport)
            {
                ResolveControlledVehicle(false);
            }
            else if (!ResolveControlledVehicle(true))
            {
                driving = false;
                yield break;
            }

            AdvanceDemandToCurrentTime();
            passengerManager?.UpdateRequestStates(episodeTimeSeconds);

            int nextStopId = -1;
            if (TrySelectPPOTrainingRouteStop(out nextStopId))
            {
                if (nextStopId < 1)
                {
                    yield break;
                }
            }
            else if (nextStopSelector.UsesMlAgentsDecisionPolicy)
            {
                bool decisionStarted = nextStopSelector.BeginDecision(currentStopId, stops, passengerManager, episodeTimeSeconds);
                if (decisionStarted)
                {
                    float waitStartTime = Time.time;
                    while (!episodeFinished && !nextStopSelector.TryConsumeDecision(out nextStopId))
                    {
                        if (Time.time - waitStartTime >= nextStopSelector.MaxDecisionWaitSeconds)
                        {
                            nextStopSelector.CancelDecision();
                            break;
                        }

                        yield return null;
                    }
                }
            }
            else
            {
                switch (NextStopPolicy)
                {
                    case DRTNextStopPolicy.AllStationRunner:
                        nextStopId = nextStopSelector.SelectAllStationRunnerStopId(
                            currentStopId,
                            stops,
                            episodeTimeSeconds);
                        break;
                    case DRTNextStopPolicy.GreedyNearestFeasible:
                        nextStopId = nextStopSelector.SelectGreedyNearestFeasibleStopId(
                            currentStopId,
                            stops,
                            passengerManager,
                            episodeTimeSeconds);
                        break;
                    case DRTNextStopPolicy.Fifo:
                        nextStopId = nextStopSelector.SelectFifoStopId(
                            currentStopId,
                            stops,
                            passengerManager,
                            episodeTimeSeconds);
                        break;
                    default:
                        nextStopId = nextStopSelector.SelectVanillaSequentialStopId(
                            currentStopId,
                            stops,
                            passengerManager,
                            episodeTimeSeconds);
                        break;
                }
            }

            if (stops.Count > 1 && nextStopId == currentStopId)
            {
                int replacementStopId = GetNextNonCurrentStopId(currentStopId);
                LogInfo(
                    $"Next stop matched current stop. current={currentStopId}, " +
                    $"requested={nextStopId}, replacement={replacementStopId}");
                nextStopId = replacementStopId;
            }

            if (!TryGetStop(nextStopId, out DRTStop nextStop))
            {
                FinishEpisode($"No valid next stop found. Requested Stop={nextStopId}");
                yield break;
            }

            if (UsesMatrixTeleport)
            {
                yield return ExecuteMatrixTeleportLeg(nextStop);
                yield break;
            }

            Vector3 servicePoint = GetStopServicePoint(nextStop);
            LogRouteDiagnostics(nextStop, servicePoint);
            Vector3 routeStartPoint = GetGleyRouteStartPoint(out int routeStartWaypointIndex);
            var path = API.GetPath(routeStartPoint, servicePoint, controlledVehicleType);
            if (path == null || path.Count == 0)
            {
                driving = false;
                ClearAssignedPathVisualization();
                Debug.LogWarning(
                    $"[BUSCONTROLLER] PathAssignmentSkipped driver={ControlledDriverName}, vehicle={ControlledVehicleName}, candidateStop={nextStop.StopId}, " +
                    $"candidateObject={nextStop.name}, routeStartWaypoint={routeStartWaypointIndex}. " +
                    "Check that this BusStop is close to a Gley traffic waypoint and allowed for this vehicle type.");
                HandleVehicleFailure($"Path assignment failed. Requested Stop={nextStop.StopId}", failurePenalty);
                yield break;
            }

            ConfigurePPOVehicleDestinationEpisodeEnd(nextStop.StopId);
            if (vehicleDriver == null || !vehicleDriver.SetPath(path, servicePoint))
            {
                driving = false;
                ClearAssignedPathVisualization();
                HandleVehicleFailure($"Controlled vehicle path assignment failed. Requested Stop={nextStop.StopId}", failurePenalty);
                yield break;
            }

            targetStopId = nextStop.StopId;
            driving = true;
            UpdateAssignedPathVisualization(routeStartPoint, path, servicePoint);
            BeginRouteLeg(nextStop.StopId, path.Count, assignedPathDistanceMeters);
            ResetLegSafetyState(GetControlledVehicleBodyPosition());
            LogPathAssignment(nextStop, servicePoint, path, routeStartWaypointIndex);
        }

        private int GetNextNonCurrentStopId(int stopId)
        {
            if (stops == null || stops.Count == 0)
            {
                return -1;
            }

            int firstValidStopId = -1;
            for (int i = 0; i < stops.Count; i++)
            {
                if (stops[i] == null)
                {
                    continue;
                }

                if (firstValidStopId < 1)
                {
                    firstValidStopId = stops[i].StopId;
                }

                if (stops[i].StopId == stopId)
                {
                    for (int offset = 1; offset < stops.Count; offset++)
                    {
                        DRTStop candidate = stops[(i + offset) % stops.Count];
                        if (candidate != null && candidate.StopId != stopId)
                        {
                            return candidate.StopId;
                        }
                    }
                }
            }

            return firstValidStopId != stopId ? firstValidStopId : -1;
        }

        private Vector3 GetGleyRouteStartPoint(out int waypointIndex)
        {
            if (currentStopId > 0 && TryGetStop(currentStopId, out DRTStop currentStop))
            {
                Vector3 currentStopServicePoint = GetStopServicePoint(currentStop);
                var currentStopWaypoint = API.GetClosestWaypoint(currentStopServicePoint);
                if (currentStopWaypoint != null)
                {
                    waypointIndex = currentStopWaypoint.ListIndex;
                    return currentStopWaypoint.Position;
                }
            }

            Vector3 bodyPosition = GetControlledVehicleBodyPosition();
            Transform controlledTransform = GetControlledVehicleTransform();
            Vector3 forward = controlledTransform != null
                ? controlledTransform.forward
                : transform.forward;

            var directedWaypoint = API.GetClosestWaypointInDirection(bodyPosition, forward);
            if (directedWaypoint != null)
            {
                waypointIndex = directedWaypoint.ListIndex;
                return directedWaypoint.Position;
            }

            var closestWaypoint = API.GetClosestWaypoint(bodyPosition);
            if (closestWaypoint != null)
            {
                waypointIndex = closestWaypoint.ListIndex;
                return closestWaypoint.Position;
            }

            waypointIndex = -1;
            return bodyPosition;
        }

        private IEnumerator ExecuteMatrixTeleportLeg(DRTStop nextStop)
        {
            int originStopId = currentStopId > 0 ? currentStopId : startStopId;

            if (!EnsureTravelTimeMatrix() ||
                !travelTimeMatrix.TryGetTravelTimeSeconds(originStopId, nextStop.StopId, out float travelSeconds))
            {
                FinishFailedEpisode($"Travel time matrix lookup failed. from={originStopId}, to={nextStop.StopId}");
                yield break;
            }

            targetStopId = nextStop.StopId;
            driving = false;
            waitingForArrivalProximity = false;
            BeginRouteLeg(nextStop.StopId, 0, travelSeconds * matrixNominalSpeedMetersPerSecond);

            episodeTimeSeconds += travelSeconds;
            episodeTravelDistanceMeters += travelSeconds * matrixNominalSpeedMetersPerSecond;
            AdvanceDemandToCurrentTime();
            passengerManager?.UpdateRequestStates(episodeTimeSeconds);
            TeleportControlledVehicleToStop(nextStop);
            RecordVehicleTrace("matrix_arrival");

            if (logMatrixTravel && !SuppressUnityLogsDuringMatrixTeleport)
            {
                Debug.Log(
                    $"[BUSCONTROLLER] MatrixTeleport from={originStopId}, to={nextStop.StopId}, " +
                    $"travel={travelSeconds:0.00}s, episodeTime={episodeTimeSeconds:0.00}s");
            }

            if (ProcessStopArrivalAndMaybeFinish(nextStop.StopId))
            {
                yield break;
            }

            if (AdvanceMatrixDwellAndMaybeFinish())
            {
                yield break;
            }

            yield return null;
            SendToNextStop();
        }

        private bool AdvanceMatrixDwellAndMaybeFinish()
        {
            float dwellEpisodeSeconds = GetDwellEpisodeSeconds();
            if (dwellEpisodeSeconds <= 0f)
            {
                return false;
            }

            episodeTimeSeconds += dwellEpisodeSeconds;
            AdvanceDemandToCurrentTime();
            passengerManager?.UpdateRequestStates(episodeTimeSeconds);

            if (episodeTimeSeconds >= episodeLengthSeconds)
            {
                FinishEpisode("Episode time ended.");
                return true;
            }

            return false;
        }

        private float GetDwellEpisodeSeconds()
        {
            return Mathf.Max(0f, dwellSeconds) * Mathf.Max(0.01f, simulationSecondsPerRealSecond);
        }

        private bool ProcessStopArrivalAndMaybeFinish(int reachedStopId)
        {
            currentStopId = reachedStopId;

            if (passengerManager == null || nextStopSelector == null)
            {
                FinishEpisode("Passenger manager or next stop selector missing at stop arrival.");
                return true;
            }

            AdvanceDemandToCurrentTime();
            var stopResult = passengerManager.ProcessStopArrival(
                currentStopId,
                episodeTimeSeconds,
                SuppressUnityLogsDuringMatrixTeleport || !logStopProcessing);
            CompleteActiveRouteLeg(currentStopId, stopResult);
            nextStopSelector.RecordStopArrival(stopResult, episodeTimeSeconds);

            if (ShouldFinishPPOTrainingRouteAtStop(currentStopId))
            {
                FinishEpisode($"PPO training route completed. route={PPOTrainingRouteName}");
                return true;
            }

            if (nextStopSelector.IsAllStationRunComplete)
            {
                FinishEpisode("All station runner completed.");
                return true;
            }

            if (!UsesAllStationRunner &&
                stopWhenAllRequestsCompleted &&
                !HasUnfinishedOrPendingRequests() &&
                !ShouldDeferPassengerCompletionFinish())
            {
                FinishEpisode("All passenger requests completed.");
                return true;
            }

            if (!UsesAllStationRunner && episodeTimeSeconds >= episodeLengthSeconds)
            {
                FinishEpisode("Episode time ended.");
                return true;
            }

            return false;
        }

        private void AdvanceDemandToCurrentTime()
        {
            demandGenerator?.SpawnDueRequests(episodeTimeSeconds, SuppressUnityLogsDuringMatrixTeleport);
        }

        private bool HasUnfinishedOrPendingRequests()
        {
            bool hasUnfinishedRequests = passengerManager != null &&
                                         passengerManager.HasUnfinishedRequests(episodeTimeSeconds);
            bool hasPendingScenarioDemand = demandGenerator != null && demandGenerator.HasPendingDemand;
            return hasUnfinishedRequests || hasPendingScenarioDemand;
        }

        private bool ShouldDeferPassengerCompletionFinish()
        {
            return IsPPOTrainingRouteEpisodeActive() &&
                   currentStopId > 0 &&
                   currentStopId != GetActivePPOTrainingRouteEndStopId();
        }

        private bool IsPPOTrainingRouteEpisodeActive()
        {
            if (!usePPOTrainingRouteEpisode && !usePPOTrainingAllStationRouteCycle)
            {
                return false;
            }

            if (usePPOTrainingAllStationRouteCycle)
            {
                return stops.Count > 1;
            }

            return ppoTrainingRouteStartStopId > 0 &&
                   ppoTrainingRouteEndStopId > 0 &&
                   ppoTrainingRouteStartStopId != ppoTrainingRouteEndStopId;
        }

        private bool TryGetPPOTrainingRouteStartStop(out int routeStartStopId)
        {
            routeStartStopId = -1;
            if (!IsPPOTrainingRouteEpisodeActive())
            {
                return false;
            }

            if (!TryEnsurePPOTrainingRouteForEpisode())
            {
                return false;
            }

            int startStop = GetActivePPOTrainingRouteStartStopId();
            if (TryGetStop(startStop, out _))
            {
                routeStartStopId = startStop;
                return true;
            }

            Debug.LogWarning(
                $"[BUSCONTROLLER] PPO training route start stop was not found. " +
                $"start={startStop}, end={GetActivePPOTrainingRouteEndStopId()}");
            return false;
        }

        private bool TrySelectPPOTrainingRouteStop(out int nextStopId)
        {
            nextStopId = -1;
            if (!IsPPOTrainingRouteEpisodeActive())
            {
                return false;
            }

            if (!TryEnsurePPOTrainingRouteForEpisode())
            {
                FinishEpisode("No valid PPO training route could be selected.");
                return true;
            }

            int routeEndStopId = GetActivePPOTrainingRouteEndStopId();
            if (!TryGetStop(routeEndStopId, out _))
            {
                FinishEpisode(
                    $"PPO training route end stop not found. " +
                    $"start={GetActivePPOTrainingRouteStartStopId()}, end={routeEndStopId}");
                return true;
            }

            if (currentStopId == routeEndStopId)
            {
                FinishEpisode($"PPO training route completed before next leg. route={PPOTrainingRouteName}");
                return true;
            }

            nextStopId = routeEndStopId;
            if (nextStopId < 1)
            {
                FinishEpisode($"No valid next stop for PPO training route. route={PPOTrainingRouteName}, current={currentStopId}");
                return true;
            }

            LogInfo(
                $"[BUSCONTROLLER] PPOTrainingRoute route={PPOTrainingRouteName}, " +
                $"current={currentStopId}, selected={nextStopId}");
            return true;
        }

        private int GetActivePPOTrainingRouteStartStopId()
        {
            return usePPOTrainingAllStationRouteCycle && activePPOTrainingRouteStartStopId > 0
                ? activePPOTrainingRouteStartStopId
                : ppoTrainingRouteStartStopId;
        }

        private int GetActivePPOTrainingRouteEndStopId()
        {
            return usePPOTrainingAllStationRouteCycle && activePPOTrainingRouteEndStopId > 0
                ? activePPOTrainingRouteEndStopId
                : ppoTrainingRouteEndStopId;
        }

        private bool TryEnsurePPOTrainingRouteForEpisode()
        {
            if (!usePPOTrainingRouteEpisode && !usePPOTrainingAllStationRouteCycle)
            {
                return false;
            }

            if (!usePPOTrainingAllStationRouteCycle)
            {
                activePPOTrainingRouteEpisodeIndex = episodeIndex;
                activePPOTrainingRouteStartStopId = ppoTrainingRouteStartStopId;
                activePPOTrainingRouteEndStopId = ppoTrainingRouteEndStopId;
                return activePPOTrainingRouteStartStopId > 0 &&
                       activePPOTrainingRouteEndStopId > 0 &&
                       activePPOTrainingRouteStartStopId != activePPOTrainingRouteEndStopId;
            }

            if (activePPOTrainingRouteEpisodeIndex == episodeIndex &&
                activePPOTrainingRouteStartStopId > 0 &&
                activePPOTrainingRouteEndStopId > 0 &&
                activePPOTrainingRouteStartStopId != activePPOTrainingRouteEndStopId)
            {
                return true;
            }

            return TrySelectNextAllStationPPOTrainingRoute();
        }

        private bool TrySelectNextAllStationPPOTrainingRoute()
        {
            List<int> stopIds = stops
                .Where(stop => stop != null && stop.StopId > 0)
                .Select(stop => stop.StopId)
                .Distinct()
                .OrderBy(stopId => stopId)
                .ToList();
            if (stopIds.Count < 2)
            {
                Debug.LogWarning("[BUSCONTROLLER] PPO all-station route cycle requires at least two valid stops.");
                return false;
            }

            int totalDirectedPairs = stopIds.Count * (stopIds.Count - 1);
            int selectedPairIndex = Mathf.Abs(ppoTrainingAllStationRouteCycleIndex) % totalDirectedPairs;
            int cursor = 0;
            for (int i = 0; i < stopIds.Count - 1; i++)
            {
                for (int j = i + 1; j < stopIds.Count; j++)
                {
                    if (cursor == selectedPairIndex)
                    {
                        SetActivePPOTrainingRoute(stopIds[i], stopIds[j], selectedPairIndex, totalDirectedPairs);
                        return true;
                    }

                    cursor++;
                    if (cursor == selectedPairIndex)
                    {
                        SetActivePPOTrainingRoute(stopIds[j], stopIds[i], selectedPairIndex, totalDirectedPairs);
                        return true;
                    }

                    cursor++;
                }
            }

            SetActivePPOTrainingRoute(stopIds[0], stopIds[1], 0, totalDirectedPairs);
            return true;
        }

        private void SetActivePPOTrainingRoute(int fromStopId, int toStopId, int selectedPairIndex, int totalDirectedPairs)
        {
            activePPOTrainingRouteEpisodeIndex = episodeIndex;
            activePPOTrainingRouteStartStopId = fromStopId;
            activePPOTrainingRouteEndStopId = toStopId;
            ppoTrainingAllStationRouteCycleIndex = (selectedPairIndex + 1) % Mathf.Max(1, totalDirectedPairs);
            LogInfo(
                $"[BUSCONTROLLER] PPO all-station route cycle selected " +
                $"episode={episodeIndex}, pairIndex={selectedPairIndex + 1}/{totalDirectedPairs}, route={fromStopId}->{toStopId}");
        }

        private bool ShouldFinishPPOTrainingRouteAtStop(int reachedStopId)
        {
            return IsPPOTrainingRouteEpisodeActive() &&
                   reachedStopId == GetActivePPOTrainingRouteEndStopId();
        }

        private void ConfigurePPOVehicleDestinationEpisodeEnd(int nextStopId)
        {
            DRTPPOVehicleDriver activePPOVehicleDriver = vehicleDriver as DRTPPOVehicleDriver ?? ppoVehicleDriver;
            DRTPPOPurePursuitVehicleDriver activePPOPurePursuitVehicleDriver =
                vehicleDriver as DRTPPOPurePursuitVehicleDriver ?? ppoPurePursuitVehicleDriver;
            if (activePPOVehicleDriver == null && activePPOPurePursuitVehicleDriver == null)
            {
                return;
            }

            bool shouldEndEpisode = !IsPPOTrainingRouteEpisodeActive() ||
                                    nextStopId == GetActivePPOTrainingRouteEndStopId();
            activePPOVehicleDriver?.SetEndEpisodeOnDestinationReached(shouldEndEpisode);
            activePPOPurePursuitVehicleDriver?.SetEndEpisodeOnDestinationReached(shouldEndEpisode);
            LogInfo(
                $"[BUSCONTROLLER] PPO leg episode-end configured. " +
                $"targetStop={nextStopId}, endOnDestination={shouldEndEpisode}, route={PPOTrainingRouteName}");
        }

        private void BeginRouteLeg(int nextStopId, int pathWaypointCount, float plannedPathDistanceMeters)
        {
            if (!ShouldCollectEpisodeExportData())
            {
                return;
            }

            if (activeRouteLeg != null)
            {
                CompleteActiveRouteLeg(-1, null);
            }

            activeRouteLeg = new DRTRouteLegRecord
            {
                LegIndex = routeLegRecords.Count + 1,
                Mode = GetTravelExecutionExportToken(),
                FromStopId = currentStopId > 0 ? currentStopId : startStopId,
                ToStopId = nextStopId,
                ArrivedStopId = -1,
                DepartureTimeSeconds = episodeTimeSeconds,
                DepartureCumulativeDistanceMeters = episodeTravelDistanceMeters,
                PlannedPathDistanceMeters = Mathf.Max(0f, plannedPathDistanceMeters),
                PathWaypointCount = Mathf.Max(0, pathWaypointCount)
            };
        }

        private void CompleteActiveRouteLeg(int reachedStopId, DRTStopProcessResult? stopResult)
        {
            if (activeRouteLeg == null)
            {
                return;
            }

            activeRouteLeg.ArrivedStopId = reachedStopId;
            activeRouteLeg.ArrivalTimeSeconds = episodeTimeSeconds;
            activeRouteLeg.ArrivalCumulativeDistanceMeters = episodeTravelDistanceMeters;
            activeRouteLeg.TravelTimeSeconds = Mathf.Max(0f, activeRouteLeg.ArrivalTimeSeconds - activeRouteLeg.DepartureTimeSeconds);
            activeRouteLeg.LegDistanceMeters = Mathf.Max(0f, activeRouteLeg.ArrivalCumulativeDistanceMeters - activeRouteLeg.DepartureCumulativeDistanceMeters);
            activeRouteLeg.Completed = stopResult.HasValue && reachedStopId == activeRouteLeg.ToStopId;

            if (stopResult.HasValue)
            {
                DRTStopProcessResult result = stopResult.Value;
                activeRouteLeg.BoardedCount = result.BoardedCount;
                activeRouteLeg.DroppedOffCount = result.DroppedOffCount;
                activeRouteLeg.WaitingCount = result.WaitingCount;
                activeRouteLeg.OnBoardCount = result.OnBoardCount;
                activeRouteLeg.CompletedPassengerCount = result.CompletedCount;
            }

            DRTRouteLegRecord completedLeg = activeRouteLeg;
            routeLegRecords.Add(completedLeg);
            string traceEvent = completedLeg.Completed ? "stop_arrival" : "leg_incomplete";
            activeRouteLeg = null;
            RecordVehicleTrace(traceEvent);
            ExportPartialAllStationMatrixCsvIfNeeded(completedLeg);
        }

        private void ResetEpisodeExportState()
        {
            routeLegRecords.Clear();
            vehicleTraceRecords.Clear();
            activeRouteLeg = null;
            episodeCsvExported = false;
            allStationMatrixCsvExported = false;
        }

        private void TeleportControlledVehicleToStop(DRTStop stop)
        {
            if (stop == null || !ResolveControlledVehicle(false))
            {
                return;
            }

            Vector3 servicePoint = GetStopServicePoint(stop);
            Transform controlledTransform = GetControlledVehicleTransform();
            Quaternion rotation = controlledTransform != null ? controlledTransform.rotation : Quaternion.identity;
            TrafficWaypoint closestWaypoint = API.IsInitialized() ? API.GetClosestWaypoint(servicePoint) : null;
            if (closestWaypoint != null)
            {
                rotation = GetWaypointForwardRotation(closestWaypoint, rotation);
            }

            vehicleDriver.TeleportTo(servicePoint, rotation, closestWaypoint != null ? closestWaypoint.ListIndex : -1);
            ApplyCameraFollow(GetControlledVehicleTransform());
            ResetLegSafetyState(GetControlledVehicleBodyPosition());
            ResetTravelDistanceSample(GetControlledVehicleBodyPosition());
        }

        private bool TryGetStop(int stopId, out DRTStop stop)
        {
            for (int i = 0; i < stops.Count; i++)
            {
                if (stops[i] != null && stops[i].StopId == stopId)
                {
                    stop = stops[i];
                    return true;
                }
            }

            stop = null;
            return false;
        }

        private void ApplyCameraFollow(Transform target)
        {
            if (!autoFollowControlledVehicle || target == null)
            {
                return;
            }

            bool targetChanged = !cameraFollowApplied || cameraFollowTarget != target;
            Camera mainCamera = Camera.main;
            if (mainCamera != null)
            {
                var cameraFollow = mainCamera.GetComponent<CameraFollow>();
                if (cameraFollow != null)
                {
                    targetChanged = targetChanged || cameraFollow.target != target;
                    cameraFollow.target = target;
                }
            }

            if (!targetChanged)
            {
                return;
            }

            if (API.IsInitialized() && ShouldSyncTrafficCameraWithFollowTarget())
            {
                API.SetCamera(target);
            }

            cameraFollowTarget = target;
            cameraFollowApplied = true;
        }

        private bool ShouldSyncTrafficCameraWithFollowTarget()
        {
            return physicalDriveMode != DRTPhysicalDriveMode.NoisyGley;
        }

        private void RefreshCameraFollowTargetIfNeeded()
        {
            if (!autoFollowControlledVehicle || UsesMatrixTeleport)
            {
                return;
            }

            Transform target = GetControlledVehicleTransform();
            if (target == null && UsesGleyVehicleControl)
            {
                ResolveControlledVehicle(false);
                target = GetControlledVehicleTransform();
            }

            ApplyCameraFollow(target);
        }

        private void EnsureBackgroundTrafficStateInitialized()
        {
            if (backgroundTrafficStateInitialized)
            {
                return;
            }

            backgroundTrafficEnabled = backgroundTrafficEnabledOnStart;
            backgroundTrafficStateInitialized = true;
        }

        private void SyncBackgroundTrafficInspectorStateIfNeeded()
        {
            if (!Application.isPlaying || !API.IsInitialized())
            {
                return;
            }

            EnsureBackgroundTrafficStateInitialized();
            if (backgroundTrafficEnabled == backgroundTrafficEnabledOnStart &&
                appliedBackgroundTrafficEnabled == backgroundTrafficEnabledOnStart &&
                appliedBackgroundTrafficDensity == enabledTrafficDensity &&
                !HasBackgroundTrafficSafetyProfileChanged())
            {
                return;
            }

            backgroundTrafficEnabled = backgroundTrafficEnabledOnStart;
            ApplyBackgroundTrafficState();
        }

        private void ApplyBackgroundTrafficState()
        {
            if (!API.IsInitialized())
            {
                return;
            }

            if (backgroundTrafficEnabled)
            {
                int requestedNpcDensity = Mathf.Clamp(enabledTrafficDensity, 0, MaxBackgroundTrafficDensity);
                int gleyDensity = GetGleyTrafficDensityTarget(requestedNpcDensity);
                API.SetTrafficDensity(gleyDensity);
                ApplyBackgroundTrafficSafetyProfile();
                int trimmedCount = TrimBackgroundTrafficVehiclesToDensity(requestedNpcDensity);
                appliedBackgroundTrafficEnabled = true;
                appliedBackgroundTrafficDensity = requestedNpcDensity;
                LogInfo(
                    $"[BUSCONTROLLER] Background traffic enabled. npcDensity={requestedNpcDensity}, " +
                    $"gleyDensity={gleyDensity}, trimmed={trimmedCount}");
                return;
            }

            API.SetTrafficDensity(0);
            int removedCount = RemoveBackgroundTrafficVehicles();
            appliedBackgroundTrafficEnabled = false;
            appliedBackgroundTrafficDensity = 0;
            MarkBackgroundTrafficSafetyProfileApplied();
            LogInfo($"[BUSCONTROLLER] Background traffic disabled. removed={removedCount}");
        }

        private bool HasBackgroundTrafficSafetyProfileChanged()
        {
            return !Mathf.Approximately(appliedBackgroundTrafficSafeGapMeters, backgroundTrafficSafeGapMeters) ||
                   !Mathf.Approximately(appliedBackgroundTrafficMinTriggerLengthMeters, backgroundTrafficMinTriggerLengthMeters) ||
                   !Mathf.Approximately(appliedBackgroundTrafficMaxTriggerLengthMeters, backgroundTrafficMaxTriggerLengthMeters) ||
                   !Mathf.Approximately(appliedBackgroundTrafficBrakeTimeSeconds, backgroundTrafficBrakeTimeSeconds) ||
                   !Mathf.Approximately(appliedBackgroundTrafficSpeedLimitMetersPerSecond, backgroundTrafficSpeedLimitMetersPerSecond);
        }

        private void ApplyBackgroundTrafficSafetyProfile()
        {
            Delegates.SetModifyTriggerSize(BackgroundTrafficTriggerSizeModifier);

            var vehicles = API.GetAllVehicles();
            if (vehicles == null)
            {
                MarkBackgroundTrafficSafetyProfileApplied();
                return;
            }

            for (int i = 0; i < vehicles.Length; i++)
            {
                VehicleComponent vehicle = vehicles[i];
                if (vehicle == null)
                {
                    continue;
                }

                bool isControlledGleyVehicle = UsesGleyVehicleControl && i == GetActiveGleyVehicleIndex();
                ApplyBackgroundTrafficSafetyProfile(vehicle, !isControlledGleyVehicle);
            }

            MarkBackgroundTrafficSafetyProfileApplied();
        }

        private void ApplyBackgroundTrafficSafetyProfile(VehicleComponent vehicle, bool applyNpcSpeedLimit)
        {
            vehicle.distanceToStop = backgroundTrafficSafeGapMeters;
            vehicle.brakeTime = backgroundTrafficBrakeTimeSeconds;
            vehicle.updateTrigger = true;
            vehicle.maxTriggerLength = backgroundTrafficMaxTriggerLengthMeters;

            BoxCollider frontCollider = GetFrontTriggerCollider(vehicle);
            if (frontCollider != null)
            {
                SetFrontTriggerLength(frontCollider, backgroundTrafficMinTriggerLengthMeters);
            }

            if (applyNpcSpeedLimit &&
                backgroundTrafficSpeedLimitMetersPerSecond > 0f &&
                vehicle.MovementInfo != null)
            {
                vehicle.MovementInfo.SetMaxVehicleSpeed(backgroundTrafficSpeedLimitMetersPerSecond);
            }
        }

        private void BackgroundTrafficTriggerSizeModifier(
            float currentSpeed,
            BoxCollider frontCollider,
            float maxSpeed,
            float minTriggerLength,
            float maxTriggerLength)
        {
            float minLength = Mathf.Max(minTriggerLength, backgroundTrafficMinTriggerLengthMeters);
            float maxLength = Mathf.Max(minLength, backgroundTrafficMaxTriggerLengthMeters);
            float speedT = maxSpeed > 10f
                ? Mathf.InverseLerp(10f, maxSpeed, currentSpeed)
                : 1f;
            SetFrontTriggerLength(frontCollider, Mathf.Lerp(minLength, maxLength, Mathf.Clamp01(speedT)));
        }

        private static BoxCollider GetFrontTriggerCollider(VehicleComponent vehicle)
        {
            if (vehicle == null || vehicle.frontTrigger == null || vehicle.frontTrigger.childCount == 0)
            {
                return null;
            }

            return vehicle.frontTrigger.GetChild(0).GetComponent<BoxCollider>();
        }

        private static void SetFrontTriggerLength(BoxCollider frontCollider, float length)
        {
            if (frontCollider == null)
            {
                return;
            }

            length = Mathf.Max(0.1f, length);
            frontCollider.size = new Vector3(frontCollider.size.x, frontCollider.size.y, length);
            frontCollider.center = new Vector3(frontCollider.center.x, frontCollider.center.y, length * 0.5f);
        }

        private void MarkBackgroundTrafficSafetyProfileApplied()
        {
            appliedBackgroundTrafficSafeGapMeters = backgroundTrafficSafeGapMeters;
            appliedBackgroundTrafficMinTriggerLengthMeters = backgroundTrafficMinTriggerLengthMeters;
            appliedBackgroundTrafficMaxTriggerLengthMeters = backgroundTrafficMaxTriggerLengthMeters;
            appliedBackgroundTrafficBrakeTimeSeconds = backgroundTrafficBrakeTimeSeconds;
            appliedBackgroundTrafficSpeedLimitMetersPerSecond = backgroundTrafficSpeedLimitMetersPerSecond;
        }

        private int GetGleyTrafficDensityTarget(int npcDensity)
        {
            int controlledGleyVehicleCount = UsesGleyVehicleControl ? 1 : 0;
            int requestedTotal = npcDensity + controlledGleyVehicleCount;
            var vehicles = API.GetAllVehicles();
            return vehicles != null && vehicles.Length > 0
                ? Mathf.Clamp(requestedTotal, 0, vehicles.Length)
                : Mathf.Max(0, requestedTotal);
        }

        private int TrimBackgroundTrafficVehiclesToDensity(int targetNpcCount)
        {
            var vehicles = API.GetAllVehicles();
            if (vehicles == null)
            {
                return 0;
            }

            int activeBackgroundCount = CountActiveBackgroundTrafficVehicles(vehicles);
            int removedCount = 0;
            for (int i = vehicles.Length - 1; i >= 0 && activeBackgroundCount > targetNpcCount; i--)
            {
                if (vehicles[i] == null || !vehicles[i].gameObject.activeSelf)
                {
                    continue;
                }

                if (UsesGleyVehicleControl && i == GetActiveGleyVehicleIndex())
                {
                    continue;
                }

                API.RemoveVehicle(i);
                activeBackgroundCount--;
                removedCount++;
            }

            return removedCount;
        }

        private int RemoveBackgroundTrafficVehicles()
        {
            var vehicles = API.GetAllVehicles();
            if (vehicles == null)
            {
                return 0;
            }

            int removedCount = 0;
            for (int i = 0; i < vehicles.Length; i++)
            {
                if (vehicles[i] != null && vehicles[i].gameObject.activeSelf)
                {
                    if (UsesGleyVehicleControl && i == GetActiveGleyVehicleIndex())
                    {
                        continue;
                    }

                    API.RemoveVehicle(i);
                    removedCount++;
                }
            }

            return removedCount;
        }

        private int CountActiveBackgroundTrafficVehicles()
        {
            if (!API.IsInitialized())
            {
                return 0;
            }

            var vehicles = API.GetAllVehicles();
            return CountActiveBackgroundTrafficVehicles(vehicles);
        }

        private int CountActiveBackgroundTrafficVehicles(VehicleComponent[] vehicles)
        {
            if (vehicles == null)
            {
                return 0;
            }

            int count = 0;
            for (int i = 0; i < vehicles.Length; i++)
            {
                if (vehicles[i] != null && vehicles[i].gameObject.activeSelf)
                {
                    if (UsesGleyVehicleControl && i == GetActiveGleyVehicleIndex())
                    {
                        continue;
                    }

                    count++;
                }
            }

            return count;
        }

        public bool TryGetTravelTimeSeconds(int fromStopId, int toStopId, out float seconds)
        {
            seconds = 0f;
            return EnsureTravelTimeMatrix() &&
                   travelTimeMatrix.TryGetTravelTimeSeconds(fromStopId, toStopId, out seconds);
        }

        public bool TryGetAverageStopTravelTimeMinutes(IReadOnlyList<DRTStop> selectedStops, out float averageMinutes)
        {
            averageMinutes = 0f;
            return EnsureTravelTimeMatrix() &&
                   travelTimeMatrix.TryGetAverageTravelTimeMinutes(selectedStops, out averageMinutes);
        }

        private bool EnsureTravelTimeMatrix()
        {
            if (travelTimeMatrix.IsLoaded)
            {
                return true;
            }

            if (stops.Count == 0)
            {
                LoadStops(false);
            }

            if (stops.Count == 0)
            {
                return false;
            }

            if (travelTimeMatrixLoadAttempted)
            {
                return false;
            }

            travelTimeMatrixLoadAttempted = true;

            TextAsset csvAsset = Resources.Load<TextAsset>(travelTimeMatrixResourceName);
            string loadError = null;
            if (csvAsset != null && travelTimeMatrix.LoadFromCsv(csvAsset.text, stops, out loadError))
            {
                LogInfo(
                    $"[BUSCONTROLLER] Loaded travel time matrix resource={travelTimeMatrixResourceName}, " +
                    $"stops={travelTimeMatrix.StopCount}");
                return true;
            }

            if (csvAsset == null)
            {
                Debug.LogError($"[BUSCONTROLLER] Travel time matrix resource not found. resource={travelTimeMatrixResourceName}");
                return false;
            }

            Debug.LogError($"[BUSCONTROLLER] Travel time matrix CSV invalid. resource={travelTimeMatrixResourceName}, error={loadError}");
            return false;
        }

        public void InvalidateTravelTimeMatrix()
        {
            travelTimeMatrix.Clear();
            travelTimeMatrixLoadAttempted = false;
        }

        private void LogRouteDiagnostics(DRTStop stop, Vector3 servicePoint)
        {
            Transform controlledTransform = GetControlledVehicleTransform();
            if (controlledTransform == null || stop == null)
            {
                return;
            }

            Vector3 bodyPosition = GetControlledVehicleBodyPosition();
            var closestStopWaypoint = API.GetClosestWaypoint(stop.Position);
            var closestPlayerWaypoint = API.GetClosestWaypoint(bodyPosition);
            var directedPlayerWaypoint = API.GetClosestWaypointInDirection(bodyPosition, controlledTransform.forward);
            float stopToWaypointDistance = closestStopWaypoint != null
                ? GetPlanarDistance(stop.Position, closestStopWaypoint.Position)
                : float.PositiveInfinity;
            float bodyToStopDistance = GetPlanarDistance(bodyPosition, stop.Position);
            float bodyToServiceDistance = GetPlanarDistance(bodyPosition, servicePoint);
            int currentWaypoint = closestPlayerWaypoint != null ? closestPlayerWaypoint.ListIndex : -1;
            int directedWaypoint = directedPlayerWaypoint != null ? directedPlayerWaypoint.ListIndex : -1;

            Debug.Log(
                $"[BUSCONTROLLER] RouteRequest driver={ControlledDriverName}, vehicle={ControlledVehicleName}, currentWaypoint={currentWaypoint}, " +
                $"directedWaypoint={directedWaypoint}, " +
                $"targetStop={stop.StopId}, targetObject={stop.name}, bodyToService={FormatMeters(bodyToServiceDistance)}, " +
                $"bodyToStopMarker={FormatMeters(bodyToStopDistance)}, " +
                $"servicePoint={FormatVector(servicePoint)}, stopMarker={FormatVector(stop.Position)}, " +
                $"stopToClosestWaypoint={FormatMeters(stopToWaypointDistance)}");
        }

        private void UpdateAssignedPathVisualization(Vector3 routeStartPoint, List<int> path, Vector3 servicePoint)
        {
            assignedPathPoints.Clear();
            assignedPathDistanceMeters = 0f;

            if (path == null || path.Count == 0)
            {
                ApplyAssignedPathLineRenderer();
                return;
            }

            assignedPathPoints.Add(OffsetAssignedPathPoint(routeStartPoint));

            for (int i = 0; i < path.Count; i++)
            {
                var waypoint = API.GetWaypointFromIndex(path[i]);
                if (waypoint != null)
                {
                    assignedPathPoints.Add(OffsetAssignedPathPoint(waypoint.Position));
                }
            }

            assignedPathPoints.Add(OffsetAssignedPathPoint(servicePoint));
            assignedPathDistanceMeters = CalculateAssignedPathDistanceMeters();
            ApplyAssignedPathLineRenderer();
        }

        private void ClearAssignedPathVisualization()
        {
            assignedPathPoints.Clear();
            assignedPathDistanceMeters = 0f;
            ApplyAssignedPathLineRenderer();
        }

        private Vector3 OffsetAssignedPathPoint(Vector3 point)
        {
            point.y += assignedPathVerticalOffset;
            return point;
        }

        private float CalculateAssignedPathDistanceMeters()
        {
            if (assignedPathPoints.Count < 2)
            {
                return 0f;
            }

            float distance = 0f;
            for (int i = 1; i < assignedPathPoints.Count; i++)
            {
                distance += GetPlanarDistance(assignedPathPoints[i - 1], assignedPathPoints[i]);
            }

            return distance;
        }

        private void EnsureAssignedPathLineRenderer()
        {
            if (assignedPathLineRenderer != null)
            {
                return;
            }

            GameObject lineObject = new GameObject("DRT Assigned Path");
            lineObject.transform.SetParent(transform, false);

            assignedPathLineRenderer = lineObject.AddComponent<LineRenderer>();
            assignedPathLineRenderer.useWorldSpace = true;
            assignedPathLineRenderer.alignment = LineAlignment.View;
            assignedPathLineRenderer.textureMode = LineTextureMode.Tile;
            assignedPathLineRenderer.numCapVertices = 4;
            assignedPathLineRenderer.numCornerVertices = 4;
            assignedPathLineRenderer.enabled = false;

            Shader shader = Shader.Find("Sprites/Default");
            if (shader == null)
            {
                shader = Shader.Find("Unlit/Color");
            }

            if (shader != null)
            {
                assignedPathLineRenderer.material = new Material(shader);
            }
        }

        private void ApplyAssignedPathLineRenderer()
        {
            if (!showAssignedPath || !showAssignedPathInGame || assignedPathPoints.Count < 2)
            {
                if (assignedPathLineRenderer != null)
                {
                    assignedPathLineRenderer.positionCount = 0;
                    assignedPathLineRenderer.enabled = false;
                }

                return;
            }

            EnsureAssignedPathLineRenderer();
            assignedPathLineRenderer.enabled = true;
            assignedPathLineRenderer.startWidth = assignedPathLineWidth;
            assignedPathLineRenderer.endWidth = assignedPathLineWidth;
            assignedPathLineRenderer.startColor = assignedPathColor;
            assignedPathLineRenderer.endColor = assignedPathColor;
            if (assignedPathLineRenderer.material != null)
            {
                assignedPathLineRenderer.material.color = assignedPathColor;
            }

            assignedPathLineRenderer.positionCount = assignedPathPoints.Count;
            assignedPathLineRenderer.SetPositions(assignedPathPoints.ToArray());
        }

        private void LogPathAssignment(DRTStop targetStop, Vector3 servicePoint, List<int> path, int routeStartWaypointIndex)
        {
            string endpointDescription = DescribePathEndpoint(path, servicePoint, out float endToServiceDistance);
            float warningDistance = Mathf.Max(15f, arrivalDistanceMeters + stopWaypointSnapDistanceMeters);
            string integrity = endToServiceDistance <= warningDistance ? "ok" : "warning";
            string pathPreview = BuildPathPreview(path, 12);

            string message =
                $"[BUSCONTROLLER] PathAssigned integrity={integrity}, driver={ControlledDriverName}, vehicle={ControlledVehicleName}, " +
                $"targetStop={targetStop.StopId}, targetObject={targetStop.name}, routeStartWaypoint={routeStartWaypointIndex}, " +
                $"pathWaypoints={path.Count}, visualPathDistance={FormatMeters(assignedPathDistanceMeters)}, servicePoint={FormatVector(servicePoint)}, {endpointDescription}, " +
                $"pathPreview=[{pathPreview}]";

            if (integrity == "warning")
            {
                Debug.LogWarning(message);
            }
            else
            {
                Debug.Log(message);
            }
        }

        private string BuildPathPreview(List<int> path, int maxItems)
        {
            if (path == null || path.Count == 0)
            {
                return "-";
            }

            var builder = new System.Text.StringBuilder();
            int count = Mathf.Min(path.Count, Mathf.Max(1, maxItems));
            for (int i = 0; i < count; i++)
            {
                var waypoint = API.GetWaypointFromIndex(path[i]);
                if (i > 0)
                {
                    builder.Append(" -> ");
                }

                builder.Append(waypoint != null ? waypoint.Name : path[i].ToString());
            }

            if (path.Count > count)
            {
                builder.Append(" -> ...");
            }

            return builder.ToString();
        }

        private string DescribePathEndpoint(List<int> path, Vector3 servicePoint, out float endToServiceDistance)
        {
            endToServiceDistance = float.PositiveInfinity;

            if (path == null || path.Count == 0)
            {
                return "pathEndpoint=none";
            }

            int firstWaypointIndex = path[0];
            int lastWaypointIndex = path[path.Count - 1];
            var lastWaypoint = API.GetWaypointFromIndex(lastWaypointIndex);
            if (lastWaypoint == null)
            {
                return $"firstWaypoint={firstWaypointIndex}, lastWaypoint={lastWaypointIndex}, endpointPosition=unknown, endToService=n/a";
            }

            endToServiceDistance = GetPlanarDistance(lastWaypoint.Position, servicePoint);
            return
                $"firstWaypoint={firstWaypointIndex}, lastWaypoint={lastWaypointIndex}, " +
                $"endpointPosition={FormatVector(lastWaypoint.Position)}, endToService={FormatMeters(endToServiceDistance)}";
        }

        private void MonitorVehicleFailureIfNeeded()
        {
            if (!failEpisodeOnVehicleFault || !initialized || episodeFinished || !API.IsInitialized())
            {
                return;
            }

            if (!ResolveControlledVehicle(true) || vehicleDriver == null || vehicleDriver.VehicleTransform == null || !vehicleDriver.VehicleTransform.gameObject.activeSelf)
            {
                HandleVehicleFailure("Controlled vehicle is missing or inactive.", failurePenalty);
                return;
            }

            if (vehicleDriver.HasCriticalFault)
            {
                string reason = string.IsNullOrWhiteSpace(vehicleDriver.CriticalFaultReason)
                    ? "Controlled vehicle reported a critical fault."
                    : vehicleDriver.CriticalFaultReason;
                HandleVehicleFailure(reason, failurePenalty, false);
                return;
            }

            Vector3 bodyPosition = GetControlledVehicleBodyPosition();
            if (bodyPosition.y <= fallYThreshold)
            {
                HandleVehicleFailure($"Controlled vehicle fell below safety Y threshold. y={bodyPosition.y:0.00}", failurePenalty);
                return;
            }

            if (!UsesPPOVehicleTraining && IsTooFarFromTrafficWaypoint(bodyPosition, out float waypointDistance))
            {
                HandleVehicleFailure($"Controlled vehicle left traffic network. nearestWaypointDistance={FormatMeters(waypointDistance)}", failurePenalty);
                return;
            }

            if (!driving || targetStopId <= 0)
            {
                return;
            }

            if (TryConsumeVehicleDestinationReached())
            {
                BeginDwellAtTarget();
                return;
            }

            float targetDistance = GetTargetDistanceMeters();
            if (float.IsInfinity(targetDistance))
            {
                HandleVehicleFailure($"Target distance unavailable. targetStop={targetStopId}", failurePenalty);
                return;
            }

            if (targetDistance <= arrivalDistanceMeters)
            {
                BeginDwellAtTarget();
                return;
            }

            float movedDistance = hasVehicleMovementSample
                ? GetPlanarDistance(bodyPosition, lastVehicleMovementPosition)
                : float.PositiveInfinity;

            if (!hasVehicleMovementSample || movedDistance >= minimumVehicleMovementMeters)
            {
                ResetLegSafetyState(bodyPosition);
                return;
            }

            if (vehicleDriver.IsTemporarilyBlocked)
            {
                TrackTrafficBlockWait(bodyPosition, targetDistance, movedDistance, vehicleDriver.TemporaryBlockReason);
                return;
            }

            float stillSeconds = Time.realtimeSinceStartup - lastVehicleMovementRealtime;
            if (stillSeconds >= noMovementTimeoutRealSeconds)
            {
                float noMovementPenalty = UsesPPOVehicleTraining ? 0f : failurePenalty;
                HandleVehicleFailure(
                    $"Controlled vehicle body unchanged for {noMovementTimeoutRealSeconds:0.0}s real time. " +
                    $"targetStop={targetStopId}, moved={FormatMeters(movedDistance)}, " +
                    $"distance={FormatMeters(targetDistance)}, speed={GetVehicleSpeedMS():0.00}m/s",
                    noMovementPenalty);
            }
        }

        private void TrackTrafficBlockWait(Vector3 bodyPosition, float targetDistance, float movedDistance, string reason)
        {
            string currentReason = string.IsNullOrWhiteSpace(reason) ? "traffic wait" : reason;
            if (!hasTrafficBlockSample || trafficBlockReason != currentReason)
            {
                hasTrafficBlockSample = true;
                trafficBlockStartRealtime = Time.realtimeSinceStartup;
                trafficBlockReason = currentReason;
            }

            lastVehicleMovementPosition = bodyPosition;
            lastVehicleMovementRealtime = Time.realtimeSinceStartup;
            hasVehicleMovementSample = true;

            if (trafficBlockTimeoutRealSeconds <= 0f)
            {
                return;
            }

            float blockedSeconds = Time.realtimeSinceStartup - trafficBlockStartRealtime;
            if (blockedSeconds >= trafficBlockTimeoutRealSeconds)
            {
                HandleVehicleFailure(
                    $"Controlled vehicle remained blocked by {trafficBlockReason} for {trafficBlockTimeoutRealSeconds:0.0}s real time. " +
                    $"targetStop={targetStopId}, moved={FormatMeters(movedDistance)}, " +
                    $"distance={FormatMeters(targetDistance)}, speed={GetVehicleSpeedMS():0.00}m/s",
                    failurePenalty);
            }
        }

        private void HandleVehicleFailure(string reason, float penalty, bool reportToPPOAgent = true)
        {
            if (!UsesPPOVehicleTraining)
            {
                FinishFailedEpisode(reason);
                return;
            }

            if (reportToPPOAgent &&
                vehicleDriver is DRTPPOVehicleDriver activePPOVehicleDriver &&
                !activePPOVehicleDriver.HasCriticalFault)
            {
                activePPOVehicleDriver.ReportExternalCriticalFault(reason, penalty);
            }
            else if (reportToPPOAgent &&
                     vehicleDriver is DRTPPOPurePursuitVehicleDriver activePPOPurePursuitVehicleDriver &&
                     !activePPOPurePursuitVehicleDriver.HasCriticalFault)
            {
                activePPOPurePursuitVehicleDriver.ReportExternalCriticalFault(reason, penalty);
            }

            LogInfo($"[BUSCONTROLLER] PPO training episode reset after vehicle failure. {reason}");
            ResetEpisodeFromAgent();
        }

        private bool TryConsumeVehicleDestinationReached()
        {
            DRTPPOVehicleDriver activePPOVehicleDriver = vehicleDriver as DRTPPOVehicleDriver ?? ppoVehicleDriver;
            if (activePPOVehicleDriver != null && activePPOVehicleDriver.ConsumeDestinationReached())
            {
                return true;
            }

            DRTPPOPurePursuitVehicleDriver activePPOPurePursuitVehicleDriver =
                vehicleDriver as DRTPPOPurePursuitVehicleDriver ?? ppoPurePursuitVehicleDriver;
            return activePPOPurePursuitVehicleDriver != null &&
                   activePPOPurePursuitVehicleDriver.ConsumeDestinationReached();
        }

        private bool IsTooFarFromTrafficWaypoint(Vector3 position, out float waypointDistance)
        {
            waypointDistance = 0f;

            if (maxRoadWaypointDistanceMeters <= 0f)
            {
                return false;
            }

            var waypoint = API.GetClosestWaypoint(position);
            if (waypoint == null)
            {
                waypointDistance = float.PositiveInfinity;
                return true;
            }

            waypointDistance = GetPlanarDistance(position, waypoint.Position);
            return waypointDistance > maxRoadWaypointDistanceMeters;
        }

        private void ResetLegSafetyState(Vector3 vehiclePosition)
        {
            lastVehicleMovementPosition = vehiclePosition;
            lastVehicleMovementRealtime = Time.realtimeSinceStartup;
            hasVehicleMovementSample = true;
            hasTrafficBlockSample = false;
            trafficBlockReason = string.Empty;
        }

        private void ResetTravelDistanceSample(Vector3 vehiclePosition)
        {
            lastTravelDistanceSamplePosition = vehiclePosition;
            hasTravelDistanceSample = true;
        }

        private void TrackPhysicalTravelDistanceIfNeeded()
        {
            if (UsesMatrixTeleport || !initialized || episodeFinished || GetControlledVehicleTransform() == null)
            {
                return;
            }

            Vector3 bodyPosition = GetControlledVehicleBodyPosition();
            if (!hasTravelDistanceSample)
            {
                ResetTravelDistanceSample(bodyPosition);
                return;
            }

            float distance = GetPlanarDistance(lastTravelDistanceSamplePosition, bodyPosition);
            if (distance > 0.001f)
            {
                episodeTravelDistanceMeters += distance;
                lastTravelDistanceSamplePosition = bodyPosition;
            }
        }

        private void RecordVehicleTraceIfNeeded()
        {
            // Route analysis exports are stop/leg based. Per-second samples add noise and are not persisted.
        }

        private void RecordVehicleTrace(string eventName)
        {
            if (!ShouldCollectEpisodeExportData())
            {
                return;
            }

            Vector3 position = GetControlledVehicleBodyPosition();
            vehicleTraceRecords.Add(new DRTVehicleTraceRecord
            {
                SampleIndex = vehicleTraceRecords.Count + 1,
                EventName = eventName,
                EpisodeTimeSeconds = episodeTimeSeconds,
                CurrentStopId = currentStopId,
                TargetStopId = targetStopId,
                Position = position,
                CumulativeDistanceMeters = episodeTravelDistanceMeters,
                SpeedMetersPerSecond = GetVehicleSpeedMS(),
                Blocked = IsVehicleTemporarilyBlocked,
                BlockReason = TemporaryBlockReason
            });
        }

        private void LogMovementDiagnosticsIfNeeded()
        {
            if (!logMovementDiagnostics || !initialized || Time.time < nextMovementDiagnosticTime)
            {
                return;
            }

            nextMovementDiagnosticTime = Time.time + movementDiagnosticsIntervalSeconds;

            if (!driving && !waitingForArrivalProximity)
            {
                return;
            }

            Debug.Log(
                $"[BUSCONTROLLER] MoveStatus driver={ControlledDriverName}, vehicle={ControlledVehicleName}, targetStop={targetStopId}, " +
                $"distance={FormatMeters(GetTargetDistanceMeters())}, speed={GetVehicleSpeedMS():0.00}m/s, " +
                $"pathPoints={vehicleDriver?.PathPointCount}, remainingPoints={vehicleDriver?.RemainingPathPointCount}, " +
                $"waitingForArrival={waitingForArrivalProximity}, blocked={IsVehicleTemporarilyBlocked}:{TemporaryBlockReason}");
        }

        private float GetVehicleSpeedMS()
        {
            if (vehicleDriver != null)
            {
                return vehicleDriver.CurrentSpeedMS;
            }

            Transform controlledTransform = GetControlledVehicleTransform();
            if (controlledTransform == null)
            {
                return 0f;
            }

            var body = controlledTransform.GetComponent<Rigidbody>();
            if (body == null)
            {
                return 0f;
            }

#if UNITY_6000_0_OR_NEWER
            return body.linearVelocity.magnitude;
#else
            return body.velocity.magnitude;
#endif
        }

        private float GetTargetDistanceMeters()
        {
            return targetStopId > 0 ? GetDistanceToStopMeters(targetStopId) : float.PositiveInfinity;
        }

        private float GetDistanceToStopMeters(int stopId)
        {
            if (!API.IsInitialized())
            {
                return float.PositiveInfinity;
            }

            if (!TryGetStop(stopId, out DRTStop stop))
            {
                return float.PositiveInfinity;
            }

            if (!ResolveControlledVehicle(false))
            {
                return float.PositiveInfinity;
            }

            return GetPlanarDistance(GetControlledVehicleBodyPosition(), GetStopServicePoint(stop));
        }

        private Vector3 GetStopServicePoint(DRTStop stop)
        {
            if (stop == null)
            {
                return Vector3.zero;
            }

            var closestStopWaypoint = API.GetClosestWaypoint(stop.Position);
            if (closestStopWaypoint == null)
            {
                return stop.Position;
            }

            float stopToWaypointDistance = GetPlanarDistance(stop.Position, closestStopWaypoint.Position);
            return stopToWaypointDistance <= stopWaypointSnapDistanceMeters
                ? closestStopWaypoint.Position
                : stop.Position;
        }

        private Vector3 GetControlledVehicleBodyPosition()
        {
            if (vehicleDriver != null)
            {
                return vehicleDriver.BodyPosition;
            }

            Transform controlledTransform = GetControlledVehicleTransform();
            return controlledTransform != null ? controlledTransform.position : Vector3.zero;
        }

        private static float GetPlanarDistance(Vector3 a, Vector3 b)
        {
            a.y = 0f;
            b.y = 0f;
            return Vector3.Distance(a, b);
        }

        private static string FormatMeters(float value)
        {
            return float.IsInfinity(value) ? "n/a" : $"{value:0.00}m";
        }

        private void LogInfo(string message)
        {
            if (SuppressUnityLogsDuringMatrixTeleport)
            {
                return;
            }

            Debug.Log(message);
        }

        private void FinishFailedEpisode(string reason)
        {
            if (!Mathf.Approximately(failurePenalty, 0f))
            {
                nextStopSelector?.RecordExternalPenalty(failurePenalty, reason);
            }

            FinishEpisode(reason);
        }

        private void FinishEpisode(string reason)
        {
            if (episodeFinished)
            {
                return;
            }

            bool completedAllRequests = UsesAllStationRunner
                ? nextStopSelector != null && nextStopSelector.IsAllStationRunComplete
                : !HasUnfinishedOrPendingRequests();
            episodeFinished = true;
            driving = false;
            waitingForArrivalProximity = false;
            CompleteActiveRouteLeg(-1, null);
            RecordVehicleTrace("episode_finished");
            vehicleDriver?.StopAndHold(true);

            LogInfo($"[BUSCONTROLLER] Episode finished. {reason}");

            if (logEpisodeSummary && passengerManager != null && !SuppressUnityLogsDuringMatrixTeleport)
            {
                passengerManager.LogSummary();
            }

            float averageWaitSeconds = passengerManager != null ? passengerManager.GetAverageConfirmedWaitTime() : 0f;
            float averageRideSeconds = passengerManager != null ? passengerManager.GetAverageCompletedRideTime() : 0f;
            float serviceRate = passengerManager != null ? passengerManager.GetServiceRate() : 0f;
            int completedCount = passengerManager != null ? passengerManager.GetCompletedCount() : 0;
            ExportEpisodeCsvIfNeeded(reason, completedAllRequests);
            ExportAllStationMatrixCsvIfNeeded(reason);

            nextStopSelector?.NotifyEpisodeFinished(
                completedAllRequests,
                episodeTravelDistanceMeters,
                averageWaitSeconds,
                averageRideSeconds,
                serviceRate,
                completedCount);
        }

        private void ExportEpisodeCsvIfNeeded(string finishReason, bool completedAllRequests)
        {
            if (episodeCsvExported || !ShouldCollectEpisodeExportData() || passengerManager == null)
            {
                return;
            }

            episodeCsvExported = true;

            try
            {
                DateTime exportTimestamp = DateTime.Now;
                string exportDirectory = GetEpisodeExportDirectory();
                Directory.CreateDirectory(exportDirectory);

                string fileStem = BuildEpisodeExportFileStem();
                string episodeFileName = $"{fileStem}_episode.csv";
                string traceFileName = $"{fileStem}_trace.csv";
                string episodePath = System.IO.Path.Combine(exportDirectory, episodeFileName);
                string tracePath = System.IO.Path.Combine(exportDirectory, traceFileName);
                File.WriteAllText(episodePath, BuildEpisodeCsv(finishReason, completedAllRequests, exportTimestamp), Encoding.UTF8);
                File.WriteAllText(tracePath, BuildTraceCsv(), Encoding.UTF8);

                LogInfo(
                    $"[BUSCONTROLLER] Episode CSV exported. episode={episodeFileName}, trace={traceFileName}, " +
                    $"directory={exportDirectory}");
            }
            catch (Exception exception)
            {
                Debug.LogError($"[BUSCONTROLLER] Failed to export episode CSV. {exception}");
            }
        }

        private void ExportPartialAllStationMatrixCsvIfNeeded(DRTRouteLegRecord completedLeg)
        {
            if (completedLeg == null ||
                !completedLeg.Completed ||
                !UsesAllStationRunner ||
                travelExecutionMode != DRTTravelExecutionMode.PhysicalDrive)
            {
                return;
            }

            ExportAllStationMatrixCsv(
                true,
                $"partial leg {completedLeg.FromStopId}->{completedLeg.ToStopId}",
                false);
        }

        private void ExportAllStationMatrixCsvIfNeeded(string finishReason)
        {
            if (allStationMatrixCsvExported || !UsesAllStationRunner)
            {
                return;
            }

            allStationMatrixCsvExported = true;

            if (travelExecutionMode != DRTTravelExecutionMode.PhysicalDrive)
            {
                Debug.LogWarning("[BUSCONTROLLER] All Station Runner finished, but matrix CSV was not updated because travel mode is not Physical Drive.");
                return;
            }

            bool completedAllStationRun = nextStopSelector != null && nextStopSelector.IsAllStationRunComplete;
            if (!completedAllStationRun)
            {
                Debug.LogWarning($"[BUSCONTROLLER] All Station Runner ended before completion. Saving partial matrix CSV. reason={finishReason}");
            }

            ExportAllStationMatrixCsv(!completedAllStationRun, finishReason, true);
        }

        private void ExportAllStationMatrixCsv(bool allowPartial, string reason, bool finalExport)
        {
            if (UsesPPOVehicleControl)
            {
                ExportAllStationPPOSamplesCsv(reason, finalExport);
                return;
            }

            if (!TryBuildAllStationMatrixCsv(
                    allowPartial,
                    out string matrixCsv,
                    out string samplesCsv,
                    out string buildError,
                    out int measuredPairs,
                    out int updatedPairs,
                    out int preservedPairs,
                    out int totalPairs,
                    out bool hasCompleteMatrix))
            {
                string message = $"[BUSCONTROLLER] All Station Runner matrix CSV skipped. {buildError}";
                if (finalExport)
                {
                    Debug.LogError(message);
                }
                else
                {
                    Debug.LogWarning(message);
                }

                return;
            }

            try
            {
                string samplesPath = GetCurrentAllStationSamplesCsvPath();
                Directory.CreateDirectory(System.IO.Path.GetDirectoryName(samplesPath));
                File.WriteAllText(samplesPath, samplesCsv, Encoding.UTF8);

                string matrixPath = GetTravelTimeMatrixCsvPath();
                Directory.CreateDirectory(System.IO.Path.GetDirectoryName(matrixPath));
                File.WriteAllText(matrixPath, matrixCsv, new UTF8Encoding(false));
                InvalidateTravelTimeMatrix();

#if UNITY_EDITOR
                AssetDatabase.Refresh();
#endif

                if (!hasCompleteMatrix)
                {
                    string message =
                        $"[BUSCONTROLLER] All Station Runner partial matrix CSV saved. Missing pairs are 0 until measured. {buildError}";
                    if (finalExport)
                    {
                        Debug.LogError(message);
                    }
                    else
                    {
                        Debug.LogWarning(message);
                    }
                }

                Debug.Log(
                    $"[BUSCONTROLLER] All Station Runner {(hasCompleteMatrix ? "matrix CSV" : "partial matrix CSV")} " +
                    $"{(finalExport ? "exported" : "partial save")}. " +
                    $"reason={reason}, measured={measuredPairs}/{totalPairs}, updated={updatedPairs}, preserved={preservedPairs}, " +
                    $"matrix={matrixPath}, samples={samplesPath}");
            }
            catch (Exception exception)
            {
                Debug.LogError($"[BUSCONTROLLER] Failed to export All Station Runner matrix CSV. {exception}");
            }
        }

        private void ExportAllStationPPOSamplesCsv(string reason, bool finalExport)
        {
            if (!TryBuildAllStationSamplesCsv(
                    out string samplesCsv,
                    out string buildError,
                    out int measuredPairs,
                    out int totalPairs))
            {
                string message = $"[BUSCONTROLLER] All Station Runner PPO sample CSV skipped. {buildError}";
                if (finalExport)
                {
                    Debug.LogError(message);
                }
                else
                {
                    Debug.LogWarning(message);
                }

                return;
            }

            try
            {
                string exportDirectory = GetEpisodeExportDirectory();
                Directory.CreateDirectory(exportDirectory);
                string samplesPath = System.IO.Path.Combine(exportDirectory, $"{BuildEpisodeExportFileStem()}_ppo_samples.csv");
                File.WriteAllText(samplesPath, samplesCsv, Encoding.UTF8);

#if UNITY_EDITOR
                AssetDatabase.Refresh();
#endif

                Debug.Log(
                    $"[BUSCONTROLLER] All Station Runner PPO sample CSV {(finalExport ? "exported" : "partial save")}. " +
                    $"reason={reason}, measured={measuredPairs}/{totalPairs}, samples={samplesPath}. " +
                    "Travel-time matrix was not overwritten for PPO drive mode.");
            }
            catch (Exception exception)
            {
                Debug.LogError($"[BUSCONTROLLER] Failed to export All Station Runner PPO sample CSV. {exception}");
            }
        }

        private bool TryBuildAllStationSamplesCsv(
            out string samplesCsv,
            out string error,
            out int measuredPairs,
            out int totalPairs)
        {
            samplesCsv = string.Empty;
            error = null;
            measuredPairs = 0;
            totalPairs = 0;

            var sortedStops = stops
                .Where(stop => stop != null)
                .OrderBy(stop => stop.StopId)
                .ToList();
            if (sortedStops.Count < 2)
            {
                error = "At least two stops are required.";
                return false;
            }

            totalPairs = sortedStops.Count * (sortedStops.Count - 1);
            var samplesByPair = new Dictionary<string, List<float>>();
            for (int i = 0; i < routeLegRecords.Count; i++)
            {
                DRTRouteLegRecord leg = routeLegRecords[i];
                if (leg == null ||
                    !leg.Completed ||
                    leg.FromStopId <= 0 ||
                    leg.ToStopId <= 0 ||
                    leg.FromStopId == leg.ToStopId ||
                    leg.ArrivedStopId != leg.ToStopId ||
                    leg.TravelTimeSeconds <= 0f)
                {
                    continue;
                }

                string key = BuildStopPairKey(leg.FromStopId, leg.ToStopId);
                if (!samplesByPair.TryGetValue(key, out List<float> samples))
                {
                    samples = new List<float>();
                    samplesByPair[key] = samples;
                }

                samples.Add(leg.TravelTimeSeconds);
            }

            var samplesBuilder = new StringBuilder();
            samplesBuilder.AppendLine("from_stop_id,to_stop_id,sample_count,matrix_time_seconds,sample_time_seconds");
            for (int row = 0; row < sortedStops.Count; row++)
            {
                int fromStopId = sortedStops[row].StopId;
                for (int column = 0; column < sortedStops.Count; column++)
                {
                    int toStopId = sortedStops[column].StopId;
                    if (fromStopId == toStopId)
                    {
                        continue;
                    }

                    string key = BuildStopPairKey(fromStopId, toStopId);
                    bool hasSample = samplesByPair.TryGetValue(key, out List<float> samples) && samples.Count > 0;
                    if (!hasSample)
                    {
                        continue;
                    }

                    measuredPairs++;
                    samplesBuilder
                        .Append(fromStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                        .Append(toStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                        .Append(samples.Count.ToString(CultureInfo.InvariantCulture)).Append(',')
                        .Append(',')
                        .Append(FormatCsvFloat(Median(samples)))
                        .AppendLine();
                }
            }

            samplesCsv = samplesBuilder.ToString();
            return true;
        }

        private bool TryBuildAllStationMatrixCsv(
            bool allowPartial,
            out string matrixCsv,
            out string samplesCsv,
            out string error,
            out int measuredPairs,
            out int updatedPairs,
            out int preservedPairs,
            out int totalPairs,
            out bool hasCompleteMatrix)
        {
            matrixCsv = string.Empty;
            samplesCsv = string.Empty;
            error = null;
            measuredPairs = 0;
            updatedPairs = 0;
            preservedPairs = 0;
            totalPairs = 0;
            hasCompleteMatrix = false;

            var sortedStops = stops
                .Where(stop => stop != null)
                .OrderBy(stop => stop.StopId)
                .ToList();
            if (sortedStops.Count < 2)
            {
                error = "At least two stops are required.";
                return false;
            }

            totalPairs = sortedStops.Count * (sortedStops.Count - 1);
            Dictionary<string, float> baselineByPair = null;
            string baselineError = null;
            if (allowPartial && !TryLoadExistingMatrixValues(sortedStops, out baselineByPair, out baselineError))
            {
                baselineByPair = null;
            }

            var samplesByPair = new Dictionary<string, List<float>>();
            LoadHistoricalAllStationSampleData(
                GetCurrentAllStationSamplesCsvPath(),
                samplesByPair);

            for (int i = 0; i < routeLegRecords.Count; i++)
            {
                DRTRouteLegRecord leg = routeLegRecords[i];
                if (leg == null ||
                    !leg.Completed ||
                    leg.FromStopId <= 0 ||
                    leg.ToStopId <= 0 ||
                    leg.FromStopId == leg.ToStopId ||
                    leg.ArrivedStopId != leg.ToStopId ||
                    leg.TravelTimeSeconds <= 0f)
                {
                    continue;
                }

                string key = BuildStopPairKey(leg.FromStopId, leg.ToStopId);
                AddSampleValue(samplesByPair, key, leg.TravelTimeSeconds);
            }

            var matrixBuilder = new StringBuilder();
            var samplesBuilder = new StringBuilder();
            samplesBuilder.AppendLine("from_stop_id,to_stop_id,sample_count,matrix_time_seconds,sample_time_seconds");
            string firstMissingPair = null;

            for (int row = 0; row < sortedStops.Count; row++)
            {
                if (row > 0)
                {
                    matrixBuilder.AppendLine();
                }

                int fromStopId = sortedStops[row].StopId;
                for (int column = 0; column < sortedStops.Count; column++)
                {
                    if (column > 0)
                    {
                        matrixBuilder.Append(',');
                    }

                    int toStopId = sortedStops[column].StopId;
                    if (fromStopId == toStopId)
                    {
                        matrixBuilder.Append('0');
                        continue;
                    }

                    string key = BuildStopPairKey(fromStopId, toStopId);
                    bool hasSample = samplesByPair.TryGetValue(key, out List<float> samples) && samples.Count > 0;
                    float matrixSeconds;
                    if (hasSample)
                    {
                        matrixSeconds = Median(samples);
                        measuredPairs++;
                        updatedPairs++;
                    }
                    else if (allowPartial && baselineByPair != null && baselineByPair.TryGetValue(key, out float baselineSeconds))
                    {
                        matrixSeconds = baselineSeconds;
                        preservedPairs++;
                    }
                    else
                    {
                        matrixSeconds = 0f;
                        if (firstMissingPair == null)
                        {
                            firstMissingPair = $"from={fromStopId}, to={toStopId}";
                        }
                    }

                    matrixBuilder.Append(FormatCsvFloat(matrixSeconds));
                    samplesBuilder
                        .Append(fromStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                        .Append(toStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                        .Append((hasSample ? samples.Count : 0).ToString(CultureInfo.InvariantCulture)).Append(',')
                        .Append(FormatCsvFloat(matrixSeconds)).Append(',')
                        .Append(CsvEscape(hasSample ? FormatCsvFloatList(samples) : string.Empty))
                        .AppendLine();
                }
            }

            matrixCsv = matrixBuilder.ToString();
            samplesCsv = samplesBuilder.ToString();
            hasCompleteMatrix = firstMissingPair == null;
            if (hasCompleteMatrix)
            {
                return true;
            }

            error = $"Missing completed route leg sample. {firstMissingPair}";
            if (!string.IsNullOrWhiteSpace(baselineError))
            {
                error += $"; existing matrix unavailable: {baselineError}";
            }

            return true;
        }

        private bool TryLoadExistingMatrixValues(
            IReadOnlyList<DRTStop> sortedStops,
            out Dictionary<string, float> valuesByPair,
            out string error)
        {
            valuesByPair = new Dictionary<string, float>();
            error = null;

            if (sortedStops == null || sortedStops.Count < 2)
            {
                error = "At least two stops are required.";
                return false;
            }

            string csvText = null;
            string matrixPath = GetTravelTimeMatrixCsvPath();
            if (File.Exists(matrixPath))
            {
                csvText = File.ReadAllText(matrixPath, Encoding.UTF8);
            }
            else
            {
                TextAsset csvAsset = Resources.Load<TextAsset>(travelTimeMatrixResourceName);
                if (csvAsset != null)
                {
                    csvText = csvAsset.text;
                }
            }

            if (string.IsNullOrWhiteSpace(csvText))
            {
                error = $"Existing matrix CSV not found. path={matrixPath}, resource={travelTimeMatrixResourceName}";
                return false;
            }

            string[] rawLines = csvText.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var lines = new List<string>();
            for (int i = 0; i < rawLines.Length; i++)
            {
                string line = rawLines[i].Trim().TrimStart('\uFEFF');
                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                lines.Add(line);
            }

            if (lines.Count != sortedStops.Count)
            {
                error = $"Existing matrix must have {sortedStops.Count} rows, but has {lines.Count}.";
                return false;
            }

            for (int row = 0; row < lines.Count; row++)
            {
                string[] columns = lines[row].Split(',');
                if (columns.Length != sortedStops.Count)
                {
                    error = $"Existing matrix row {row + 1} must have {sortedStops.Count} columns, but has {columns.Length}.";
                    return false;
                }

                int fromStopId = sortedStops[row].StopId;
                for (int column = 0; column < columns.Length; column++)
                {
                    int toStopId = sortedStops[column].StopId;
                    if (fromStopId == toStopId)
                    {
                        continue;
                    }

                    string cell = columns[column].Trim();
                    if (string.IsNullOrEmpty(cell))
                    {
                        continue;
                    }

                    if (!float.TryParse(cell, NumberStyles.Float, CultureInfo.InvariantCulture, out float seconds))
                    {
                        error = $"Existing matrix row {row + 1}, column {column + 1} is not a number.";
                        return false;
                    }

                    if (seconds > 0f)
                    {
                        valuesByPair[BuildStopPairKey(fromStopId, toStopId)] = seconds;
                    }
                }
            }

            return true;
        }

        private string GetTravelTimeMatrixCsvPath()
        {
            string resourcesPath = System.IO.Path.Combine(Application.dataPath, "DRT", "Resources");
            string fileName = string.IsNullOrWhiteSpace(travelTimeMatrixResourceName)
                ? "drt_stop_travel_time_matrix"
                : travelTimeMatrixResourceName;
            return System.IO.Path.Combine(resourcesPath, fileName + ".csv");
        }

        private string GetCurrentAllStationSamplesCsvPath()
        {
            string samplesFileName = $"{BuildEpisodeExportFileStem()}_matrix_samples.csv";
            return System.IO.Path.Combine(GetEpisodeExportDirectory(), samplesFileName);
        }

        private bool ShouldCollectEpisodeExportData()
        {
            return exportEpisodeCsvOnEpisodeEnd || UsesAllStationRunner;
        }

        private static bool IsMlAgentsTrainingSession()
        {
            return Academy.Instance != null && Academy.Instance.IsCommunicatorOn;
        }

        private string GetEpisodeExportRootDirectory()
        {
#if UNITY_EDITOR
            string projectRoot = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, ".."));
            return System.IO.Path.Combine(projectRoot, "DRT_Episode_Exports");
#else
            return System.IO.Path.Combine(Application.persistentDataPath, "DRT_Episode_Exports");
#endif
        }

        private string GetEpisodeExportDirectory()
        {
            EnsureEpisodeExportRunTimestamp();
            string policyDirectory = GetNextStopPolicyExportToken();
            string timestampDirectory = episodeExportRunTimestamp.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            return System.IO.Path.Combine(GetEpisodeExportRootDirectory(), policyDirectory, timestampDirectory);
        }

        private void EnsureEpisodeExportRunTimestamp()
        {
            if (hasEpisodeExportRunTimestamp)
            {
                return;
            }

            episodeExportRunTimestamp = DateTime.Now;
            hasEpisodeExportRunTimestamp = true;
        }

        private string BuildEpisodeExportFileStem()
        {
            string modeToken = GetTravelExecutionExportToken();
            string policyToken = GetNextStopPolicyExportToken();
            string scenarioId = demandGenerator != null
                ? demandGenerator.ExportScenarioId
                : Mathf.Max(0, passengerManager != null ? passengerManager.Requests.Count : 0).ToString(CultureInfo.InvariantCulture);
            return SanitizeFileName($"drt_{modeToken}_{policyToken}_scenario_{scenarioId}_ep{episodeIndex:000}");
        }

        private string GetTravelExecutionExportToken()
        {
            if (travelExecutionMode == DRTTravelExecutionMode.MatrixTeleport)
            {
                return "matrix";
            }

            if (travelExecutionMode == DRTTravelExecutionMode.Train)
            {
                return "train";
            }

            switch (physicalDriveMode)
            {
                case DRTPhysicalDriveMode.PPOAutonomous:
                    return "physical_ppo";
                case DRTPhysicalDriveMode.PPOPurePursuit:
                    return "physical_ppo_purepursuit";
                case DRTPhysicalDriveMode.NoisyGley:
                    return "physical_noisy_gley";
                default:
                    return "physical_gley";
            }
        }

        private string GetNextStopPolicyExportToken()
        {
            switch (NextStopPolicy)
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
                default:
                    return "train";
            }
        }

        private string BuildEpisodeCsv(string finishReason, bool completedAllRequests, DateTime exportTimestamp)
        {
            var builder = new StringBuilder();
            builder.AppendLine("section,key,value");
            AppendEpisodeRow(builder, "metadata", "schema_version", "1");
            AppendEpisodeRow(builder, "metadata", "generated_at", exportTimestamp.ToString("o", CultureInfo.InvariantCulture));
            AppendEpisodeRow(builder, "summary", "episode_index", episodeIndex.ToString(CultureInfo.InvariantCulture));
            AppendEpisodeRow(builder, "summary", "travel_mode", GetTravelExecutionExportToken());
            AppendEpisodeRow(builder, "summary", "travel_execution_mode", TravelExecutionModeName);
            AppendEpisodeRow(builder, "summary", "physical_drive_mode", PhysicalDriveModeName);
            AppendEpisodeRow(builder, "summary", "ppo_drive_policy", PPODrivePolicyName);
            AppendEpisodeRow(builder, "summary", "ppo_speed_limit_enabled", usePPOSpeedLimit ? "1" : "0");
            AppendEpisodeRow(builder, "summary", "ppo_speed_limit_mps", FormatCsvFloat(ppoSpeedLimitMetersPerSecond));
            AppendEpisodeRow(builder, "summary", "ppo_training_route_enabled", UsesPPOTrainingRouteEpisode ? "1" : "0");
            AppendEpisodeRow(builder, "summary", "ppo_training_all_station_route_cycle_enabled", usePPOTrainingAllStationRouteCycle ? "1" : "0");
            AppendEpisodeRow(builder, "summary", "ppo_training_route_start_stop_id", PPOTrainingRouteStartStopId.ToString(CultureInfo.InvariantCulture));
            AppendEpisodeRow(builder, "summary", "ppo_training_route_end_stop_id", PPOTrainingRouteEndStopId.ToString(CultureInfo.InvariantCulture));
            AppendEpisodeRow(builder, "summary", "policy", GetNextStopPolicyExportToken());
            AppendEpisodeRow(builder, "summary", "next_stop_policy", NextStopPolicyName);
            AppendEpisodeRow(builder, "summary", "scenario_id", demandGenerator != null ? demandGenerator.ExportScenarioId : string.Empty);
            AppendEpisodeRow(builder, "summary", "scenario_description", demandGenerator != null ? demandGenerator.LoadedScenarioDescription : string.Empty);
            AppendEpisodeRow(builder, "summary", "finish_reason", finishReason);
            AppendEpisodeRow(builder, "summary", "completed_all_requests", completedAllRequests ? "1" : "0");
            AppendEpisodeRow(builder, "summary", "episode_time_seconds", FormatCsvFloat(episodeTimeSeconds));
            AppendEpisodeRow(builder, "summary", "episode_distance_meters", FormatCsvFloat(episodeTravelDistanceMeters));
            AppendEpisodeRow(builder, "summary", "total_passengers", passengerManager != null ? passengerManager.Requests.Count.ToString(CultureInfo.InvariantCulture) : "0");
            AppendEpisodeRow(builder, "summary", "completed_passengers", passengerManager != null ? passengerManager.GetCompletedCount().ToString(CultureInfo.InvariantCulture) : "0");
            AppendEpisodeRow(builder, "summary", "service_rate", passengerManager != null ? FormatCsvFloat(passengerManager.GetServiceRate()) : "0");
            AppendEpisodeRow(builder, "summary", "average_wait_seconds", passengerManager != null ? FormatCsvFloat(passengerManager.GetAverageConfirmedWaitTime()) : "0");
            AppendEpisodeRow(builder, "summary", "average_ride_seconds", passengerManager != null ? FormatCsvFloat(passengerManager.GetAverageCompletedRideTime()) : "0");
            builder.AppendLine(",");
            AppendRouteLegsCsv(builder);
            builder.AppendLine(",");
            AppendPassengersCsv(builder);
            return builder.ToString();
        }

        private void AppendRouteLegsCsv(StringBuilder builder)
        {
            builder.AppendLine("section,leg_index,mode,from_stop_id,to_stop_id,arrived_stop_id,completed,departure_time_seconds,arrival_time_seconds,travel_time_seconds,leg_distance_meters,cumulative_distance_meters,planned_path_distance_meters,path_waypoint_count,boarded_count,dropped_off_count,boarded_passenger_ids,dropped_off_passenger_ids,waiting_count,on_board_count,completed_passenger_count");
            foreach (var leg in routeLegRecords)
            {
                builder
                    .Append("route_leg").Append(',')
                    .Append(leg.LegIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(CsvEscape(leg.Mode)).Append(',')
                    .Append(leg.FromStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(leg.ToStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(FormatCsvIntOrBlank(leg.ArrivedStopId)).Append(',')
                    .Append(leg.Completed ? "1" : "0").Append(',')
                    .Append(FormatCsvFloat(leg.DepartureTimeSeconds)).Append(',')
                    .Append(FormatCsvFloat(leg.ArrivalTimeSeconds)).Append(',')
                    .Append(FormatCsvFloat(leg.TravelTimeSeconds)).Append(',')
                    .Append(FormatCsvFloat(leg.LegDistanceMeters)).Append(',')
                    .Append(FormatCsvFloat(leg.ArrivalCumulativeDistanceMeters)).Append(',')
                    .Append(FormatCsvFloat(leg.PlannedPathDistanceMeters)).Append(',')
                    .Append(leg.PathWaypointCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(leg.BoardedCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(leg.DroppedOffCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(CsvEscape(FormatPassengerIdList(GetBoardedPassengerIds(leg)))).Append(',')
                    .Append(CsvEscape(FormatPassengerIdList(GetDroppedOffPassengerIds(leg)))).Append(',')
                    .Append(leg.WaitingCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(leg.OnBoardCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(leg.CompletedPassengerCount.ToString(CultureInfo.InvariantCulture))
                    .AppendLine();
            }
        }

        private void AppendPassengersCsv(StringBuilder builder)
        {
            builder.AppendLine("section,passenger_id,origin_stop_id,destination_stop_id,request_time_seconds,status,pickup_time_seconds,dropoff_time_seconds,actual_pickup_stop_id,actual_dropoff_stop_id,wait_time_seconds,ride_time_seconds,total_service_time_seconds");
            foreach (var request in passengerManager.Requests.OrderBy(request => request.PassengerId))
            {
                builder
                    .Append("passenger").Append(',')
                    .Append(request.PassengerId.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(request.OriginStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(request.DestinationStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(FormatCsvFloat(request.RequestTimeSeconds)).Append(',')
                    .Append(CsvEscape(request.Status.ToString())).Append(',')
                    .Append(FormatCsvFloatOrBlank(request.PickupTimeSeconds)).Append(',')
                    .Append(FormatCsvFloatOrBlank(request.DropoffTimeSeconds)).Append(',')
                    .Append(FormatCsvIntOrBlank(request.ActualPickupStopId)).Append(',')
                    .Append(FormatCsvIntOrBlank(request.ActualDropoffStopId)).Append(',')
                    .Append(FormatCsvFloat(request.GetWaitTime(episodeTimeSeconds))).Append(',')
                    .Append(FormatCsvFloat(request.GetRideTime(episodeTimeSeconds))).Append(',')
                    .Append(FormatCsvFloat(request.GetTotalServiceTime(episodeTimeSeconds)))
                    .AppendLine();
            }
        }

        private string BuildTraceCsv()
        {
            var builder = new StringBuilder();
            builder.AppendLine("leg_index,from_stop_id,to_stop_id,arrived_stop_id,completed,departure_time_seconds,arrival_time_seconds,travel_time_seconds,leg_distance_meters,cumulative_distance_meters,planned_path_distance_meters,path_waypoint_count,boarded_passenger_ids,dropped_off_passenger_ids,arrival_x,arrival_y,arrival_z,blocked,block_reason");

            foreach (var leg in routeLegRecords)
            {
                DRTVehicleTraceRecord arrivalTrace = FindNearestVehicleTrace(leg.ArrivalTimeSeconds);
                builder
                    .Append(leg.LegIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(leg.FromStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(leg.ToStopId.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(FormatCsvIntOrBlank(leg.ArrivedStopId)).Append(',')
                    .Append(leg.Completed ? "1" : "0").Append(',')
                    .Append(FormatCsvFloat(leg.DepartureTimeSeconds)).Append(',')
                    .Append(FormatCsvFloat(leg.ArrivalTimeSeconds)).Append(',')
                    .Append(FormatCsvFloat(leg.TravelTimeSeconds)).Append(',')
                    .Append(FormatCsvFloat(leg.LegDistanceMeters)).Append(',')
                    .Append(FormatCsvFloat(leg.ArrivalCumulativeDistanceMeters)).Append(',')
                    .Append(FormatCsvFloat(leg.PlannedPathDistanceMeters)).Append(',')
                    .Append(leg.PathWaypointCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(CsvEscape(FormatPassengerIdList(GetBoardedPassengerIds(leg)))).Append(',')
                    .Append(CsvEscape(FormatPassengerIdList(GetDroppedOffPassengerIds(leg)))).Append(',')
                    .Append(arrivalTrace != null ? FormatCsvFloat(arrivalTrace.Position.x) : string.Empty).Append(',')
                    .Append(arrivalTrace != null ? FormatCsvFloat(arrivalTrace.Position.y) : string.Empty).Append(',')
                    .Append(arrivalTrace != null ? FormatCsvFloat(arrivalTrace.Position.z) : string.Empty).Append(',')
                    .Append(arrivalTrace != null && arrivalTrace.Blocked ? "1" : "0").Append(',')
                    .Append(CsvEscape(arrivalTrace != null ? arrivalTrace.BlockReason : string.Empty))
                    .AppendLine();
            }

            return builder.ToString();
        }

        private DRTVehicleTraceRecord FindNearestVehicleTrace(float episodeTime)
        {
            if (vehicleTraceRecords.Count == 0)
            {
                return null;
            }

            DRTVehicleTraceRecord nearestTrace = null;
            float nearestDelta = float.PositiveInfinity;
            for (int i = 0; i < vehicleTraceRecords.Count; i++)
            {
                DRTVehicleTraceRecord trace = vehicleTraceRecords[i];
                float delta = Mathf.Abs(trace.EpisodeTimeSeconds - episodeTime);
                if (delta < nearestDelta)
                {
                    nearestTrace = trace;
                    nearestDelta = delta;
                }
            }

            return nearestTrace;
        }

        private List<int> GetBoardedPassengerIds(DRTRouteLegRecord leg)
        {
            if (passengerManager == null || leg.ArrivedStopId <= 0)
            {
                return new List<int>();
            }

            return passengerManager.Requests
                .Where(request => request != null &&
                                  request.ActualPickupStopId == leg.ArrivedStopId &&
                                  IsSameEpisodeTime(request.PickupTimeSeconds, leg.ArrivalTimeSeconds))
                .Select(request => request.PassengerId)
                .OrderBy(passengerId => passengerId)
                .ToList();
        }

        private List<int> GetDroppedOffPassengerIds(DRTRouteLegRecord leg)
        {
            if (passengerManager == null || leg.ArrivedStopId <= 0)
            {
                return new List<int>();
            }

            return passengerManager.Requests
                .Where(request => request != null &&
                                  request.ActualDropoffStopId == leg.ArrivedStopId &&
                                  IsSameEpisodeTime(request.DropoffTimeSeconds, leg.ArrivalTimeSeconds))
                .Select(request => request.PassengerId)
                .OrderBy(passengerId => passengerId)
                .ToList();
        }

        private static bool IsSameEpisodeTime(float recordedTime, float currentEpisodeTime)
        {
            return recordedTime >= 0f && Mathf.Abs(recordedTime - currentEpisodeTime) <= 0.05f;
        }

        private static string SanitizeFileName(string fileName)
        {
            string sanitized = fileName;
            foreach (char invalidChar in System.IO.Path.GetInvalidFileNameChars())
            {
                sanitized = sanitized.Replace(invalidChar, '_');
            }

            return sanitized;
        }

        private static void AppendEpisodeRow(StringBuilder builder, string section, string key, string value)
        {
            builder
                .Append(CsvEscape(section))
                .Append(',')
                .Append(CsvEscape(key))
                .Append(',')
                .Append(CsvEscape(value))
                .AppendLine();
        }

        private static string FormatPassengerIdList(List<int> passengerIds)
        {
            return passengerIds == null || passengerIds.Count == 0
                ? string.Empty
                : string.Join("|", passengerIds);
        }

        private static string FormatCsvFloatList(List<float> values)
        {
            if (values == null || values.Count == 0)
            {
                return string.Empty;
            }

            return string.Join("|", values.Select(FormatCsvFloat));
        }

        private static float Median(List<float> values)
        {
            if (values == null || values.Count == 0)
            {
                return 0f;
            }

            var sortedValues = new List<float>(values);
            sortedValues.Sort();
            int middle = sortedValues.Count / 2;
            return sortedValues.Count % 2 == 1
                ? sortedValues[middle]
                : (sortedValues[middle - 1] + sortedValues[middle]) * 0.5f;
        }

        private static string BuildStopPairKey(int fromStopId, int toStopId)
        {
            return $"{fromStopId}->{toStopId}";
        }

        private static string FormatCsvFloat(float value)
        {
            return value.ToString("0.###", CultureInfo.InvariantCulture);
        }

        private static string FormatCsvFloatOrBlank(float value)
        {
            return value >= 0f ? FormatCsvFloat(value) : string.Empty;
        }

        private static string FormatCsvIntOrBlank(int value)
        {
            return value > 0 ? value.ToString(CultureInfo.InvariantCulture) : string.Empty;
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

        private void OnDrawGizmos()
        {
            if (!showAssignedPath || !showAssignedPathGizmos || assignedPathPoints == null || assignedPathPoints.Count < 2)
            {
                return;
            }

            Gizmos.color = assignedPathColor;
            for (int i = 1; i < assignedPathPoints.Count; i++)
            {
                Gizmos.DrawLine(assignedPathPoints[i - 1], assignedPathPoints[i]);
            }

            Gizmos.color = assignedPathWaypointColor;
            for (int i = 0; i < assignedPathPoints.Count; i++)
            {
                Gizmos.DrawSphere(assignedPathPoints[i], assignedPathWaypointRadius);
            }
        }

        private void OnValidate()
        {
            vehicleIndex = Mathf.Max(0, vehicleIndex);
            startStopId = Mathf.Max(1, startStopId);
            dwellSeconds = Mathf.Max(0f, dwellSeconds);
            arrivalDistanceMeters = Mathf.Max(0.05f, arrivalDistanceMeters);
            stopWaypointSnapDistanceMeters = Mathf.Max(0.05f, stopWaypointSnapDistanceMeters);
            arrivalWaitTimeoutSeconds = Mathf.Max(0.5f, arrivalWaitTimeoutSeconds);
            controlledVehicleSpeedMultiplier = Mathf.Max(0.1f, controlledVehicleSpeedMultiplier);
            gleyControlledVehicleSpeedMultiplier = Mathf.Clamp(gleyControlledVehicleSpeedMultiplier, 0.1f, 2f);
            playerWaypointReachDistanceMeters = Mathf.Max(0.5f, playerWaypointReachDistanceMeters);
            noisyGleyLateralNoise = Mathf.Clamp(noisyGleyLateralNoise, 0f, MaxNoisyGleyLateralNoise);
            noisyGleySpeedNoise = Mathf.Clamp(noisyGleySpeedNoise, 0f, 0.5f);
            noisyGleyNoiseStrength = Mathf.Clamp(noisyGleyNoiseStrength, 0f, MaxNoisyGleyNoiseStrength);
            noisyGleyNoiseFrequency = Mathf.Clamp(noisyGleyNoiseFrequency, 0.01f, 5f);
            noisyGleyNoiseIrregularity = Mathf.Clamp01(noisyGleyNoiseIrregularity);
            useGleyVehicleControlInPhysicalDrive = IsGleyBasedPhysicalDriveMode;
            ppoSpeedLimitMetersPerSecond = Mathf.Max(0.5f, ppoSpeedLimitMetersPerSecond);
            ppoPurePursuitMinLookaheadMeters = Mathf.Max(0.1f, ppoPurePursuitMinLookaheadMeters);
            ppoPurePursuitMaxLookaheadMeters = Mathf.Max(
                ppoPurePursuitMinLookaheadMeters + 0.1f,
                ppoPurePursuitMaxLookaheadMeters);
            ppoPurePursuitZeroActionLookaheadNormalized = Mathf.Clamp01(ppoPurePursuitZeroActionLookaheadNormalized);
            ppoPurePursuitMinTargetSpeedRatio = Mathf.Clamp01(ppoPurePursuitMinTargetSpeedRatio);
            ppoPurePursuitMaxTargetSpeedRatio = Mathf.Clamp(
                ppoPurePursuitMaxTargetSpeedRatio,
                ppoPurePursuitMinTargetSpeedRatio,
                1f);
            ppoPurePursuitZeroActionSpeedNormalized = Mathf.Clamp01(ppoPurePursuitZeroActionSpeedNormalized);
            ppoPurePursuitThrottleInputSmoothing = Mathf.Max(0.1f, ppoPurePursuitThrottleInputSmoothing);
            ppoPurePursuitSteeringInputSmoothing = Mathf.Max(0.1f, ppoPurePursuitSteeringInputSmoothing);
            ppoPurePursuitCurvatureSmoothingBeta = Mathf.Clamp(ppoPurePursuitCurvatureSmoothingBeta, 0.01f, 1f);
            ppoPurePursuitSpeedRewardPerSecond = Mathf.Max(0f, ppoPurePursuitSpeedRewardPerSecond);
            ppoPurePursuitProgressRewardPerMeter = Mathf.Max(0f, ppoPurePursuitProgressRewardPerMeter);
            ppoPurePursuitDestinationProgressRewardPerMeter = Mathf.Max(0f, ppoPurePursuitDestinationProgressRewardPerMeter);
            ppoPurePursuitDestinationReward = Mathf.Max(0f, ppoPurePursuitDestinationReward);
            ppoPurePursuitLookaheadChangePenaltyPerMeter = Mathf.Max(0f, ppoPurePursuitLookaheadChangePenaltyPerMeter);
            ppoPurePursuitLateralErrorPenaltyPerMeter = Mathf.Max(0f, ppoPurePursuitLateralErrorPenaltyPerMeter);
            ppoPurePursuitLocalLateralVelocityPenaltyPerMeter =
                Mathf.Max(0f, ppoPurePursuitLocalLateralVelocityPenaltyPerMeter);
            ppoPurePursuitLocalLateralVelocityCurvatureGain =
                Mathf.Max(0f, ppoPurePursuitLocalLateralVelocityCurvatureGain);
            ppoPurePursuitHeadingErrorPenaltyPerMeter = Mathf.Max(0f, ppoPurePursuitHeadingErrorPenaltyPerMeter);
            ppoPurePursuitOverspeedPenaltyPerSecond = Mathf.Max(0f, ppoPurePursuitOverspeedPenaltyPerSecond);
            ppoPurePursuitMaxCrossTrackErrorMeters = Mathf.Max(0.1f, ppoPurePursuitMaxCrossTrackErrorMeters);
            ppoPurePursuitHardCrossTrackLimitMeters = Mathf.Max(0.5f, ppoPurePursuitHardCrossTrackLimitMeters);
            ppoPurePursuitAssignedRouteExitPenaltyMagnitude = Mathf.Max(0f, ppoPurePursuitAssignedRouteExitPenaltyMagnitude);
            ppoPurePursuitNoMovementTimeoutRealSeconds = Mathf.Max(0.5f, ppoPurePursuitNoMovementTimeoutRealSeconds);
            ppoPurePursuitMinimumMovementMeters = Mathf.Max(0.01f, ppoPurePursuitMinimumMovementMeters);
            vehicleTraceSampleIntervalSeconds = Mathf.Max(0.1f, vehicleTraceSampleIntervalSeconds);
            matrixNominalSpeedMetersPerSecond = Mathf.Max(0.1f, matrixNominalSpeedMetersPerSecond);
            enabledTrafficDensity = Mathf.Clamp(enabledTrafficDensity, 0, MaxBackgroundTrafficDensity);
            backgroundTrafficSafeGapMeters = Mathf.Max(0.5f, backgroundTrafficSafeGapMeters);
            backgroundTrafficMinTriggerLengthMeters = Mathf.Max(1f, backgroundTrafficMinTriggerLengthMeters);
            backgroundTrafficMaxTriggerLengthMeters = Mathf.Max(
                backgroundTrafficMinTriggerLengthMeters,
                backgroundTrafficMaxTriggerLengthMeters);
            backgroundTrafficBrakeTimeSeconds = Mathf.Max(0.5f, backgroundTrafficBrakeTimeSeconds);
            backgroundTrafficSpeedLimitMetersPerSecond = Mathf.Max(0f, backgroundTrafficSpeedLimitMetersPerSecond);
            episodeLengthSeconds = Mathf.Max(1f, episodeLengthSeconds);
            simulationSecondsPerRealSecond = Mathf.Max(0.01f, simulationSecondsPerRealSecond);
            ppoTrainingRouteStartStopId = Mathf.Max(1, ppoTrainingRouteStartStopId);
            ppoTrainingRouteEndStopId = Mathf.Max(1, ppoTrainingRouteEndStopId);
            movementDiagnosticsIntervalSeconds = Mathf.Max(0.25f, movementDiagnosticsIntervalSeconds);
            noMovementTimeoutRealSeconds = Mathf.Max(1f, noMovementTimeoutRealSeconds);
            trafficBlockTimeoutRealSeconds = Mathf.Max(0f, trafficBlockTimeoutRealSeconds);
            minimumVehicleMovementMeters = Mathf.Max(0.01f, minimumVehicleMovementMeters);
            maxRoadWaypointDistanceMeters = Mathf.Max(0f, maxRoadWaypointDistanceMeters);
            assignedPathLineWidth = Mathf.Max(0.01f, assignedPathLineWidth);
            assignedPathWaypointRadius = Mathf.Max(0.01f, assignedPathWaypointRadius);
            assignedPathVerticalOffset = Mathf.Max(0f, assignedPathVerticalOffset);
            ApplyAssignedPathLineRenderer();
        }

        private sealed class DRTRouteLegRecord
        {
            public int LegIndex;
            public string Mode;
            public int FromStopId;
            public int ToStopId;
            public int ArrivedStopId;
            public bool Completed;
            public float DepartureTimeSeconds;
            public float ArrivalTimeSeconds;
            public float TravelTimeSeconds;
            public float LegDistanceMeters;
            public float DepartureCumulativeDistanceMeters;
            public float ArrivalCumulativeDistanceMeters;
            public float PlannedPathDistanceMeters;
            public int PathWaypointCount;
            public int BoardedCount;
            public int DroppedOffCount;
            public int WaitingCount;
            public int OnBoardCount;
            public int CompletedPassengerCount;
        }

        private sealed class DRTVehicleTraceRecord
        {
            public int SampleIndex;
            public string EventName;
            public float EpisodeTimeSeconds;
            public int CurrentStopId;
            public int TargetStopId;
            public Vector3 Position;
            public float CumulativeDistanceMeters;
            public float SpeedMetersPerSecond;
            public bool Blocked;
            public string BlockReason;
        }

        private void LogStopMap()
        {
            if (stops.Count == 0)
            {
                return;
            }

            string stopMap = string.Join(
                "; ",
                stops.Select(stop => $"S{stop.StopId}={stop.name}@{FormatVector(stop.Position)}"));
            Debug.Log($"[BUSCONTROLLER] StopMap {stopMap}");
        }

        private static bool TryParseStopIdFromName(string objectName, out int stopId)
        {
            stopId = -1;
            if (string.IsNullOrEmpty(objectName))
            {
                return false;
            }

            int end = objectName.Length - 1;
            while (end >= 0 && !char.IsDigit(objectName[end]))
            {
                end--;
            }

            if (end < 0)
            {
                return false;
            }

            int start = end;
            while (start >= 0 && char.IsDigit(objectName[start]))
            {
                start--;
            }

            string numberText = objectName.Substring(start + 1, end - start);
            return int.TryParse(numberText, out stopId) && stopId >= 1;
        }

        private static string FormatVector(Vector3 value)
        {
            return $"({value.x:0.00},{value.y:0.00},{value.z:0.00})";
        }
    }
}
