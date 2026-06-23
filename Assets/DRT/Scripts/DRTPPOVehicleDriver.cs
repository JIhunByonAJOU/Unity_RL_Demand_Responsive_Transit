using System;
using System.Collections.Generic;
using Gley.TrafficSystem;
using Gley.UrbanSystem;
using Unity.Barracuda;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

using VehicleTypes = Gley.TrafficSystem.User.VehicleTypes;

namespace DRT
{
    [AddComponentMenu("DRT/DRT PPO Vehicle Driver")]
    [RequireComponent(typeof(BehaviorParameters))]
    public class DRTPPOVehicleDriver : Agent, IDRTVehicleDriver
    {
        private const string BehaviorName = "DRTDrivePPO";
        private const int GlobalObservationCount = 10;
        private const int LookaheadWaypointCount = 5;
        private const int ObservationsPerWaypoint = 8;
        private const int RayCount = 9;
        private const int ObservationsPerRay = 3;
        private const float MaxWaypointSpeedKmh = 100f;
        private const float MaxLaneWidthMeters = 8f;
        private const float SteeringSaturationThreshold = 0.95f;
        private const float MinimumSteeringInputSmoothing = 20f;

        [SerializeField] private PlayerCar playerCar;
        [SerializeField] private Rigidbody vehicleRigidbody;
        [SerializeField] private Transform bodyTransform;
        [SerializeField] private Transform vehicleRoot;
        [SerializeField] private VehicleTypes vehicleType = VehicleTypes.Car;
        [SerializeField] private float speedMultiplier = 1f;
        [SerializeField] private float waypointReachDistanceMeters = 6f;
        [SerializeField] private float finalReachDistanceMeters = 2.5f;
        [SerializeField] private float destinationTailDistanceMeters = 12f;
        [SerializeField] private float referenceWaypointSpacingMeters = 1f;
        [SerializeField] private float referenceWaypointPassDistanceMeters = 0.5f;

        [Header("Policy")]
        [SerializeField, InspectorName("Mode")] private DRTPPODrivePolicy drivePolicy = DRTPPODrivePolicy.MLAgentsTraining;
        [SerializeField, InspectorName("ONNX Model")] private NNModel onnxInferenceModel;
        [SerializeField] private InferenceDevice onnxInferenceDevice = InferenceDevice.Default;

        [Header("Control")]
        [SerializeField] private float baseCruiseSpeedMetersPerSecond = 5f;
        [SerializeField, InspectorName("Use Speed Limit")] private bool usePolicySpeedLimit = true;
        [SerializeField] private float maxPolicySpeedMetersPerSecond = 5f;
        [SerializeField] private float speedLimitBrakeInput = -0.45f;
        [SerializeField] private float curveSpeedBrakeInput = -0.85f;
        [SerializeField] private float curveSpeedThrottleCutMarginMetersPerSecond = 0.1f;
        [SerializeField] private float curveSpeedFullBrakeOverspeedMetersPerSecond = 1.5f;
        [HideInInspector, SerializeField] private bool endEpisodeOnDestinationReached = true;
        [SerializeField] private float maxObservationSpeedMetersPerSecond = 12f;
        [SerializeField] private float maxSteeringAngleForFullInput = 45f;
        [SerializeField] private float hardTurnAngle = 75f;
        [SerializeField] private float slowDownDistanceMeters = 10f;
        [SerializeField] private float lookAheadTimeSeconds = 0.35f;
        [SerializeField] private float minLookAheadMeters = 4f;
        [SerializeField] private float maxLookAheadMeters = 16f;
        [SerializeField] private float steeringInputSmoothing = 20f;
        [SerializeField] private float throttleInputSmoothing = 6f;
        [SerializeField] private float destinationApproachSpeedMetersPerSecond = 0.9f;
        [SerializeField] private float destinationStopDistanceMultiplier = 2f;
        [SerializeField] private float destinationApproachBrakeInput = -0.65f;
        [SerializeField] private float destinationApproachCreepThrottle = 0.25f;
        [SerializeField] private float destinationApproachRecoveryThrottle = 0.35f;

        [Header("Observation")]
        [SerializeField] private float maxObservationDistanceMeters = 80f;
        [SerializeField] private float maxCrossTrackErrorMeters = 6f;
        [SerializeField] private float firstLookaheadObservationMeters = 2f;
        [SerializeField] private float lookaheadObservationSpacingMeters = 4f;
        [SerializeField] private float rayLengthMeters = 25f;
        [SerializeField] private float rayHeightMeters = 0.8f;
        [SerializeField] private LayerMask rayLayerMask = ~0;

        [Header("Reward")]
        [SerializeField] private float pathProgressRewardPerMeter = 0.08f;
        [SerializeField] private float destinationProgressRewardPerMeter = 0.002f;
        [SerializeField] private float headingAlignmentReward = 0.03f;
        [SerializeField] private float waypointHeadingReward = 0.02f;
        [SerializeField] private float curvePenalty = -0.03f;
        [SerializeField] private float crossTrackPenalty = -0.12f;
        [SerializeField] private float steeringCorrectionReward = 0f;
        [SerializeField] private float waypointPassedReward = 0.05f;
        [SerializeField] private float destinationReward = 2f;
        [SerializeField] private float collisionPenalty = -2f;
        [SerializeField] private float assignedRouteExitPenalty = -1.5f;
        [SerializeField] private float referenceFaultPenalty = -1f;
        [SerializeField] private float reversePenalty = -0.5f;
        [SerializeField] private float overspeedPenaltyPerSecond = 0f;
        [SerializeField] private float overspeedMarginMetersPerSecond = 0.5f;
        [SerializeField] private float curveOverspeedPenaltyMultiplier = 1f;
        [SerializeField] private float curveOverspeedMarginMetersPerSecond = 0.1f;
        [SerializeField, Range(0f, 1f)] private float curveMinimumSpeedFactor = 0.5f;
        [SerializeField] private float frontCurveLookaheadMeters = 20f;
        [SerializeField] private float curvePreviewLookaheadMeters = 30f;
        [SerializeField] private float frontCurveSampleSpacingMeters = 2f;
        [SerializeField, Range(0.1f, 1f)] private float curveAllowedSpeedSafetyFactor = 0.75f;
        [SerializeField] private float curveLateralAccelerationLimitMetersPerSecondSquared = 2f;
        [SerializeField] private float curveComfortableDecelerationMetersPerSecondSquared = 2f;
        [SerializeField] private float frontVehicleClearanceMeters = 3f;
        [SerializeField] private float sideVehicleClearanceMeters = 1f;
        [SerializeField] private float vehicleClearanceScanRadiusMeters = 12f;
        [SerializeField] private float sideVehicleLongitudinalWindowMeters = 6f;
        [SerializeField] private LayerMask vehicleClearanceLayerMask = ~0;
        [SerializeField] private float frontVehiclePenaltyPerSecond = 0f;
        [SerializeField] private float sideVehiclePenaltyPerSecond = 0f;
        [SerializeField] private float unblockedIdlePenaltyPerSecond = 0f;
        [SerializeField] private float unblockedIdleGraceSeconds = 1.5f;
        [SerializeField] private float idleSpeedThresholdMetersPerSecond = 0.2f;
        [SerializeField] private float idleProgressThresholdMetersPerSecond = 0.1f;
        [SerializeField] private float idleProgressWindowSeconds = 1f;
        [SerializeField] private float idleFrontBlockClearanceMeters = 5f;
        [SerializeField] private float idleDestinationExemptionMeters = 5f;
        [SerializeField] private float lateralAccelerationFreeMetersPerSecondSquared = 1.5f;
        [SerializeField] private float longitudinalAccelerationFreeMetersPerSecondSquared = 2f;
        [SerializeField] private float longitudinalJerkFreeMetersPerSecondCubed = 4f;
        [SerializeField] private float localLateralVelocityFreeMetersPerSecond = 0.25f;
        [SerializeField] private float lateralAccelerationPenaltyPerSecond = 0f;
        [SerializeField] private float longitudinalAccelerationPenaltyPerSecond = 0f;
        [SerializeField] private float longitudinalJerkPenaltyPerSecond = 0f;
        [SerializeField] private float localLateralVelocityPenaltyPerSecond = 0.12f;
        [SerializeField] private float lateralOscillationVelocityThresholdMetersPerSecond = 0.25f;
        [SerializeField] private float lateralOscillationFlipWindowSeconds = 1f;
        [SerializeField] private float lateralOscillationPenaltyPerFlip = 0.03f;
        [SerializeField, Range(0.01f, 1f)] private float motionSmoothingFactor = 0.2f;

        [Header("Safety")]
        [SerializeField] private float noProgressTimeoutRealSeconds = 30f;
        [SerializeField] private float minimumMovementMeters = 0.25f;
        [SerializeField] private float hardCrossTrackLimitMeters = 4f;
        [SerializeField] private float reverseGraceSeconds = 2f;

        private readonly List<int> pathWaypointIndexes = new List<int>();
        private readonly List<TrafficWaypoint> pathWaypoints = new List<TrafficWaypoint>();
        private readonly List<Vector3> pathPoints = new List<Vector3>();
        private readonly List<float> pathCumulativeDistances = new List<float>();
        private readonly Queue<float> idleProgressWindowSamples = new Queue<float>();
        private Collider[] ownColliders;
        private int targetPointIndex;
        private Vector3 finalDestination;
        private bool driving;
        private float targetSteeringInput;
        private float targetThrottleInput;
        private float currentSteeringInput;
        private float currentThrottleInput;
        private float lastPathProgressMeters;
        private float lastDestinationDistance;
        private int lastFrenetSegmentIndex;
        private Vector3 lastMovementPosition;
        private float lastMovementRealtime;
        private float reverseSeconds;
        private float previousYawDegrees;
        private float yawRateDegreesPerSecond;
        private Vector3 previousVelocity;
        private Vector3 smoothedLocalAcceleration;
        private float previousSmoothedLongitudinalAcceleration;
        private float smoothedLongitudinalJerk;
        private int previousLateralVelocitySign;
        private float lateralOscillationClockSeconds;
        private float lastLateralVelocityFlipSeconds;
        private float lastLateralVelocitySignSeconds;
        private float unblockedIdleSeconds;
        private float idleProgressWindowDistanceMeters;
        private float idleProgressWindowDurationSeconds;
        private bool hasMotionSample;
        private bool criticalFault;
        private string criticalFaultReason = string.Empty;
        private bool warnedSharedBehaviorHost;
        private bool destinationReachedPending;
        private PlayerCar steeringCapacityOwner;
        private float originalPlayerMaxSteeringAngle;
        private bool hasOriginalPlayerMaxSteeringAngle;
        private bool steeringCapacityOverrideActive;
        private float episodeAdeSumMeters;
        private float episodeHeadingErrorSumDegrees;
        private float episodeMaxAdeMeters;
        private float episodeMaxHeadingErrorDegrees;
        private int episodeTrackingMetricSamples;

        private int ObservationSize =>
            GlobalObservationCount +
            LookaheadWaypointCount * ObservationsPerWaypoint +
            RayCount * ObservationsPerRay;

        public bool IsDriving => driving;
        public VehicleTypes VehicleType => vehicleType;
        public Transform VehicleTransform => GetVehicleRoot();
        public string VehicleName => VehicleTransform != null ? VehicleTransform.name : name;
        public int PathPointCount => pathPoints.Count;
        public int RemainingPathPointCount => driving ? Mathf.Max(0, pathPoints.Count - targetPointIndex) : 0;
        public Vector3 BodyPosition => GetBodyPosition();
        public bool IsTemporarilyBlocked => false;
        public string TemporaryBlockReason => string.Empty;
        public bool HasCriticalFault => criticalFault;
        public string CriticalFaultReason => criticalFaultReason;

        public float CurrentSpeedMS
        {
            get
            {
                ResolveReferences();
                if (vehicleRigidbody == null)
                {
                    return 0f;
                }

#if UNITY_6000_0_OR_NEWER
                return vehicleRigidbody.linearVelocity.magnitude;
#else
                return vehicleRigidbody.velocity.magnitude;
#endif
            }
        }

        public void Configure(
            VehicleTypes newVehicleType,
            float newSpeedMultiplier,
            float newWaypointReachDistance,
            float newFinalReachDistance,
            DRTPPODrivePolicy newDrivePolicy,
            NNModel newOnnxInferenceModel,
            InferenceDevice newOnnxInferenceDevice)
        {
            Configure(
                null,
                newVehicleType,
                newSpeedMultiplier,
                newWaypointReachDistance,
                newFinalReachDistance,
                newDrivePolicy,
                newOnnxInferenceModel,
                newOnnxInferenceDevice);
        }

        public void Configure(
            Transform newVehicleRoot,
            VehicleTypes newVehicleType,
            float newSpeedMultiplier,
            float newWaypointReachDistance,
            float newFinalReachDistance,
            DRTPPODrivePolicy newDrivePolicy,
            NNModel newOnnxInferenceModel,
            InferenceDevice newOnnxInferenceDevice)
        {
            if (newVehicleRoot != null)
            {
                vehicleRoot = newVehicleRoot;
            }

            vehicleType = newVehicleType;
            speedMultiplier = Mathf.Max(0.1f, newSpeedMultiplier);
            waypointReachDistanceMeters = Mathf.Max(0.5f, newWaypointReachDistance);
            finalReachDistanceMeters = Mathf.Max(0.25f, newFinalReachDistance);
            drivePolicy = newDrivePolicy;
            onnxInferenceModel = newOnnxInferenceModel;
            onnxInferenceDevice = newOnnxInferenceDevice;
            steeringInputSmoothing = Mathf.Max(MinimumSteeringInputSmoothing, steeringInputSmoothing);
            ResolveReferences();
            ApplyPlayerSteeringCapacityOverride();
            ConfigureBehaviorParameters();
        }

        public bool SetPath(List<int> waypointIndexes, Vector3 destination)
        {
            ResolveReferences();
            ClearCriticalFault();
            pathWaypointIndexes.Clear();
            pathWaypoints.Clear();
            pathPoints.Clear();
            pathCumulativeDistances.Clear();
            targetPointIndex = 0;
            finalDestination = destination;
            targetSteeringInput = 0f;
            targetThrottleInput = 0f;
            currentSteeringInput = 0f;
            currentThrottleInput = 0f;
            reverseSeconds = 0f;
            ResetIdleTracking();
            destinationReachedPending = false;
            Vector3 bodyPosition = GetBodyPosition();
            var rawPoints = new List<Vector3>();
            var rawWaypoints = new List<TrafficWaypoint>();
            var rawWaypointIndexes = new List<int>();
            AddRawRoutePoint(rawPoints, rawWaypoints, rawWaypointIndexes, bodyPosition, null, -1);

            if (waypointIndexes != null)
            {
                for (int i = 0; i < waypointIndexes.Count; i++)
                {
                    TrafficWaypoint waypoint = API.GetWaypointFromIndex(waypointIndexes[i]);
                    if (waypoint == null)
                    {
                        continue;
                    }

                    AddRawRoutePoint(rawPoints, rawWaypoints, rawWaypointIndexes, waypoint.Position, waypoint, waypointIndexes[i]);
                }
            }

            TrafficWaypoint lastWaypoint = rawWaypoints.Count > 0 ? rawWaypoints[rawWaypoints.Count - 1] : null;
            int lastWaypointIndex = rawWaypointIndexes.Count > 0 ? rawWaypointIndexes[rawWaypointIndexes.Count - 1] : -1;
            AddRawRoutePoint(rawPoints, rawWaypoints, rawWaypointIndexes, destination, lastWaypoint, lastWaypointIndex);
            Vector3 destinationTailPoint = GetDestinationTailPoint(destination, bodyPosition, rawPoints);
            AddRawRoutePoint(rawPoints, rawWaypoints, rawWaypointIndexes, destinationTailPoint, lastWaypoint, lastWaypointIndex);
            BuildResampledReferencePath(rawPoints, rawWaypoints, rawWaypointIndexes);
            targetPointIndex = pathPoints.Count > 1 ? 1 : 0;
            driving = pathPoints.Count > 0 && playerCar != null;
            previousYawDegrees = GetVehicleYaw();
            InitializeRewardState(bodyPosition);

            if (playerCar != null)
            {
                playerCar.SetExternalInput(0f, 0f, true);
            }

            return driving;
        }

        public void StopAndHold(bool zeroVelocity)
        {
            driving = false;
            pathWaypointIndexes.Clear();
            pathWaypoints.Clear();
            pathPoints.Clear();
            pathCumulativeDistances.Clear();
            targetPointIndex = 0;
            targetSteeringInput = 0f;
            targetThrottleInput = 0f;
            currentSteeringInput = 0f;
            currentThrottleInput = 0f;
            ResetIdleTracking();

            if (playerCar != null)
            {
                playerCar.SetExternalInput(0f, 0f, true);
            }

            if (zeroVelocity)
            {
                SetVelocity(Vector3.zero, Vector3.zero);
            }
        }

        public void ReleaseControl()
        {
            driving = false;
            pathWaypointIndexes.Clear();
            pathWaypoints.Clear();
            pathPoints.Clear();
            pathCumulativeDistances.Clear();
            targetPointIndex = 0;
            targetSteeringInput = 0f;
            targetThrottleInput = 0f;
            currentSteeringInput = 0f;
            currentThrottleInput = 0f;
            ResetIdleTracking();
            ClearCriticalFault();
            destinationReachedPending = false;
            ResetEpisodeTrackingMetrics();

            if (playerCar != null)
            {
                playerCar.ClearExternalInput();
            }

            RestorePlayerSteeringCapacity();
        }

        public void TeleportTo(Vector3 position, Quaternion rotation, int nextWaypointIndex)
        {
            ResolveReferences();
            StopAndHold(true);
            ClearCriticalFault();
            destinationReachedPending = false;
            ResetEpisodeTrackingMetrics();

            if (vehicleRigidbody != null)
            {
                vehicleRigidbody.position = position;
                vehicleRigidbody.rotation = rotation;
            }

            Transform root = GetVehicleRoot();
            if (root != null)
            {
                root.SetPositionAndRotation(position, rotation);
            }

            ResetAgentLocalPose();
            SetVelocity(Vector3.zero, Vector3.zero);
            Physics.SyncTransforms();
        }

        public void ClearCriticalFault()
        {
            criticalFault = false;
            criticalFaultReason = string.Empty;
        }

        public void ReportExternalCriticalFault(string reason, float penalty)
        {
            RegisterCriticalFault(reason, penalty);
        }

        public void SetEndEpisodeOnDestinationReached(bool shouldEndEpisode)
        {
            endEpisodeOnDestinationReached = shouldEndEpisode;
        }

        public void ConfigureSpeedLimit(bool enabled, float speedLimitMetersPerSecond)
        {
            usePolicySpeedLimit = enabled;
            maxPolicySpeedMetersPerSecond = Mathf.Max(0.5f, speedLimitMetersPerSecond);
        }

        public bool ConsumeDestinationReached()
        {
            if (!destinationReachedPending)
            {
                return false;
            }

            destinationReachedPending = false;
            return true;
        }

        private void Awake()
        {
            if (DisableIfSharingNextStopBehaviorParameters())
            {
                return;
            }

            ResolveReferences();
            ApplyPlayerSteeringCapacityOverride();
            ConfigureBehaviorParameters();
        }

        public override void Initialize()
        {
            if (DisableIfSharingNextStopBehaviorParameters())
            {
                return;
            }

            ResolveReferences();
            ApplyPlayerSteeringCapacityOverride();
            ConfigureBehaviorParameters();
        }

        protected override void OnDisable()
        {
            RestorePlayerSteeringCapacity();
            base.OnDisable();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            ResolveReferences();

            Vector3 bodyPosition = GetBodyPosition();
            Vector3 localVelocity = InverseVehicleDirection(GetVelocity());
            float destinationDistance = pathPoints.Count > 0
                ? GetPlanarDistance(bodyPosition, finalDestination)
                : maxObservationDistanceMeters;
            FrenetProjection projection = ProjectToPath(bodyPosition);
            float crossTrackError = projection.IsValid ? projection.LateralError : 0f;
            Vector3 routeTangent = projection.IsValid ? projection.Tangent : GetVehicleForward();
            GetHeadingFeatures(routeTangent, out float headingDot, out float headingCross);
            float totalPathDistance = pathPoints.Count > 1 ? GetPathCumulativeDistance(pathPoints.Count - 1) : 0f;
            float pathProgressRatio = projection.IsValid && totalPathDistance > 0.001f
                ? Mathf.Clamp01(projection.S / totalPathDistance)
                : 0f;

            sensor.AddObservation(Mathf.Clamp01(CurrentSpeedMS / Mathf.Max(0.1f, maxObservationSpeedMetersPerSecond)));
            sensor.AddObservation(Mathf.Clamp(localVelocity.z / Mathf.Max(0.1f, maxObservationSpeedMetersPerSecond), -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(localVelocity.x / Mathf.Max(0.1f, maxObservationSpeedMetersPerSecond), -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(yawRateDegreesPerSecond / 180f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp01(destinationDistance / Mathf.Max(1f, maxObservationDistanceMeters)));
            sensor.AddObservation(pathProgressRatio);
            sensor.AddObservation(Mathf.Clamp01(crossTrackError / Mathf.Max(0.1f, maxCrossTrackErrorMeters)));
            sensor.AddObservation(headingDot);
            sensor.AddObservation(headingCross);
            sensor.AddObservation(driving ? 1f : 0f);

            for (int i = 0; i < LookaheadWaypointCount; i++)
            {
                AddWaypointObservation(sensor, bodyPosition, projection, i);
            }

            AddRayObservations(sensor);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            if (!driving)
            {
                return;
            }

            var continuousActions = actionBuffers.ContinuousActions;
            targetSteeringInput = continuousActions.Length > 0 ? Mathf.Clamp(continuousActions[0], -1f, 1f) : 0f;
            targetThrottleInput = continuousActions.Length > 1 ? Mathf.Clamp(continuousActions[1], -1f, 1f) : 0f;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            ComputeHeuristicControl(out float steering, out float throttle);
            var continuousActionsOut = actionsOut.ContinuousActions;
            if (continuousActionsOut.Length > 0)
            {
                continuousActionsOut[0] = steering;
            }

            if (continuousActionsOut.Length > 1)
            {
                continuousActionsOut[1] = throttle;
            }
        }

        private void FixedUpdate()
        {
            UpdateYawRate();

            if (!driving)
            {
                return;
            }

            ResolveReferences();
            if (playerCar == null || vehicleRigidbody == null || pathPoints.Count == 0)
            {
                RegisterCriticalFault("PPO vehicle references missing.", referenceFaultPenalty);
                return;
            }

            if (criticalFault)
            {
                return;
            }

            UpdateMotionSamples();
            RequestDecision();
            ApplyControl();
            AdvancePathProgress();
            ApplyStepRewardsAndSafety();
        }

        private void OnCollisionEnter(Collision collision)
        {
            NotifyVehicleCollision(collision);
        }

        public void NotifyVehicleCollision(Collision collision)
        {
            if (!driving || collision == null || IsOwnCollider(collision.collider))
            {
                return;
            }

            RegisterCriticalFault($"Collision with {collision.collider.name}.", collisionPenalty);
        }

        private void ConfigureBehaviorParameters()
        {
            if (DisableIfSharingNextStopBehaviorParameters())
            {
                return;
            }

            var behaviorParameters = GetComponent<BehaviorParameters>();
            if (behaviorParameters == null)
            {
                return;
            }

            behaviorParameters.hideFlags = HideFlags.HideInInspector;
            switch (drivePolicy)
            {
                case DRTPPODrivePolicy.ONNXInference:
                    behaviorParameters.Model = onnxInferenceModel;
                    behaviorParameters.InferenceDevice = onnxInferenceDevice;
                    behaviorParameters.BehaviorType = BehaviorType.InferenceOnly;
                    break;
                case DRTPPODrivePolicy.HeuristicPurePursuit:
                    behaviorParameters.Model = null;
                    behaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
                    break;
                default:
                    behaviorParameters.Model = null;
                    behaviorParameters.BehaviorType = BehaviorType.Default;
                    break;
            }

            behaviorParameters.BehaviorName = BehaviorName;
            behaviorParameters.BrainParameters.VectorObservationSize = ObservationSize;
            behaviorParameters.BrainParameters.NumStackedVectorObservations = 1;
            behaviorParameters.BrainParameters.ActionSpec = new ActionSpec(2, Array.Empty<int>());
        }

        private void ApplyPlayerSteeringCapacityOverride()
        {
            if (playerCar == null)
            {
                return;
            }

            if (steeringCapacityOwner != playerCar)
            {
                RestorePlayerSteeringCapacity();
                steeringCapacityOwner = playerCar;
                originalPlayerMaxSteeringAngle = playerCar.maxSteeringAngle;
                hasOriginalPlayerMaxSteeringAngle = true;
            }

            float desiredMaxSteer = Mathf.Max(1f, maxSteeringAngleForFullInput);
            steeringCapacityOverrideActive = hasOriginalPlayerMaxSteeringAngle &&
                                             originalPlayerMaxSteeringAngle < desiredMaxSteer;
            if (playerCar.maxSteeringAngle < desiredMaxSteer)
            {
                playerCar.maxSteeringAngle = desiredMaxSteer;
            }
        }

        private void RestorePlayerSteeringCapacity()
        {
            if (steeringCapacityOwner != null && hasOriginalPlayerMaxSteeringAngle)
            {
                steeringCapacityOwner.maxSteeringAngle = originalPlayerMaxSteeringAngle;
            }

            steeringCapacityOwner = null;
            originalPlayerMaxSteeringAngle = 0f;
            hasOriginalPlayerMaxSteeringAngle = false;
            steeringCapacityOverrideActive = false;
        }

        private bool DisableIfSharingNextStopBehaviorParameters()
        {
            if (GetComponent<DRTNextStopSelector>() == null)
            {
                return false;
            }

            ReleaseControl();
            enabled = false;
            if (!warnedSharedBehaviorHost)
            {
                Debug.LogWarning(
                    "[DRTPPOVehicleDriver] Disabled root PPO driver because it shares BehaviorParameters " +
                    "with DRTNextStopSelector. DRTBusController will use the DRTDrivePPOAgent child instead.");
                warnedSharedBehaviorHost = true;
            }

            return true;
        }

        private void ResolveReferences()
        {
            if (vehicleRoot == null)
            {
                vehicleRoot = playerCar != null
                    ? playerCar.transform
                    : vehicleRigidbody != null
                        ? vehicleRigidbody.transform
                        : transform;
            }

            Transform root = GetVehicleRoot();

            if (playerCar == null)
            {
                playerCar = root != null
                    ? root.GetComponent<PlayerCar>() ?? root.GetComponentInChildren<PlayerCar>() ?? root.GetComponentInParent<PlayerCar>()
                    : GetComponent<PlayerCar>();
            }

            if (vehicleRigidbody == null)
            {
                vehicleRigidbody = root != null
                    ? root.GetComponent<Rigidbody>() ?? root.GetComponentInChildren<Rigidbody>() ?? root.GetComponentInParent<Rigidbody>()
                    : GetComponent<Rigidbody>();
            }

            if (bodyTransform == null)
            {
                bodyTransform = FindChildRecursive(root, "BodyHolder");
            }

            if (ownColliders == null || ownColliders.Length == 0)
            {
                ownColliders = root != null
                    ? root.GetComponentsInChildren<Collider>()
                    : GetComponentsInChildren<Collider>();
            }
        }

        private void AddRawRoutePoint(
            List<Vector3> rawPoints,
            List<TrafficWaypoint> rawWaypoints,
            List<int> rawWaypointIndexes,
            Vector3 point,
            TrafficWaypoint waypoint,
            int waypointIndex)
        {
            if (rawPoints.Count > 0 && GetPlanarDistance(rawPoints[rawPoints.Count - 1], point) < 0.05f)
            {
                rawWaypoints[rawWaypoints.Count - 1] = waypoint ?? rawWaypoints[rawWaypoints.Count - 1];
                rawWaypointIndexes[rawWaypointIndexes.Count - 1] = waypointIndex >= 0
                    ? waypointIndex
                    : rawWaypointIndexes[rawWaypointIndexes.Count - 1];
                return;
            }

            rawPoints.Add(point);
            rawWaypoints.Add(waypoint);
            rawWaypointIndexes.Add(waypointIndex);
        }

        private void BuildResampledReferencePath(
            List<Vector3> rawPoints,
            List<TrafficWaypoint> rawWaypoints,
            List<int> rawWaypointIndexes)
        {
            pathWaypointIndexes.Clear();
            pathWaypoints.Clear();
            pathPoints.Clear();
            pathCumulativeDistances.Clear();
            if (rawPoints == null || rawPoints.Count == 0)
            {
                return;
            }

            AddPathPoint(rawPoints[0], rawWaypoints[0], rawWaypointIndexes[0], true);
            if (rawPoints.Count == 1)
            {
                return;
            }

            float spacing = Mathf.Max(0.1f, referenceWaypointSpacingMeters);
            float nextSampleDistance = spacing;
            float routeDistance = 0f;
            for (int i = 0; i < rawPoints.Count - 1; i++)
            {
                Vector3 start = rawPoints[i];
                Vector3 end = rawPoints[i + 1];
                float segmentLength = GetPlanarDistance(start, end);
                if (segmentLength < 0.001f)
                {
                    continue;
                }

                float segmentStartDistance = routeDistance;
                float segmentEndDistance = routeDistance + segmentLength;
                while (nextSampleDistance <= segmentEndDistance)
                {
                    float t = Mathf.Clamp01((nextSampleDistance - segmentStartDistance) / segmentLength);
                    Vector3 sample = Vector3.Lerp(start, end, t);
                    AddPathPoint(sample, rawWaypoints[i + 1], rawWaypointIndexes[i + 1]);
                    nextSampleDistance += spacing;
                }

                routeDistance = segmentEndDistance;
            }

            int lastIndex = rawPoints.Count - 1;
            AddPathPoint(rawPoints[lastIndex], rawWaypoints[lastIndex], rawWaypointIndexes[lastIndex], true);
        }

        private void AddPathPoint(Vector3 point, TrafficWaypoint waypoint = null, int waypointIndex = -1, bool force = false)
        {
            if (pathPoints.Count > 0 && GetPlanarDistance(pathPoints[pathPoints.Count - 1], point) < 0.001f)
            {
                int lastIndex = pathPoints.Count - 1;
                pathWaypoints[lastIndex] = waypoint ?? pathWaypoints[lastIndex];
                pathWaypointIndexes[lastIndex] = waypointIndex >= 0
                    ? waypointIndex
                    : pathWaypointIndexes[lastIndex];
                return;
            }

            if (!force && pathPoints.Count > 0 && GetPlanarDistance(pathPoints[pathPoints.Count - 1], point) < 0.5f)
            {
                return;
            }

            float cumulativeDistance = 0f;
            if (pathPoints.Count > 0)
            {
                int lastIndex = pathPoints.Count - 1;
                cumulativeDistance = pathCumulativeDistances.Count > lastIndex
                    ? pathCumulativeDistances[lastIndex]
                    : 0f;
                cumulativeDistance += GetPlanarDistance(pathPoints[lastIndex], point);
            }

            pathPoints.Add(point);
            pathCumulativeDistances.Add(cumulativeDistance);
            pathWaypoints.Add(waypoint);
            pathWaypointIndexes.Add(waypointIndex);
        }

        private Vector3 GetDestinationTailPoint(Vector3 destination, Vector3 bodyPosition, List<Vector3> routePoints = null)
        {
            List<Vector3> points = routePoints ?? pathPoints;
            Vector3 anchor = points.Count > 0 ? points[points.Count - 1] : destination;
            Vector3 direction = Vector3.zero;
            if (points.Count >= 2)
            {
                direction = points[points.Count - 1] - points[points.Count - 2];
            }

            if (direction.sqrMagnitude < 0.001f)
            {
                direction = destination - bodyPosition;
            }

            direction.y = 0f;
            if (direction.sqrMagnitude < 0.001f)
            {
                direction = GetVehicleForward();
                direction.y = 0f;
            }

            if (direction.sqrMagnitude < 0.001f)
            {
                direction = Vector3.forward;
            }

            direction.Normalize();
            Vector3 toDestination = destination - anchor;
            toDestination.y = 0f;
            float destinationProjection = Mathf.Max(0f, Vector3.Dot(toDestination, direction));
            float tailDistance = destinationProjection + Mathf.Max(0f, destinationTailDistanceMeters);
            Vector3 tailPoint = anchor + direction * tailDistance;
            tailPoint.y = destination.y;
            return tailPoint;
        }

        private void AddWaypointObservation(
            VectorSensor sensor,
            Vector3 bodyPosition,
            FrenetProjection projection,
            int observationIndex)
        {
            if (!projection.IsValid || pathPoints.Count == 0)
            {
                AddEmptyWaypointObservation(sensor);
                return;
            }

            float distanceAhead =
                Mathf.Max(0f, firstLookaheadObservationMeters) +
                Mathf.Max(0.1f, lookaheadObservationSpacingMeters) * observationIndex;
            float sampleS = projection.S + distanceAhead;
            if (!TryGetPathPointAtDistance(sampleS, out Vector3 referencePoint, out _))
            {
                AddEmptyWaypointObservation(sensor);
                return;
            }

            Vector3 localPoint = InverseVehiclePoint(referencePoint);
            localPoint.y = 0f;
            float distance = GetPlanarDistance(bodyPosition, referencePoint);
            int pointIndex = GetPathPointIndexAtDistance(sampleS);
            TrafficWaypoint waypoint = pointIndex >= 0 && pointIndex < pathWaypoints.Count
                ? pathWaypoints[pointIndex]
                : null;

            sensor.AddObservation(1f);
            sensor.AddObservation(Mathf.Clamp(localPoint.x / Mathf.Max(1f, maxObservationDistanceMeters), -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(localPoint.z / Mathf.Max(1f, maxObservationDistanceMeters), -1f, 1f));
            sensor.AddObservation(Mathf.Clamp01(distance / Mathf.Max(1f, maxObservationDistanceMeters)));
            sensor.AddObservation(waypoint != null && waypoint.Stop ? 1f : 0f);
            sensor.AddObservation(waypoint != null && waypoint.GiveWay ? 1f : 0f);
            sensor.AddObservation(waypoint != null ? Mathf.Clamp01(waypoint.MaxSpeed / MaxWaypointSpeedKmh) : 0f);
            sensor.AddObservation(waypoint != null ? Mathf.Clamp01(waypoint.LaneWidth / MaxLaneWidthMeters) : 0f);
        }

        private static void AddEmptyWaypointObservation(VectorSensor sensor)
        {
            for (int i = 0; i < ObservationsPerWaypoint; i++)
            {
                sensor.AddObservation(0f);
            }
        }

        private void AddRayObservations(VectorSensor sensor)
        {
            for (int i = 0; i < RayCount; i++)
            {
                float angle = Mathf.Lerp(-80f, 80f, RayCount == 1 ? 0.5f : (float)i / (RayCount - 1));
                Vector3 direction = Quaternion.AngleAxis(angle, Vector3.up) * GetVehicleForward();
                bool hit = TryRaycast(direction, out RaycastHit rayHit, out bool vehicleHit);
                float normalizedDistance = hit
                    ? Mathf.Clamp01(rayHit.distance / Mathf.Max(0.1f, rayLengthMeters))
                    : 1f;

                sensor.AddObservation(normalizedDistance);
                sensor.AddObservation(vehicleHit ? 1f : 0f);
                sensor.AddObservation(hit && !vehicleHit ? 1f : 0f);
            }
        }

        private void ApplyControl()
        {
            currentSteeringInput = Mathf.MoveTowards(
                currentSteeringInput,
                targetSteeringInput,
                Mathf.Max(0.1f, steeringInputSmoothing) * Time.fixedDeltaTime);
            currentThrottleInput = Mathf.MoveTowards(
                currentThrottleInput,
                targetThrottleInput,
                Mathf.Max(0.1f, throttleInputSmoothing) * Time.fixedDeltaTime);

            float speedLimit = GetPolicySpeedLimit();
            if (speedLimit > 0f && CurrentSpeedMS > speedLimit)
            {
                currentThrottleInput = Mathf.Min(currentThrottleInput, speedLimitBrakeInput);
            }

            currentThrottleInput = ShapeFrontCurveSpeedThrottle(currentThrottleInput);
            currentThrottleInput = ShapeDestinationApproachThrottle(currentThrottleInput);
            playerCar.SetExternalInput(currentSteeringInput, currentThrottleInput, true);
            RecordSteeringControlStats();
        }

        private void RecordSteeringControlStats()
        {
            float targetAbs = Mathf.Abs(targetSteeringInput);
            float appliedAbs = Mathf.Abs(currentSteeringInput);
            float steeringLag = Mathf.Abs(targetSteeringInput - currentSteeringInput);
            float maxSteerDegrees = GetMaxSteerDegrees();

            RecordStat("DRTDrive/TargetSteerAbs", targetAbs);
            RecordStat("DRTDrive/TargetSteer", targetSteeringInput);
            RecordStat("DRTDrive/AppliedSteerAbs", appliedAbs);
            RecordStat("DRTDrive/SteerLagAbs", steeringLag);
            RecordStat("DRTDrive/AppliedSteerDeg", currentSteeringInput * maxSteerDegrees);
            RecordStat("DRTDrive/MaxSteerDeg", maxSteerDegrees);
            RecordStat("DRTDrive/TargetSteerSaturated", targetAbs >= SteeringSaturationThreshold ? 1f : 0f);
            RecordStat("DRTDrive/AppliedSteerSaturated", appliedAbs >= SteeringSaturationThreshold ? 1f : 0f);
            RecordStat("DRTDrive/SteeringCapacityOverride", steeringCapacityOverrideActive ? 1f : 0f);
            RecordStat("DRTDrive/TargetThrottle", targetThrottleInput);
            RecordStat("DRTDrive/AppliedThrottle", currentThrottleInput);
            RecordStat("DRTDrive/SpeedMS", CurrentSpeedMS);
            RecordStat("DRTDrive/ForwardSpeedMS", InverseVehicleDirection(GetVelocity()).z);
        }

        private float ShapeFrontCurveSpeedThrottle(float throttleInput)
        {
            if (!driving || pathPoints.Count < 3)
            {
                return throttleInput;
            }

            FrenetProjection projection = ProjectToPath(GetBodyPosition());
            if (!projection.IsValid)
            {
                return throttleInput;
            }

            float speedLimit = GetGleySpeedLimitMetersPerSecond(projection);
            if (speedLimit <= 0f)
            {
                return throttleInput;
            }

            float allowedSpeed = GetFrontCurveAllowedSpeedMetersPerSecond(projection.S, speedLimit);
            float overspeed = CurrentSpeedMS - allowedSpeed;
            RecordStat("DRTDrive/CurveAllowedSpeedMS", allowedSpeed);
            RecordStat("DRTDrive/CurveSpeedErrorMS", overspeed);
            if (overspeed <= curveSpeedThrottleCutMarginMetersPerSecond)
            {
                RecordStat("DRTDrive/CurveSpeedBrakeApplied", 0f);
                return throttleInput;
            }

            float severity = Mathf.Clamp01(
                (overspeed - curveSpeedThrottleCutMarginMetersPerSecond) /
                Mathf.Max(0.01f, curveSpeedFullBrakeOverspeedMetersPerSecond));
            float brakeInput = Mathf.Lerp(0f, curveSpeedBrakeInput, severity);
            RecordStat("DRTDrive/CurveSpeedBrakeApplied", -brakeInput);
            return Mathf.Min(throttleInput, brakeInput);
        }

        private float ShapeDestinationApproachThrottle(float throttleInput)
        {
            if (!driving || pathPoints.Count == 0)
            {
                return throttleInput;
            }

            float finalDistance = GetPlanarDistance(GetBodyPosition(), finalDestination);
            if (finalDistance > slowDownDistanceMeters)
            {
                return throttleInput;
            }

            float forwardSpeed = InverseVehicleDirection(GetVelocity()).z;
            if (forwardSpeed < -0.05f)
            {
                return destinationApproachRecoveryThrottle;
            }

            float distanceRatio = Mathf.Clamp01(
                (finalDistance - finalReachDistanceMeters) /
                Mathf.Max(0.1f, slowDownDistanceMeters - finalReachDistanceMeters));
            float stopDistance = finalReachDistanceMeters * Mathf.Max(1f, destinationStopDistanceMultiplier);
            float stopRatio = Mathf.Clamp01((finalDistance - finalReachDistanceMeters) / Mathf.Max(0.1f, stopDistance - finalReachDistanceMeters));
            float stopLimitedSpeed = destinationApproachSpeedMetersPerSecond * stopRatio;
            float targetApproachSpeed = Mathf.Lerp(
                stopLimitedSpeed,
                Mathf.Max(destinationApproachSpeedMetersPerSecond, 1.5f),
                distanceRatio);

            if (forwardSpeed > targetApproachSpeed + 0.35f)
            {
                return Mathf.Min(throttleInput, destinationApproachBrakeInput);
            }

            if (targetApproachSpeed > 0.05f && forwardSpeed < targetApproachSpeed)
            {
                return Mathf.Max(throttleInput, destinationApproachCreepThrottle);
            }

            return Mathf.Max(throttleInput, 0f);
        }

        private void AdvancePathProgress()
        {
            Vector3 bodyPosition = GetBodyPosition();
            FrenetProjection projection = ProjectToPath(bodyPosition);
            if (!projection.IsValid)
            {
                return;
            }

            int previousTargetPointIndex = targetPointIndex;
            float passedDistance = projection.S + Mathf.Max(0.01f, referenceWaypointPassDistanceMeters);
            while (targetPointIndex < pathPoints.Count - 1 &&
                   GetPathCumulativeDistance(targetPointIndex) <= passedDistance)
            {
                targetPointIndex++;
            }

            int passedCount = targetPointIndex - previousTargetPointIndex;
            if (passedCount <= 0)
            {
                return;
            }

            AddReward(waypointPassedReward * passedCount);
            RecordStat("DRTDrive/WaypointPassed", passedCount, StatAggregationMethod.Sum);
            RecordStat("DRTDrive/TargetPointIndex", targetPointIndex, StatAggregationMethod.MostRecent);
        }

        private void ApplyStepRewardsAndSafety()
        {
            Vector3 bodyPosition = GetBodyPosition();
            float destinationDistance = GetPlanarDistance(bodyPosition, finalDestination);
            FrenetProjection projection = ProjectToPath(bodyPosition);
            Vector3 routeTangent = projection.IsValid ? projection.Tangent : GetVehicleForward();
            GetHeadingFeatures(routeTangent, out float headingDot, out float headingCross);
            RecordTrackingMetrics(projection.LateralError, headingDot);
            if (projection.IsValid)
            {
                RecordStat("DRTDrive/SignedLateralError", projection.SignedLateralError);
                RecordStat("DRTDrive/HeadingCross", headingCross);
            }

            if (hardCrossTrackLimitMeters > 0f && projection.LateralError > hardCrossTrackLimitMeters)
            {
                RegisterCriticalFault(
                    $"PPO vehicle exceeded lateral path error. e_y={projection.LateralError:0.00}m",
                    assignedRouteExitPenalty);
                return;
            }

            if (destinationDistance <= finalReachDistanceMeters)
            {
                destinationReachedPending = true;
                if (endEpisodeOnDestinationReached)
                {
                    AddReward(destinationReward);
                    RecordStat("DRTDrive/DestinationReached", 1f, StatAggregationMethod.Sum);
                    EmitEpisodeTrackingMetrics();
                    EndEpisode();
                }
                else
                {
                    AddReward(waypointPassedReward);
                    RecordStat("DRTDrive/IntermediateDestinationReached", 1f, StatAggregationMethod.Sum);
                }

                StopAndHold(true);
                return;
            }

            float pathProgressMeters = 0f;
            if (projection.IsValid)
            {
                pathProgressMeters = Mathf.Max(0f, projection.S - lastPathProgressMeters);
                if (pathProgressMeters > 0f)
                {
                    AddReward(pathProgressMeters * pathProgressRewardPerMeter);
                    lastPathProgressMeters = Mathf.Max(lastPathProgressMeters, projection.S);
                }

                lastFrenetSegmentIndex = projection.SegmentIndex;
                float destinationProgressMeters = Mathf.Max(0f, lastDestinationDistance - destinationDistance);
                if (destinationProgressMeters > 0f)
                {
                    AddReward(destinationProgressMeters * destinationProgressRewardPerMeter);
                }

                lastDestinationDistance = destinationDistance;
                ApplyPathTrackingReward(bodyPosition, projection, headingDot, headingCross);
            }

            ApplyReversePenalty();
            ApplyVehicleClearancePenalty();
            ApplyOverspeedPenalty(projection);
            ApplyRoughnessPenalty();
            ApplyUnblockedIdlePenalty(destinationDistance, pathProgressMeters);
            ApplyNoMovementTermination(bodyPosition);
        }

        private void ApplyReversePenalty()
        {
            Vector3 localVelocity = InverseVehicleDirection(GetVelocity());
            if (localVelocity.z < -0.5f)
            {
                reverseSeconds += Time.fixedDeltaTime;
                if (reverseSeconds >= reverseGraceSeconds)
                {
                    AddReward(reversePenalty * Time.fixedDeltaTime);
                }
                return;
            }

            reverseSeconds = 0f;
        }

        private void ApplyPathTrackingReward(
            Vector3 bodyPosition,
            FrenetProjection projection,
            float headingDot,
            float headingCross)
        {
            float dt = Time.fixedDeltaTime;
            float normalizedCrossTrackError = Mathf.Clamp01(
                projection.LateralError /
                Mathf.Max(0.1f, maxCrossTrackErrorMeters));
            float nextWaypointHeadingDot = GetCurrentWaypointHeadingDot(bodyPosition);
            float curveStrength = GetCurveStrength();
            float steeringCorrection = -Mathf.Sign(headingCross) * targetSteeringInput;
            float trackingReward =
                headingAlignmentReward * headingDot +
                waypointHeadingReward * nextWaypointHeadingDot +
                curvePenalty * Mathf.Abs(headingCross) * curveStrength +
                crossTrackPenalty * normalizedCrossTrackError * normalizedCrossTrackError +
                steeringCorrectionReward * steeringCorrection * normalizedCrossTrackError;

            float lookaheadDistance = GetDynamicLookAheadDistance();
            RecordStat("DRTDrive/TrackingLookaheadMeters", lookaheadDistance);
            float lookaheadS = projection.S + lookaheadDistance;
            if (TryGetPathPointAtDistance(lookaheadS, out Vector3 lookaheadPoint, out Vector3 lookaheadTangent))
            {
                Vector3 forward = GetVehicleForward();
                Vector3 toLookahead = lookaheadPoint - bodyPosition;
                toLookahead.y = 0f;
                lookaheadTangent.y = 0f;

                if (forward.sqrMagnitude > 0.001f && toLookahead.sqrMagnitude > 0.001f)
                {
                    float alphaRadians = Mathf.Abs(Vector3.SignedAngle(forward, toLookahead.normalized, Vector3.up)) * Mathf.Deg2Rad;
                    float alphaDegrees = alphaRadians * Mathf.Rad2Deg;
                    float maxSteerDegrees = GetMaxSteerDegrees();
                    float geometricSteerDemand = alphaDegrees / Mathf.Max(1f, maxSteerDegrees);
                    RecordStat("DRTDrive/LookaheadBearingDeg", alphaDegrees);
                    RecordStat("DRTDrive/GeometricSteerDemandAbs", geometricSteerDemand);
                    RecordStat(
                        "DRTDrive/GeometricSteerDemandSaturated",
                        geometricSteerDemand >= SteeringSaturationThreshold ? 1f : 0f);
                }

                if (forward.sqrMagnitude > 0.001f && lookaheadTangent.sqrMagnitude > 0.001f)
                {
                    float headingErrorDegrees = Mathf.Abs(Vector3.SignedAngle(forward, lookaheadTangent.normalized, Vector3.up));
                    RecordStat("DRTDrive/LookaheadHeadingErrorDeg", headingErrorDegrees);
                }
            }

            AddReward(trackingReward * dt);
            RecordStat("DRTDrive/TrackingShapingRewardPerSecond", trackingReward);
        }

        private void ApplyVehicleClearancePenalty()
        {
            if (vehicleClearanceScanRadiusMeters <= 0f || ownColliders == null || ownColliders.Length == 0)
            {
                return;
            }

            Collider[] colliders = Physics.OverlapSphere(
                GetBodyPosition(),
                vehicleClearanceScanRadiusMeters,
                vehicleClearanceLayerMask,
                QueryTriggerInteraction.Ignore);
            if (colliders == null || colliders.Length == 0)
            {
                return;
            }

            float maxFrontSeverity = 0f;
            float maxSideSeverity = 0f;
            for (int i = 0; i < colliders.Length; i++)
            {
                Collider other = colliders[i];
                if (other == null || IsOwnCollider(other) || !IsVehicleCollider(other))
                {
                    continue;
                }

                if (!TryGetVehicleSurfaceSeparation(other, out Vector3 localSeparation, out Vector3 localOtherCenter))
                {
                    continue;
                }

                float frontClearance = Mathf.Max(0f, localSeparation.z);
                bool frontCandidate = localOtherCenter.z > 0f &&
                                      Mathf.Abs(localSeparation.x) <= sideVehicleClearanceMeters;
                if (frontCandidate && frontClearance < frontVehicleClearanceMeters)
                {
                    float severity = Mathf.Clamp01((frontVehicleClearanceMeters - frontClearance) /
                                                   Mathf.Max(0.01f, frontVehicleClearanceMeters));
                    maxFrontSeverity = Mathf.Max(maxFrontSeverity, severity * severity);
                }

                float sideClearance = Mathf.Abs(localSeparation.x);
                bool sideCandidate = Mathf.Abs(localOtherCenter.x) > 0.1f &&
                                     Mathf.Abs(localOtherCenter.z) <= sideVehicleLongitudinalWindowMeters;
                if (sideCandidate && sideClearance < sideVehicleClearanceMeters)
                {
                    float severity = Mathf.Clamp01((sideVehicleClearanceMeters - sideClearance) /
                                                   Mathf.Max(0.01f, sideVehicleClearanceMeters));
                    maxSideSeverity = Mathf.Max(maxSideSeverity, severity * severity);
                }
            }

            float penalty =
                frontVehiclePenaltyPerSecond * maxFrontSeverity +
                sideVehiclePenaltyPerSecond * maxSideSeverity;
            if (penalty <= 0f)
            {
                return;
            }

            AddReward(-penalty * Time.fixedDeltaTime);
            RecordStat("DRTDrive/VehicleClearancePenalty", penalty);
        }

        private void ApplyUnblockedIdlePenalty(float destinationDistance, float pathProgressMeters)
        {
            float dt = Time.fixedDeltaTime;
            Vector3 localVelocity = InverseVehicleDirection(GetVelocity());
            float pathProgressSpeed = UpdateIdleProgressWindow(pathProgressMeters, dt);
            bool lowSpeed =
                CurrentSpeedMS < idleSpeedThresholdMetersPerSecond &&
                Mathf.Abs(localVelocity.z) < idleSpeedThresholdMetersPerSecond;
            bool stopped = pathProgressSpeed < idleProgressThresholdMetersPerSecond;
            bool nearDestination =
                destinationDistance <= Mathf.Max(finalReachDistanceMeters, idleDestinationExemptionMeters);

            bool frontBlocked = false;
            float frontBlockClearance = float.PositiveInfinity;
            if (stopped && !nearDestination)
            {
                frontBlocked = IsFrontBlockedForIdle(out frontBlockClearance);
            }

            RecordStat("DRTDrive/IdleStopped", stopped ? 1f : 0f);
            RecordStat("DRTDrive/IdleLowSpeed", lowSpeed ? 1f : 0f);
            RecordStat("DRTDrive/IdleProgressSpeed", pathProgressSpeed);
            RecordStat("DRTDrive/IdleProgressWindowDistance", idleProgressWindowDistanceMeters);
            RecordStat("DRTDrive/IdleNearDestination", nearDestination ? 1f : 0f);
            RecordStat("DRTDrive/IdleFrontBlocked", frontBlocked ? 1f : 0f);
            if (!float.IsPositiveInfinity(frontBlockClearance))
            {
                RecordStat("DRTDrive/IdleFrontBlockClearance", frontBlockClearance);
            }

            if (!stopped || nearDestination || frontBlocked)
            {
                unblockedIdleSeconds = 0f;
                return;
            }

            unblockedIdleSeconds += dt;
            RecordStat("DRTDrive/UnblockedIdleSeconds", unblockedIdleSeconds);
            if (unblockedIdleSeconds < unblockedIdleGraceSeconds)
            {
                return;
            }

            AddReward(-unblockedIdlePenaltyPerSecond * dt);
            RecordStat("DRTDrive/UnblockedIdlePenalty", unblockedIdlePenaltyPerSecond);
        }

        private float UpdateIdleProgressWindow(float pathProgressMeters, float dt)
        {
            float sampleDistance = Mathf.Max(0f, pathProgressMeters);
            idleProgressWindowSamples.Enqueue(sampleDistance);
            idleProgressWindowDistanceMeters += sampleDistance;
            idleProgressWindowDurationSeconds += dt;

            float windowSeconds = Mathf.Max(dt, idleProgressWindowSeconds);
            while (idleProgressWindowDurationSeconds > windowSeconds &&
                   idleProgressWindowSamples.Count > 1)
            {
                idleProgressWindowDistanceMeters -= idleProgressWindowSamples.Dequeue();
                idleProgressWindowDurationSeconds -= dt;
            }

            return idleProgressWindowDistanceMeters /
                   Mathf.Max(0.0001f, idleProgressWindowDurationSeconds);
        }

        private void ResetIdleTracking()
        {
            unblockedIdleSeconds = 0f;
            idleProgressWindowSamples.Clear();
            idleProgressWindowDistanceMeters = 0f;
            idleProgressWindowDurationSeconds = 0f;
        }

        private bool IsFrontBlockedForIdle(out float nearestFrontClearance)
        {
            nearestFrontClearance = float.PositiveInfinity;
            if (idleFrontBlockClearanceMeters <= 0f || ownColliders == null || ownColliders.Length == 0)
            {
                return false;
            }

            Collider[] colliders = Physics.OverlapSphere(
                GetBodyPosition(),
                Mathf.Max(vehicleClearanceScanRadiusMeters, idleFrontBlockClearanceMeters),
                vehicleClearanceLayerMask,
                QueryTriggerInteraction.Ignore);
            if (colliders == null || colliders.Length == 0)
            {
                return false;
            }

            for (int i = 0; i < colliders.Length; i++)
            {
                Collider other = colliders[i];
                if (other == null || IsOwnCollider(other) || !IsVehicleCollider(other))
                {
                    continue;
                }

                if (!TryGetVehicleSurfaceSeparation(other, out Vector3 localSeparation, out Vector3 localOtherCenter))
                {
                    continue;
                }

                bool frontCandidate = localOtherCenter.z > 0f &&
                                      Mathf.Abs(localSeparation.x) <= sideVehicleClearanceMeters;
                if (!frontCandidate)
                {
                    continue;
                }

                float frontClearance = Mathf.Max(0f, localSeparation.z);
                nearestFrontClearance = Mathf.Min(nearestFrontClearance, frontClearance);
            }

            return nearestFrontClearance <= idleFrontBlockClearanceMeters;
        }

        private void ApplyOverspeedPenalty(FrenetProjection projection)
        {
            float speedLimit = GetGleySpeedLimitMetersPerSecond(projection);
            if (speedLimit <= 0f)
            {
                return;
            }

            float allowedSpeed = projection.IsValid
                ? GetFrontCurveAllowedSpeedMetersPerSecond(projection.S, speedLimit)
                : speedLimit;
            bool curveLimited = allowedSpeed < speedLimit - 0.05f;
            float margin = curveLimited
                ? Mathf.Min(overspeedMarginMetersPerSecond, curveOverspeedMarginMetersPerSecond)
                : overspeedMarginMetersPerSecond;
            float overspeed = CurrentSpeedMS - allowedSpeed - margin;
            if (overspeed <= 0f)
            {
                RecordStat("DRTDrive/AllowedSpeedMS", allowedSpeed);
                return;
            }

            float penaltyMultiplier = curveLimited
                ? Mathf.Max(1f, curveOverspeedPenaltyMultiplier)
                : 1f;
            float penalty = overspeedPenaltyPerSecond * penaltyMultiplier * overspeed * overspeed;
            AddReward(-penalty * Time.fixedDeltaTime);
            RecordStat("DRTDrive/OverspeedPenalty", penalty);
            if (curveLimited)
            {
                RecordStat("DRTDrive/CurveOverspeedPenalty", penalty);
            }
            RecordStat("DRTDrive/AllowedSpeedMS", allowedSpeed);
        }

        private void ApplyRoughnessPenalty()
        {
            if (!hasMotionSample)
            {
                return;
            }

            float lateralAccelerationExcess = Mathf.Max(
                0f,
                Mathf.Abs(smoothedLocalAcceleration.x) - lateralAccelerationFreeMetersPerSecondSquared);
            float longitudinalAccelerationExcess = Mathf.Max(
                0f,
                Mathf.Abs(smoothedLocalAcceleration.z) - longitudinalAccelerationFreeMetersPerSecondSquared);
            float longitudinalJerkExcess = Mathf.Max(
                0f,
                Mathf.Abs(smoothedLongitudinalJerk) - longitudinalJerkFreeMetersPerSecondCubed);
            Vector3 localVelocity = InverseVehicleDirection(GetVelocity());
            float localLateralVelocityAbs = Mathf.Abs(localVelocity.x);
            float localLateralVelocityExcess = Mathf.Max(
                0f,
                localLateralVelocityAbs - localLateralVelocityFreeMetersPerSecond);
            float localLateralVelocityPenalty =
                localLateralVelocityPenaltyPerSecond *
                localLateralVelocityExcess *
                localLateralVelocityExcess;
            float lateralOscillationPenalty = UpdateLateralOscillationPenalty(localVelocity.x);

            float penalty =
                lateralAccelerationPenaltyPerSecond * lateralAccelerationExcess * lateralAccelerationExcess +
                longitudinalAccelerationPenaltyPerSecond * longitudinalAccelerationExcess * longitudinalAccelerationExcess +
                longitudinalJerkPenaltyPerSecond * longitudinalJerkExcess * longitudinalJerkExcess +
                localLateralVelocityPenalty;

            RecordStat("DRTDrive/LocalLateralVelocityAbsMS", localLateralVelocityAbs);
            RecordStat("DRTDrive/LocalLateralVelocityPenalty", localLateralVelocityPenalty);
            RecordStat("DRTDrive/LateralOscillationPenalty", lateralOscillationPenalty);
            if (penalty <= 0f && lateralOscillationPenalty <= 0f)
            {
                return;
            }

            if (penalty > 0f)
            {
                AddReward(-penalty * Time.fixedDeltaTime);
            }

            if (lateralOscillationPenalty > 0f)
            {
                AddReward(-lateralOscillationPenalty);
                RecordStat("DRTDrive/LateralOscillationFlip", 1f, StatAggregationMethod.Sum);
            }

            RecordStat("DRTDrive/RoughnessPenalty", penalty);
        }

        private float UpdateLateralOscillationPenalty(float localLateralVelocity)
        {
            float dt = Mathf.Max(0.0001f, Time.fixedDeltaTime);
            lateralOscillationClockSeconds += dt;

            float threshold = Mathf.Max(0.0001f, lateralOscillationVelocityThresholdMetersPerSecond);
            int sign = 0;
            if (localLateralVelocity > threshold)
            {
                sign = 1;
            }
            else if (localLateralVelocity < -threshold)
            {
                sign = -1;
            }

            float window = Mathf.Max(dt, lateralOscillationFlipWindowSeconds);
            if (sign == 0)
            {
                if (lateralOscillationClockSeconds - lastLateralVelocitySignSeconds > window)
                {
                    previousLateralVelocitySign = 0;
                }

                return 0f;
            }

            float penalty = 0f;
            if (previousLateralVelocitySign == 0)
            {
                lastLateralVelocityFlipSeconds = lateralOscillationClockSeconds;
            }
            else if (sign != previousLateralVelocitySign)
            {
                float flipInterval = lateralOscillationClockSeconds - lastLateralVelocityFlipSeconds;
                if (flipInterval <= window)
                {
                    penalty = lateralOscillationPenaltyPerFlip;
                }

                lastLateralVelocityFlipSeconds = lateralOscillationClockSeconds;
            }

            previousLateralVelocitySign = sign;
            lastLateralVelocitySignSeconds = lateralOscillationClockSeconds;
            return penalty;
        }

        private void ApplyNoMovementTermination(Vector3 bodyPosition)
        {
            if (GetPlanarDistance(bodyPosition, lastMovementPosition) >= minimumMovementMeters)
            {
                lastMovementPosition = bodyPosition;
                lastMovementRealtime = Time.realtimeSinceStartup;
                return;
            }

            if (Time.realtimeSinceStartup - lastMovementRealtime < noProgressTimeoutRealSeconds)
            {
                return;
            }

            RecordStat("DRTDrive/NoMovementTimeout", 1f, StatAggregationMethod.Sum);
            EmitEpisodeTrackingMetrics();
            EndEpisode();
            StopAndHold(false);
        }

        private void ComputeHeuristicControl(out float steering, out float throttle)
        {
            steering = 0f;
            throttle = 0f;

            if (!driving || pathPoints.Count == 0)
            {
                return;
            }

            Vector3 bodyPosition = GetBodyPosition();
            Vector3 targetPoint = GetLookAheadPoint(bodyPosition);
            Vector3 toTarget = targetPoint - bodyPosition;
            toTarget.y = 0f;

            if (toTarget.sqrMagnitude < 0.001f)
            {
                return;
            }

            Vector3 forward = GetVehicleForward();
            forward.y = 0f;
            if (forward.sqrMagnitude < 0.001f)
            {
                forward = Vector3.forward;
            }

            float requestedSteeringAngle = Vector3.SignedAngle(forward.normalized, toTarget.normalized, Vector3.up);
            float maxSteerDegrees = GetMaxSteerDegrees();
            steering = Mathf.Clamp(requestedSteeringAngle / Mathf.Max(1f, maxSteerDegrees), -1f, 1f);

            float finalDistance = GetPlanarDistance(bodyPosition, finalDestination);
            float targetSpeed = GetHeuristicTargetSpeed(Mathf.Abs(requestedSteeringAngle), finalDistance);
            throttle = GetThrottleForTargetSpeed(targetSpeed, finalDistance);
        }

        private Vector3 GetLookAheadPoint(Vector3 bodyPosition)
        {
            float lookAheadDistance = GetDynamicLookAheadDistance();

            Vector3 previousPoint = bodyPosition;
            previousPoint.y = 0f;

            for (int i = Mathf.Clamp(targetPointIndex, 0, pathPoints.Count - 1); i < pathPoints.Count; i++)
            {
                Vector3 nextPoint = pathPoints[i];
                nextPoint.y = 0f;
                float segmentDistance = Vector3.Distance(previousPoint, nextPoint);
                if (segmentDistance >= lookAheadDistance)
                {
                    return Vector3.Lerp(previousPoint, nextPoint, lookAheadDistance / Mathf.Max(0.001f, segmentDistance));
                }

                lookAheadDistance -= segmentDistance;
                previousPoint = nextPoint;
            }

            return pathPoints[pathPoints.Count - 1];
        }

        private float GetDynamicLookAheadDistance()
        {
            return Mathf.Clamp(
                CurrentSpeedMS * Mathf.Max(0.01f, lookAheadTimeSeconds),
                minLookAheadMeters,
                maxLookAheadMeters);
        }

        private float GetHeuristicTargetSpeed(float absSteeringAngle, float finalDistance)
        {
            float targetSpeed = baseCruiseSpeedMetersPerSecond * Mathf.Max(0.1f, speedMultiplier);
            float speedLimit = GetPolicySpeedLimit();
            if (speedLimit > 0f)
            {
                targetSpeed = Mathf.Min(targetSpeed, speedLimit);
            }

            if (absSteeringAngle >= hardTurnAngle)
            {
                targetSpeed *= 0.35f;
            }
            else if (absSteeringAngle >= maxSteeringAngleForFullInput)
            {
                targetSpeed *= 0.6f;
            }

            if (finalDistance < slowDownDistanceMeters)
            {
                float slowFactor = Mathf.Clamp01(finalDistance / Mathf.Max(1f, slowDownDistanceMeters));
                targetSpeed = Mathf.Min(targetSpeed, Mathf.Lerp(1.5f, targetSpeed, slowFactor));
            }

            return Mathf.Max(0.5f, targetSpeed);
        }

        private float GetPolicySpeedLimit()
        {
            if (!usePolicySpeedLimit)
            {
                return 0f;
            }

            return maxPolicySpeedMetersPerSecond > 0f
                ? maxPolicySpeedMetersPerSecond
                : baseCruiseSpeedMetersPerSecond * Mathf.Max(0.1f, speedMultiplier);
        }

        private float GetThrottleForTargetSpeed(float targetSpeed, float finalDistance)
        {
            float currentSpeed = CurrentSpeedMS;
            if (finalDistance <= finalReachDistanceMeters * 2f && currentSpeed > 1f)
            {
                return -0.6f;
            }

            if (currentSpeed > targetSpeed + 1f)
            {
                return -0.35f;
            }

            return Mathf.Clamp((targetSpeed - currentSpeed) / Mathf.Max(1f, targetSpeed), 0.2f, 1f);
        }

        private float GetCurrentWaypointDistance(Vector3 bodyPosition)
        {
            if (pathPoints.Count == 0)
            {
                return 0f;
            }

            int index = Mathf.Clamp(targetPointIndex, 0, pathPoints.Count - 1);
            return GetPlanarDistance(bodyPosition, pathPoints[index]);
        }

        private float GetCurrentWaypointHeadingDot(Vector3 bodyPosition)
        {
            if (pathPoints.Count == 0)
            {
                return 0f;
            }

            int index = Mathf.Clamp(targetPointIndex, 0, pathPoints.Count - 1);
            Vector3 toWaypoint = pathPoints[index] - bodyPosition;
            toWaypoint.y = 0f;
            if (toWaypoint.sqrMagnitude < 0.001f)
            {
                return 1f;
            }

            return Mathf.Clamp(Vector3.Dot(GetVehicleForward(), toWaypoint.normalized), -1f, 1f);
        }

        private float GetCurveStrength()
        {
            if (pathPoints.Count < 3)
            {
                return 0f;
            }

            int centerIndex = Mathf.Clamp(targetPointIndex, 1, pathPoints.Count - 2);
            Vector3 previousSegment = pathPoints[centerIndex] - pathPoints[centerIndex - 1];
            Vector3 nextSegment = pathPoints[centerIndex + 1] - pathPoints[centerIndex];
            previousSegment.y = 0f;
            nextSegment.y = 0f;
            if (previousSegment.sqrMagnitude < 0.001f || nextSegment.sqrMagnitude < 0.001f)
            {
                return 0f;
            }

            float angle = Vector3.Angle(previousSegment.normalized, nextSegment.normalized);
            return Mathf.Clamp01(angle / Mathf.Max(1f, hardTurnAngle));
        }

        private float GetFrontCurveAllowedSpeedMetersPerSecond(float currentS, float speedLimitMetersPerSecond)
        {
            if (speedLimitMetersPerSecond <= 0f || pathPoints.Count < 3)
            {
                return speedLimitMetersPerSecond;
            }

            float sampleSpacing = Mathf.Max(0.5f, frontCurveSampleSpacingMeters);
            float horizon = Mathf.Max(
                sampleSpacing,
                Mathf.Max(frontCurveLookaheadMeters, curvePreviewLookaheadMeters));
            float maxPathDistance = GetPathCumulativeDistance(pathPoints.Count - 1);
            float endS = Mathf.Min(currentS + horizon, maxPathDistance);
            if (endS <= currentS + 0.001f ||
                !TryGetPathPointAtDistance(currentS, out _, out Vector3 previousTangent))
            {
                return speedLimitMetersPerSecond;
            }

            previousTangent.y = 0f;
            if (previousTangent.sqrMagnitude < 0.001f)
            {
                return speedLimitMetersPerSecond;
            }

            previousTangent.Normalize();
            float allowedSpeed = speedLimitMetersPerSecond;
            float maxCurveStrength = 0f;
            for (float sampleS = currentS + sampleSpacing; sampleS <= endS + 0.001f; sampleS += sampleSpacing)
            {
                if (!TryGetPathPointAtDistance(sampleS, out _, out Vector3 tangent))
                {
                    break;
                }

                tangent.y = 0f;
                if (tangent.sqrMagnitude < 0.001f)
                {
                    continue;
                }

                tangent.Normalize();
                float headingDeltaRadians = Mathf.Abs(Vector3.SignedAngle(previousTangent, tangent, Vector3.up)) * Mathf.Deg2Rad;
                float curvature = headingDeltaRadians / sampleSpacing;
                if (curvature > 0.0001f)
                {
                    float curveSpeed = Mathf.Sqrt(
                        Mathf.Max(0.01f, curveLateralAccelerationLimitMetersPerSecondSquared) /
                        Mathf.Max(0.0001f, curvature));
                    float distanceToCurve = Mathf.Max(0f, sampleS - currentS - sampleSpacing);
                    float comfortableSpeedNow = Mathf.Sqrt(
                        curveSpeed * curveSpeed +
                        2f * Mathf.Max(0.01f, curveComfortableDecelerationMetersPerSecondSquared) * distanceToCurve);
                    allowedSpeed = Mathf.Min(allowedSpeed, comfortableSpeedNow);
                    maxCurveStrength = Mathf.Max(maxCurveStrength, Mathf.Clamp01(curvature * 10f));
                }

                previousTangent = tangent;
            }

            if (maxCurveStrength > 0f)
            {
                float minimumCurveSpeed = speedLimitMetersPerSecond * Mathf.Clamp01(curveMinimumSpeedFactor);
                float safetyFactor = Mathf.Lerp(
                    1f,
                    Mathf.Clamp(curveAllowedSpeedSafetyFactor, 0.1f, 1f),
                    maxCurveStrength);
                allowedSpeed *= safetyFactor;
                allowedSpeed = Mathf.Max(minimumCurveSpeed, allowedSpeed);
            }

            RecordStat("DRTDrive/FrontCurveStrength", maxCurveStrength);
            return Mathf.Clamp(allowedSpeed, 0.1f, speedLimitMetersPerSecond);
        }

        private struct FrenetProjection
        {
            public bool IsValid;
            public float S;
            public float LateralError;
            public float SignedLateralError;
            public int SegmentIndex;
            public float SegmentT;
            public Vector3 ClosestPoint;
            public Vector3 Tangent;
        }

        private void InitializeRewardState(Vector3 bodyPosition)
        {
            FrenetProjection projection = ProjectToPath(bodyPosition);
            lastPathProgressMeters = projection.IsValid ? projection.S : 0f;
            lastDestinationDistance = GetPlanarDistance(bodyPosition, finalDestination);
            lastFrenetSegmentIndex = projection.IsValid ? projection.SegmentIndex : 0;
            lastMovementPosition = bodyPosition;
            lastMovementRealtime = Time.realtimeSinceStartup;
            previousVelocity = GetVelocity();
            smoothedLocalAcceleration = Vector3.zero;
            previousSmoothedLongitudinalAcceleration = 0f;
            smoothedLongitudinalJerk = 0f;
            previousLateralVelocitySign = 0;
            lateralOscillationClockSeconds = 0f;
            lastLateralVelocityFlipSeconds = float.NegativeInfinity;
            lastLateralVelocitySignSeconds = float.NegativeInfinity;
            ResetIdleTracking();
            hasMotionSample = false;
        }

        private void UpdateMotionSamples()
        {
            Vector3 velocity = GetVelocity();
            if (!hasMotionSample)
            {
                previousVelocity = velocity;
                hasMotionSample = true;
                return;
            }

            float dt = Mathf.Max(0.0001f, Time.fixedDeltaTime);
            Vector3 rawWorldAcceleration = (velocity - previousVelocity) / dt;
            Vector3 rawLocalAcceleration = InverseVehicleDirection(rawWorldAcceleration);
            smoothedLocalAcceleration = Vector3.Lerp(
                smoothedLocalAcceleration,
                rawLocalAcceleration,
                motionSmoothingFactor);

            float rawLongitudinalJerk =
                (smoothedLocalAcceleration.z - previousSmoothedLongitudinalAcceleration) / dt;
            smoothedLongitudinalJerk = Mathf.Lerp(
                smoothedLongitudinalJerk,
                rawLongitudinalJerk,
                motionSmoothingFactor);
            previousSmoothedLongitudinalAcceleration = smoothedLocalAcceleration.z;
            previousVelocity = velocity;
        }

        private FrenetProjection ProjectToPath(Vector3 bodyPosition)
        {
            FrenetProjection projection = new FrenetProjection
            {
                Tangent = GetVehicleForward()
            };

            if (pathPoints.Count == 0)
            {
                return projection;
            }

            Vector3 planarBody = bodyPosition;
            planarBody.y = 0f;

            if (pathPoints.Count == 1)
            {
                Vector3 toPoint = pathPoints[0];
                toPoint.y = 0f;
                Vector3 offset = planarBody - toPoint;
                projection.IsValid = true;
                projection.S = 0f;
                projection.LateralError = offset.magnitude;
                projection.SignedLateralError = projection.LateralError;
                projection.SegmentIndex = 0;
                projection.SegmentT = 0f;
                projection.ClosestPoint = toPoint;
                return projection;
            }

            float bestDistanceSquared = float.PositiveInfinity;
            int start = Mathf.Clamp(Mathf.Min(targetPointIndex, lastFrenetSegmentIndex) - 2, 0, pathPoints.Count - 2);
            int end = Mathf.Clamp(Mathf.Max(targetPointIndex + LookaheadWaypointCount, lastFrenetSegmentIndex + LookaheadWaypointCount), start, pathPoints.Count - 2);

            for (int i = start; i <= end; i++)
            {
                Vector3 a = pathPoints[i];
                Vector3 b = pathPoints[i + 1];
                a.y = 0f;
                b.y = 0f;
                Vector3 segment = b - a;
                float segmentLengthSquared = segment.sqrMagnitude;
                if (segmentLengthSquared < 0.000001f)
                {
                    continue;
                }

                float t = Mathf.Clamp01(Vector3.Dot(planarBody - a, segment) / segmentLengthSquared);
                Vector3 closest = a + segment * t;
                float distanceSquared = (planarBody - closest).sqrMagnitude;
                if (distanceSquared >= bestDistanceSquared)
                {
                    continue;
                }

                float segmentLength = Mathf.Sqrt(segmentLengthSquared);
                Vector3 tangent = segment / segmentLength;
                Vector3 lateralOffset = planarBody - closest;
                float signedLateralError = Vector3.Cross(tangent, lateralOffset).y;

                bestDistanceSquared = distanceSquared;
                projection.IsValid = true;
                projection.S = GetPathCumulativeDistance(i) + segmentLength * t;
                projection.LateralError = Mathf.Sqrt(distanceSquared);
                projection.SignedLateralError = signedLateralError;
                projection.SegmentIndex = i;
                projection.SegmentT = t;
                projection.ClosestPoint = closest;
                projection.Tangent = tangent;
            }

            return projection;
        }

        private bool TryGetPathPointAtDistance(float pathDistance, out Vector3 point, out Vector3 tangent)
        {
            point = pathPoints.Count > 0 ? pathPoints[pathPoints.Count - 1] : Vector3.zero;
            tangent = GetVehicleForward();
            if (pathPoints.Count == 0)
            {
                return false;
            }

            if (pathPoints.Count == 1)
            {
                point = pathPoints[0];
                return true;
            }

            float clampedDistance = Mathf.Clamp(pathDistance, 0f, GetPathCumulativeDistance(pathPoints.Count - 1));
            for (int i = 0; i < pathPoints.Count - 1; i++)
            {
                float startDistance = GetPathCumulativeDistance(i);
                float endDistance = GetPathCumulativeDistance(i + 1);
                if (clampedDistance > endDistance && i < pathPoints.Count - 2)
                {
                    continue;
                }

                Vector3 a = pathPoints[i];
                Vector3 b = pathPoints[i + 1];
                a.y = 0f;
                b.y = 0f;
                Vector3 segment = b - a;
                float segmentLength = segment.magnitude;
                if (segmentLength < 0.001f)
                {
                    continue;
                }

                float t = Mathf.InverseLerp(startDistance, endDistance, clampedDistance);
                point = Vector3.Lerp(pathPoints[i], pathPoints[i + 1], t);
                tangent = segment / segmentLength;
                return true;
            }

            point = pathPoints[pathPoints.Count - 1];
            Vector3 finalSegment = pathPoints[pathPoints.Count - 1] - pathPoints[pathPoints.Count - 2];
            finalSegment.y = 0f;
            tangent = finalSegment.sqrMagnitude > 0.001f ? finalSegment.normalized : GetVehicleForward();
            return true;
        }

        private float GetPathCumulativeDistance(int index)
        {
            if (pathCumulativeDistances.Count == 0)
            {
                return 0f;
            }

            return pathCumulativeDistances[Mathf.Clamp(index, 0, pathCumulativeDistances.Count - 1)];
        }

        private int GetPathPointIndexAtDistance(float pathDistance)
        {
            if (pathCumulativeDistances.Count == 0)
            {
                return -1;
            }

            float clampedDistance = Mathf.Clamp(
                pathDistance,
                0f,
                pathCumulativeDistances[pathCumulativeDistances.Count - 1]);
            for (int i = 0; i < pathCumulativeDistances.Count; i++)
            {
                if (pathCumulativeDistances[i] >= clampedDistance)
                {
                    return i;
                }
            }

            return pathCumulativeDistances.Count - 1;
        }

        private float GetGleySpeedLimitMetersPerSecond(FrenetProjection projection)
        {
            TrafficWaypoint waypoint = GetWaypointForProjection(projection);
            if (waypoint != null && waypoint.MaxSpeed > 0f)
            {
                return waypoint.MaxSpeed / 3.6f;
            }

            float policySpeedLimit = GetPolicySpeedLimit();
            if (policySpeedLimit > 0f)
            {
                return policySpeedLimit;
            }

            return baseCruiseSpeedMetersPerSecond * Mathf.Max(0.1f, speedMultiplier);
        }

        private TrafficWaypoint GetWaypointForProjection(FrenetProjection projection)
        {
            int waypointIndex = projection.IsValid
                ? projection.SegmentIndex + 1
                : targetPointIndex;
            if (waypointIndex >= 0 && waypointIndex < pathWaypoints.Count)
            {
                return pathWaypoints[waypointIndex];
            }

            if (pathWaypoints.Count == 0)
            {
                return null;
            }

            return pathWaypoints[Mathf.Clamp(targetPointIndex, 0, pathWaypoints.Count - 1)];
        }

        private bool TryGetVehicleSurfaceSeparation(
            Collider other,
            out Vector3 localSeparation,
            out Vector3 localOtherCenter)
        {
            localSeparation = Vector3.zero;
            localOtherCenter = Vector3.zero;
            if (other == null || ownColliders == null)
            {
                return false;
            }

            float bestDistanceSquared = float.PositiveInfinity;
            Vector3 bestOwnPoint = Vector3.zero;
            Vector3 bestOtherPoint = Vector3.zero;
            for (int i = 0; i < ownColliders.Length; i++)
            {
                Collider own = ownColliders[i];
                if (own == null || !own.enabled)
                {
                    continue;
                }

                Vector3 ownPoint = own.ClosestPoint(other.bounds.center);
                Vector3 otherPoint = other.ClosestPoint(ownPoint);
                ownPoint = own.ClosestPoint(otherPoint);
                float distanceSquared = (otherPoint - ownPoint).sqrMagnitude;
                if (distanceSquared >= bestDistanceSquared)
                {
                    continue;
                }

                bestDistanceSquared = distanceSquared;
                bestOwnPoint = ownPoint;
                bestOtherPoint = otherPoint;
            }

            if (float.IsInfinity(bestDistanceSquared))
            {
                return false;
            }

            localSeparation = InverseVehicleDirection(bestOtherPoint - bestOwnPoint);
            localOtherCenter = InverseVehiclePoint(other.bounds.center);
            return true;
        }

        private float GetMaxSteerDegrees()
        {
            return playerCar != null && playerCar.maxSteeringAngle > 0f
                ? playerCar.maxSteeringAngle
                : Mathf.Max(1f, maxSteeringAngleForFullInput);
        }

        private float GetCrossTrackError(Vector3 bodyPosition, out Vector3 tangent)
        {
            tangent = GetVehicleForward();
            if (pathPoints.Count == 0)
            {
                return 0f;
            }

            if (pathPoints.Count == 1)
            {
                Vector3 toOnlyPoint = pathPoints[0] - bodyPosition;
                toOnlyPoint.y = 0f;
                if (toOnlyPoint.sqrMagnitude > 0.001f)
                {
                    tangent = toOnlyPoint.normalized;
                }

                return toOnlyPoint.magnitude;
            }

            float bestDistance = float.PositiveInfinity;
            Vector3 bestTangent = tangent;
            int start = Mathf.Clamp(targetPointIndex - 2, 0, pathPoints.Count - 2);
            int end = Mathf.Clamp(targetPointIndex + LookaheadWaypointCount, start, pathPoints.Count - 2);
            Vector3 planarBody = bodyPosition;
            planarBody.y = 0f;

            for (int i = start; i <= end; i++)
            {
                Vector3 a = pathPoints[i];
                Vector3 b = pathPoints[i + 1];
                a.y = 0f;
                b.y = 0f;
                Vector3 segment = b - a;
                float segmentMagnitude = segment.magnitude;
                if (segmentMagnitude < 0.001f)
                {
                    continue;
                }

                float t = Mathf.Clamp01(Vector3.Dot(planarBody - a, segment) / (segmentMagnitude * segmentMagnitude));
                Vector3 closest = a + segment * t;
                float distance = Vector3.Distance(planarBody, closest);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestTangent = segment / segmentMagnitude;
                }
            }

            tangent = bestTangent;
            return float.IsInfinity(bestDistance) ? 0f : bestDistance;
        }

        private void GetHeadingFeatures(Vector3 routeTangent, out float dot, out float cross)
        {
            Vector3 forward = GetVehicleForward();
            forward.y = 0f;
            routeTangent.y = 0f;

            if (forward.sqrMagnitude < 0.001f || routeTangent.sqrMagnitude < 0.001f)
            {
                dot = 0f;
                cross = 0f;
                return;
            }

            forward.Normalize();
            routeTangent.Normalize();
            dot = Mathf.Clamp(Vector3.Dot(forward, routeTangent), -1f, 1f);
            cross = Mathf.Clamp(Vector3.Cross(forward, routeTangent).y, -1f, 1f);
        }

        private bool TryRaycast(Vector3 direction, out RaycastHit selectedHit, out bool vehicleHit)
        {
            selectedHit = default;
            vehicleHit = false;

            Vector3 origin = GetBodyPosition() + Vector3.up * rayHeightMeters;
            RaycastHit[] hits = Physics.RaycastAll(
                origin,
                direction.normalized,
                Mathf.Max(0.1f, rayLengthMeters),
                rayLayerMask,
                QueryTriggerInteraction.Ignore);
            if (hits == null || hits.Length == 0)
            {
                return false;
            }

            Array.Sort(hits, (left, right) => left.distance.CompareTo(right.distance));
            for (int i = 0; i < hits.Length; i++)
            {
                if (IsOwnCollider(hits[i].collider))
                {
                    continue;
                }

                selectedHit = hits[i];
                vehicleHit = IsVehicleCollider(selectedHit.collider);
                return true;
            }

            return false;
        }

        private bool IsVehicleCollider(Collider collider)
        {
            if (collider == null)
            {
                return false;
            }

            return collider.GetComponentInParent<VehicleComponent>() != null ||
                   collider.GetComponentInParent<PlayerCar>() != null;
        }

        private bool IsOwnCollider(Collider collider)
        {
            if (collider == null || ownColliders == null)
            {
                return false;
            }

            for (int i = 0; i < ownColliders.Length; i++)
            {
                if (ownColliders[i] == collider)
                {
                    return true;
                }
            }

            return false;
        }

        private void RegisterCriticalFault(string reason, float penalty)
        {
            if (criticalFault)
            {
                return;
            }

            criticalFault = true;
            criticalFaultReason = string.IsNullOrWhiteSpace(reason) ? "PPO vehicle critical fault." : reason;
            AddReward(penalty);
            RecordStat("DRTDrive/CriticalFault", 1f, StatAggregationMethod.Sum);
            EmitEpisodeTrackingMetrics();
            EndEpisode();
            StopAndHold(false);
        }

        private void RecordTrackingMetrics(float crossTrackErrorMeters, float headingDot)
        {
            float adeMeters = Mathf.Max(0f, crossTrackErrorMeters);
            float headingErrorDegrees = Mathf.Acos(Mathf.Clamp(headingDot, -1f, 1f)) * Mathf.Rad2Deg;

            episodeAdeSumMeters += adeMeters;
            episodeHeadingErrorSumDegrees += headingErrorDegrees;
            episodeMaxAdeMeters = Mathf.Max(episodeMaxAdeMeters, adeMeters);
            episodeMaxHeadingErrorDegrees = Mathf.Max(episodeMaxHeadingErrorDegrees, headingErrorDegrees);
            episodeTrackingMetricSamples++;

            RecordStat("DRTDrive/ADE", adeMeters);
            RecordStat("DRTDrive/HeadingErrorDeg", headingErrorDegrees);
        }

        private void EmitEpisodeTrackingMetrics()
        {
            if (episodeTrackingMetricSamples <= 0)
            {
                return;
            }

            float sampleCount = Mathf.Max(1, episodeTrackingMetricSamples);
            RecordStat("DRTDrive/EpisodeMeanADE", episodeAdeSumMeters / sampleCount, StatAggregationMethod.MostRecent);
            RecordStat("DRTDrive/EpisodeMeanHeadingErrorDeg", episodeHeadingErrorSumDegrees / sampleCount, StatAggregationMethod.MostRecent);
            RecordStat("DRTDrive/EpisodeMaxADE", episodeMaxAdeMeters, StatAggregationMethod.MostRecent);
            RecordStat("DRTDrive/EpisodeMaxHeadingErrorDeg", episodeMaxHeadingErrorDegrees, StatAggregationMethod.MostRecent);
        }

        private void ResetEpisodeTrackingMetrics()
        {
            episodeAdeSumMeters = 0f;
            episodeHeadingErrorSumDegrees = 0f;
            episodeMaxAdeMeters = 0f;
            episodeMaxHeadingErrorDegrees = 0f;
            episodeTrackingMetricSamples = 0;
        }

        private void UpdateYawRate()
        {
            float currentYaw = GetVehicleYaw();
            yawRateDegreesPerSecond = Mathf.DeltaAngle(previousYawDegrees, currentYaw) / Mathf.Max(0.0001f, Time.fixedDeltaTime);
            previousYawDegrees = currentYaw;
        }

        private Vector3 GetBodyPosition()
        {
            ResolveReferences();
            if (bodyTransform != null)
            {
                return bodyTransform.position;
            }

            if (vehicleRigidbody != null)
            {
                return vehicleRigidbody.position;
            }

            Transform root = GetVehicleRoot();
            return root != null ? root.position : transform.position;
        }

        private Transform GetVehicleRoot()
        {
            if (vehicleRoot != null)
            {
                return vehicleRoot;
            }

            if (playerCar != null)
            {
                return playerCar.transform;
            }

            return vehicleRigidbody != null ? vehicleRigidbody.transform : transform;
        }

        private Vector3 GetVehicleForward()
        {
            Transform root = GetVehicleRoot();
            Vector3 forward = root != null ? root.forward : transform.forward;
            forward.y = 0f;
            return forward.sqrMagnitude > 0.001f ? forward.normalized : Vector3.forward;
        }

        private float GetVehicleYaw()
        {
            Transform root = GetVehicleRoot();
            return root != null ? root.eulerAngles.y : transform.eulerAngles.y;
        }

        private Vector3 InverseVehiclePoint(Vector3 point)
        {
            Transform root = GetVehicleRoot();
            return root != null ? root.InverseTransformPoint(point) : transform.InverseTransformPoint(point);
        }

        private Vector3 InverseVehicleDirection(Vector3 direction)
        {
            Transform root = GetVehicleRoot();
            return root != null ? root.InverseTransformDirection(direction) : transform.InverseTransformDirection(direction);
        }

        private void ResetAgentLocalPose()
        {
            Transform root = GetVehicleRoot();
            if (root == null || transform == root || transform.parent != root)
            {
                return;
            }

            transform.localPosition = Vector3.zero;
            transform.localRotation = Quaternion.identity;
            transform.localScale = Vector3.one;
        }

        private Vector3 GetVelocity()
        {
            if (vehicleRigidbody == null)
            {
                return Vector3.zero;
            }

#if UNITY_6000_0_OR_NEWER
            return vehicleRigidbody.linearVelocity;
#else
            return vehicleRigidbody.velocity;
#endif
        }

        private void SetVelocity(Vector3 velocity, Vector3 angularVelocity)
        {
            if (vehicleRigidbody == null)
            {
                return;
            }

#if UNITY_6000_0_OR_NEWER
            vehicleRigidbody.linearVelocity = velocity;
#else
            vehicleRigidbody.velocity = velocity;
#endif
            vehicleRigidbody.angularVelocity = angularVelocity;
        }

        private static Transform FindChildRecursive(Transform root, string childName)
        {
            if (root == null)
            {
                return null;
            }

            for (int i = 0; i < root.childCount; i++)
            {
                Transform child = root.GetChild(i);
                if (child.name == childName)
                {
                    return child;
                }

                Transform match = FindChildRecursive(child, childName);
                if (match != null)
                {
                    return match;
                }
            }

            return null;
        }

        private static float GetPlanarDistance(Vector3 a, Vector3 b)
        {
            a.y = 0f;
            b.y = 0f;
            return Vector3.Distance(a, b);
        }

        private static void RecordStat(
            string name,
            float value,
            StatAggregationMethod aggregationMethod = StatAggregationMethod.Average)
        {
            Academy.Instance.StatsRecorder.Add(name, value, aggregationMethod);
        }

        private void OnValidate()
        {
            speedMultiplier = Mathf.Max(0.1f, speedMultiplier);
            waypointReachDistanceMeters = Mathf.Max(0.5f, waypointReachDistanceMeters);
            finalReachDistanceMeters = Mathf.Max(0.25f, finalReachDistanceMeters);
            destinationTailDistanceMeters = Mathf.Max(0f, destinationTailDistanceMeters);
            referenceWaypointSpacingMeters = Mathf.Max(0.1f, referenceWaypointSpacingMeters);
            referenceWaypointPassDistanceMeters = Mathf.Clamp(
                referenceWaypointPassDistanceMeters,
                0.01f,
                Mathf.Max(0.01f, referenceWaypointSpacingMeters));
            baseCruiseSpeedMetersPerSecond = Mathf.Max(0.5f, baseCruiseSpeedMetersPerSecond);
            maxPolicySpeedMetersPerSecond = Mathf.Max(0f, maxPolicySpeedMetersPerSecond);
            speedLimitBrakeInput = Mathf.Clamp(speedLimitBrakeInput, -1f, 0f);
            curveSpeedBrakeInput = Mathf.Clamp(curveSpeedBrakeInput, -1f, 0f);
            curveSpeedThrottleCutMarginMetersPerSecond = Mathf.Max(0f, curveSpeedThrottleCutMarginMetersPerSecond);
            curveSpeedFullBrakeOverspeedMetersPerSecond = Mathf.Max(0.01f, curveSpeedFullBrakeOverspeedMetersPerSecond);
            maxObservationSpeedMetersPerSecond = Mathf.Max(0.5f, maxObservationSpeedMetersPerSecond);
            maxSteeringAngleForFullInput = Mathf.Max(1f, maxSteeringAngleForFullInput);
            hardTurnAngle = Mathf.Max(maxSteeringAngleForFullInput, hardTurnAngle);
            slowDownDistanceMeters = Mathf.Max(1f, slowDownDistanceMeters);
            lookAheadTimeSeconds = Mathf.Max(0.01f, lookAheadTimeSeconds);
            minLookAheadMeters = Mathf.Max(0.5f, minLookAheadMeters);
            maxLookAheadMeters = Mathf.Max(minLookAheadMeters, maxLookAheadMeters);
            steeringInputSmoothing = Mathf.Max(MinimumSteeringInputSmoothing, steeringInputSmoothing);
            throttleInputSmoothing = Mathf.Max(0.1f, throttleInputSmoothing);
            destinationApproachSpeedMetersPerSecond = Mathf.Max(0.1f, destinationApproachSpeedMetersPerSecond);
            destinationStopDistanceMultiplier = Mathf.Max(1f, destinationStopDistanceMultiplier);
            destinationApproachBrakeInput = Mathf.Clamp(destinationApproachBrakeInput, -1f, 0f);
            destinationApproachCreepThrottle = Mathf.Clamp01(destinationApproachCreepThrottle);
            destinationApproachRecoveryThrottle = Mathf.Clamp01(destinationApproachRecoveryThrottle);
            maxObservationDistanceMeters = Mathf.Max(1f, maxObservationDistanceMeters);
            maxCrossTrackErrorMeters = Mathf.Max(0.1f, maxCrossTrackErrorMeters);
            firstLookaheadObservationMeters = Mathf.Max(0f, firstLookaheadObservationMeters);
            lookaheadObservationSpacingMeters = Mathf.Max(0.1f, lookaheadObservationSpacingMeters);
            rayLengthMeters = Mathf.Max(0.1f, rayLengthMeters);
            rayHeightMeters = Mathf.Max(0f, rayHeightMeters);
            pathProgressRewardPerMeter = Mathf.Max(0f, pathProgressRewardPerMeter);
            destinationProgressRewardPerMeter = Mathf.Max(0f, destinationProgressRewardPerMeter);
            headingAlignmentReward = Mathf.Max(0f, headingAlignmentReward);
            waypointHeadingReward = Mathf.Max(0f, waypointHeadingReward);
            curvePenalty = Mathf.Min(0f, curvePenalty);
            crossTrackPenalty = Mathf.Min(0f, crossTrackPenalty);
            steeringCorrectionReward = Mathf.Max(0f, steeringCorrectionReward);
            waypointPassedReward = Mathf.Max(0f, waypointPassedReward);
            destinationReward = Mathf.Max(0f, destinationReward);
            collisionPenalty = Mathf.Min(0f, collisionPenalty);
            assignedRouteExitPenalty = Mathf.Min(0f, assignedRouteExitPenalty);
            referenceFaultPenalty = Mathf.Min(0f, referenceFaultPenalty);
            reversePenalty = Mathf.Min(0f, reversePenalty);
            overspeedPenaltyPerSecond = Mathf.Max(0f, overspeedPenaltyPerSecond);
            overspeedMarginMetersPerSecond = Mathf.Max(0f, overspeedMarginMetersPerSecond);
            curveOverspeedPenaltyMultiplier = Mathf.Max(1f, curveOverspeedPenaltyMultiplier);
            curveOverspeedMarginMetersPerSecond = Mathf.Max(0f, curveOverspeedMarginMetersPerSecond);
            curveMinimumSpeedFactor = Mathf.Clamp01(curveMinimumSpeedFactor);
            frontCurveLookaheadMeters = Mathf.Max(0f, frontCurveLookaheadMeters);
            curvePreviewLookaheadMeters = Mathf.Max(0f, curvePreviewLookaheadMeters);
            frontCurveSampleSpacingMeters = Mathf.Max(0.5f, frontCurveSampleSpacingMeters);
            curveAllowedSpeedSafetyFactor = Mathf.Clamp(curveAllowedSpeedSafetyFactor, 0.1f, 1f);
            curveLateralAccelerationLimitMetersPerSecondSquared = Mathf.Max(0.01f, curveLateralAccelerationLimitMetersPerSecondSquared);
            curveComfortableDecelerationMetersPerSecondSquared = Mathf.Max(0.01f, curveComfortableDecelerationMetersPerSecondSquared);
            frontVehicleClearanceMeters = Mathf.Max(0.1f, frontVehicleClearanceMeters);
            sideVehicleClearanceMeters = Mathf.Max(0.1f, sideVehicleClearanceMeters);
            vehicleClearanceScanRadiusMeters = Mathf.Max(frontVehicleClearanceMeters, vehicleClearanceScanRadiusMeters);
            sideVehicleLongitudinalWindowMeters = Mathf.Max(0.1f, sideVehicleLongitudinalWindowMeters);
            frontVehiclePenaltyPerSecond = Mathf.Max(0f, frontVehiclePenaltyPerSecond);
            sideVehiclePenaltyPerSecond = Mathf.Max(0f, sideVehiclePenaltyPerSecond);
            unblockedIdlePenaltyPerSecond = Mathf.Max(0f, unblockedIdlePenaltyPerSecond);
            unblockedIdleGraceSeconds = Mathf.Max(0f, unblockedIdleGraceSeconds);
            idleSpeedThresholdMetersPerSecond = Mathf.Max(0f, idleSpeedThresholdMetersPerSecond);
            idleProgressThresholdMetersPerSecond = Mathf.Max(0f, idleProgressThresholdMetersPerSecond);
            idleProgressWindowSeconds = Mathf.Max(0.02f, idleProgressWindowSeconds);
            idleFrontBlockClearanceMeters = Mathf.Max(frontVehicleClearanceMeters, idleFrontBlockClearanceMeters);
            idleDestinationExemptionMeters = Mathf.Max(finalReachDistanceMeters, idleDestinationExemptionMeters);
            lateralAccelerationFreeMetersPerSecondSquared = Mathf.Max(0f, lateralAccelerationFreeMetersPerSecondSquared);
            longitudinalAccelerationFreeMetersPerSecondSquared = Mathf.Max(0f, longitudinalAccelerationFreeMetersPerSecondSquared);
            longitudinalJerkFreeMetersPerSecondCubed = Mathf.Max(0f, longitudinalJerkFreeMetersPerSecondCubed);
            localLateralVelocityFreeMetersPerSecond = Mathf.Max(0f, localLateralVelocityFreeMetersPerSecond);
            lateralAccelerationPenaltyPerSecond = Mathf.Max(0f, lateralAccelerationPenaltyPerSecond);
            longitudinalAccelerationPenaltyPerSecond = Mathf.Max(0f, longitudinalAccelerationPenaltyPerSecond);
            longitudinalJerkPenaltyPerSecond = Mathf.Max(0f, longitudinalJerkPenaltyPerSecond);
            localLateralVelocityPenaltyPerSecond = Mathf.Max(0f, localLateralVelocityPenaltyPerSecond);
            lateralOscillationVelocityThresholdMetersPerSecond = Mathf.Max(0f, lateralOscillationVelocityThresholdMetersPerSecond);
            lateralOscillationFlipWindowSeconds = Mathf.Max(0.02f, lateralOscillationFlipWindowSeconds);
            lateralOscillationPenaltyPerFlip = Mathf.Max(0f, lateralOscillationPenaltyPerFlip);
            motionSmoothingFactor = Mathf.Clamp(motionSmoothingFactor, 0.01f, 1f);
            noProgressTimeoutRealSeconds = Mathf.Max(0.5f, noProgressTimeoutRealSeconds);
            minimumMovementMeters = Mathf.Max(0.01f, minimumMovementMeters);
            hardCrossTrackLimitMeters = Mathf.Max(0.5f, hardCrossTrackLimitMeters);
            reverseGraceSeconds = Mathf.Max(0f, reverseGraceSeconds);
        }
    }

    [AddComponentMenu("")]
    public class DRTPPOVehicleCollisionRelay : MonoBehaviour
    {
        private DRTPPOVehicleDriver owner;

        public void Configure(DRTPPOVehicleDriver newOwner)
        {
            owner = newOwner;
        }

        private void OnCollisionEnter(Collision collision)
        {
            owner?.NotifyVehicleCollision(collision);
        }
    }
}
