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
using UnityEngine.Serialization;

using VehicleTypes = Gley.TrafficSystem.User.VehicleTypes;

namespace DRT
{
    [AddComponentMenu("DRT/DRT PPO Pure Pursuit Vehicle Driver")]
    [RequireComponent(typeof(BehaviorParameters))]
    public class DRTPPOPurePursuitVehicleDriver : Agent, IDRTVehicleDriver
    {
        private const string BehaviorName = "DRTPurePursuitPPO";
        private const int ObservationSize = 8;
        private const int ContinuousActionSize = 2;
        private const float DefaultWheelBaseMeters = 2.6f;
        private const float MaxWaypointSpeedKmh = 100f;

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

        [Header("Lookahead")]
        [SerializeField] private float minLookaheadMeters = 0.1f;
        [SerializeField] private float maxLookaheadMeters = 25f;
        [SerializeField, Range(0f, 1f)] private float zeroActionLookaheadNormalized = 0f;
        [SerializeField] private float teacherBaseLookaheadMeters = 4f;
        [SerializeField] private float teacherSpeedGainSeconds = 0.5f;
        [SerializeField] private float teacherCurvatureGain = 10f;

        [Header("Pure Pursuit Control")]
        [SerializeField, InspectorName("Fallback Road Speed (m/s)")] private float baseCruiseSpeedMetersPerSecond = 5f;
        [SerializeField, InspectorName("Use Fallback PPO Speed")] private bool usePolicySpeedLimit = true;
        [SerializeField, InspectorName("Fallback PPO Speed (m/s)")] private float maxPolicySpeedMetersPerSecond = 5f;
        [SerializeField, Range(0f, 1f)] private float minTargetSpeedRatio = 0f;
        [SerializeField, Range(0f, 1f)] private float maxTargetSpeedRatio = 1f;
        [SerializeField, Range(0f, 1f)] private float zeroActionSpeedNormalized = 1f;
        [SerializeField] private float speedLimitBrakeInput = -0.45f;
        [SerializeField] private float throttleInputSmoothing = 6f;
        [SerializeField] private float steeringInputSmoothing = 12f;
        [SerializeField, Range(0.01f, 1f)] private float curvatureSmoothingBeta = 0.75f;
        [SerializeField] private float maxSteeringAngleForFullInput = 45f;
        [SerializeField] private float slowDownDistanceMeters = 10f;
        [SerializeField] private float destinationApproachBrakeInput = -0.65f;
        [SerializeField] private float destinationApproachCreepThrottle = 0.25f;
        [SerializeField] private float destinationApproachRecoveryThrottle = 0.35f;
        [HideInInspector, SerializeField] private bool endEpisodeOnDestinationReached = true;

        [Header("Observation")]
        [SerializeField] private float maxObservationSpeedMetersPerSecond = 12f;
        [SerializeField] private float maxObservationCurvature = 0.5f;
        [SerializeField] private float maxCrossTrackErrorMeters = 6f;
        [SerializeField] private float midCurvatureHorizonMeters = 6f;
        [SerializeField] private float farCurvatureHorizonMeters = 12f;
        [SerializeField] private float maxCurvatureHorizonMeters = 24f;
        [SerializeField] private float curvatureSampleSpacingMeters = 1f;

        [Header("Reward Weights")]
        [SerializeField] private float speedRewardPerSecond = 0.005f;
        [SerializeField] private float progressRewardPerMeter = 0.1f;
        [SerializeField] private float waypointPassedReward = 0.05f;
        [SerializeField] private float destinationProgressRewardPerMeter = 0f;
        [SerializeField] private float lookaheadTeacherPenaltyPerMeter = 0f;
        [SerializeField] private float lookaheadChangePenaltyPerMeter = 0f;
        [SerializeField] private float curvaturePenaltyPerSecond = 0.02f;
        [FormerlySerializedAs("lateralErrorPenaltyPerSecond")]
        [SerializeField] private float lateralErrorPenaltyPerMeter = 1.5f;
        [FormerlySerializedAs("localLateralVelocityPenaltyPerSecond")]
        [SerializeField] private float localLateralVelocityPenaltyPerMeter = 1f;
        [SerializeField] private float localLateralVelocityCurvatureGain = 3f;
        [FormerlySerializedAs("headingErrorPenaltyPerSecond")]
        [SerializeField] private float headingErrorPenaltyPerMeter = 0.05f;
        [SerializeField] private float overspeedPenaltyPerSecond = 0.3f;
        [SerializeField] private float stallPenaltyPerSecond = 0.05f;
        [SerializeField] private float destinationReward = 10f;
        [SerializeField] private float collisionPenalty = -2f;
        [SerializeField] private float assignedRouteExitPenalty = -5f;
        [SerializeField] private float referenceFaultPenalty = -1f;

        [Header("Termination")]
        [SerializeField] private float hardCrossTrackLimitMeters = 4f;
        [SerializeField] private float noMovementTimeoutRealSeconds = 30f;
        [SerializeField] private float minimumMovementMeters = 0.25f;
        [SerializeField] private float stallSpeedThresholdMetersPerSecond = 0.05f;
        [SerializeField] private float stallGraceSeconds = 2f;

        private readonly List<int> pathWaypointIndexes = new List<int>();
        private readonly List<TrafficWaypoint> pathWaypoints = new List<TrafficWaypoint>();
        private readonly List<Vector3> pathPoints = new List<Vector3>();
        private readonly List<float> pathCumulativeDistances = new List<float>();

        private Collider[] ownColliders;
        private int targetPointIndex;
        private Vector3 finalDestination;
        private bool driving;
        private float targetLookaheadMeters;
        private float targetSpeedRatio = 1f;
        private float targetSpeedMetersPerSecond;
        private float smoothedLookaheadMeters;
        private float previousSmoothedLookaheadMeters;
        private float smoothedCurvatureCommand;
        private float currentSteeringInput;
        private float currentThrottleInput;
        private float lastPathProgressMeters;
        private float lastDestinationDistance;
        private Vector3 lastMovementPosition;
        private float lastMovementRealtime;
        private float stallSeconds;
        private bool criticalFault;
        private string criticalFaultReason = string.Empty;
        private bool destinationReachedPending;
        private float cachedWheelBaseMeters = DefaultWheelBaseMeters;

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
            ResolveReferences();
            ConfigureBehaviorParameters();
        }

        public void ConfigureSpeedLimit(bool enabled, float speedLimitMetersPerSecond)
        {
            usePolicySpeedLimit = enabled;
            maxPolicySpeedMetersPerSecond = Mathf.Max(0f, speedLimitMetersPerSecond);
        }

        public void ConfigurePurePursuitParameters(
            float newMinLookaheadMeters,
            float newMaxLookaheadMeters,
            float newZeroActionLookaheadNormalized,
            float newMinTargetSpeedRatio,
            float newMaxTargetSpeedRatio,
            float newZeroActionSpeedNormalized,
            float newThrottleInputSmoothing,
            float newSteeringInputSmoothing,
            float newCurvatureSmoothingBeta,
            float newSpeedRewardPerSecond,
            float newProgressRewardPerMeter,
            float newDestinationProgressRewardPerMeter,
            float newDestinationReward,
            float newLookaheadChangePenaltyPerMeter,
            float newLateralErrorPenaltyPerMeter,
            float newLocalLateralVelocityPenaltyPerMeter,
            float newLocalLateralVelocityCurvatureGain,
            float newHeadingErrorPenaltyPerMeter,
            float newOverspeedPenaltyPerSecond,
            float newMaxCrossTrackErrorMeters,
            float newHardCrossTrackLimitMeters,
            float newAssignedRouteExitPenaltyMagnitude,
            float newNoMovementTimeoutRealSeconds,
            float newMinimumMovementMeters)
        {
            minLookaheadMeters = newMinLookaheadMeters;
            maxLookaheadMeters = newMaxLookaheadMeters;
            zeroActionLookaheadNormalized = newZeroActionLookaheadNormalized;
            minTargetSpeedRatio = newMinTargetSpeedRatio;
            maxTargetSpeedRatio = newMaxTargetSpeedRatio;
            zeroActionSpeedNormalized = newZeroActionSpeedNormalized;
            throttleInputSmoothing = newThrottleInputSmoothing;
            steeringInputSmoothing = newSteeringInputSmoothing;
            curvatureSmoothingBeta = newCurvatureSmoothingBeta;
            speedRewardPerSecond = newSpeedRewardPerSecond;
            progressRewardPerMeter = newProgressRewardPerMeter;
            destinationProgressRewardPerMeter = newDestinationProgressRewardPerMeter;
            destinationReward = newDestinationReward;
            lookaheadChangePenaltyPerMeter = newLookaheadChangePenaltyPerMeter;
            lateralErrorPenaltyPerMeter = newLateralErrorPenaltyPerMeter;
            localLateralVelocityPenaltyPerMeter = newLocalLateralVelocityPenaltyPerMeter;
            localLateralVelocityCurvatureGain = newLocalLateralVelocityCurvatureGain;
            headingErrorPenaltyPerMeter = newHeadingErrorPenaltyPerMeter;
            overspeedPenaltyPerSecond = newOverspeedPenaltyPerSecond;
            maxCrossTrackErrorMeters = newMaxCrossTrackErrorMeters;
            hardCrossTrackLimitMeters = newHardCrossTrackLimitMeters;
            assignedRouteExitPenalty = -Mathf.Abs(newAssignedRouteExitPenaltyMagnitude);
            noMovementTimeoutRealSeconds = newNoMovementTimeoutRealSeconds;
            minimumMovementMeters = newMinimumMovementMeters;
            OnValidate();
        }

        public void SetEndEpisodeOnDestinationReached(bool shouldEndEpisode)
        {
            endEpisodeOnDestinationReached = shouldEndEpisode;
        }

        public void ReportExternalCriticalFault(string reason, float penalty)
        {
            RegisterCriticalFault(reason, penalty);
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
            targetLookaheadMeters = Mathf.Clamp(minLookaheadMeters, minLookaheadMeters, maxLookaheadMeters);
            targetSpeedRatio = maxTargetSpeedRatio;
            targetSpeedMetersPerSecond = 0f;
            smoothedLookaheadMeters = targetLookaheadMeters;
            previousSmoothedLookaheadMeters = smoothedLookaheadMeters;
            smoothedCurvatureCommand = 0f;
            currentSteeringInput = 0f;
            currentThrottleInput = 0f;
            stallSeconds = 0f;
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
                    if (waypoint != null)
                    {
                        AddRawRoutePoint(rawPoints, rawWaypoints, rawWaypointIndexes, waypoint.Position, waypoint, waypointIndexes[i]);
                    }
                }
            }

            TrafficWaypoint lastWaypoint = rawWaypoints.Count > 0 ? rawWaypoints[rawWaypoints.Count - 1] : null;
            int lastWaypointIndex = rawWaypointIndexes.Count > 0 ? rawWaypointIndexes[rawWaypointIndexes.Count - 1] : -1;
            AddRawRoutePoint(rawPoints, rawWaypoints, rawWaypointIndexes, destination, lastWaypoint, lastWaypointIndex);
            AddRawRoutePoint(
                rawPoints,
                rawWaypoints,
                rawWaypointIndexes,
                GetDestinationTailPoint(destination, bodyPosition, rawPoints),
                lastWaypoint,
                lastWaypointIndex);
            BuildResampledReferencePath(rawPoints, rawWaypoints, rawWaypointIndexes);

            targetPointIndex = pathPoints.Count > 1 ? 1 : 0;
            driving = pathPoints.Count > 0 && playerCar != null && vehicleRigidbody != null;
            InitializeEpisodeState(bodyPosition);

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
            currentSteeringInput = 0f;
            currentThrottleInput = 0f;
            targetSpeedRatio = maxTargetSpeedRatio;
            targetSpeedMetersPerSecond = 0f;

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
            currentSteeringInput = 0f;
            currentThrottleInput = 0f;
            targetSpeedRatio = maxTargetSpeedRatio;
            targetSpeedMetersPerSecond = 0f;

            if (playerCar != null)
            {
                playerCar.ClearExternalInput();
            }
        }

        public void TeleportTo(Vector3 position, Quaternion rotation, int nextWaypointIndex)
        {
            ResolveReferences();
            StopAndHold(true);

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

            SetVelocity(Vector3.zero, Vector3.zero);
            Physics.SyncTransforms();
        }

        public void ClearCriticalFault()
        {
            criticalFault = false;
            criticalFaultReason = string.Empty;
        }

        private void Awake()
        {
            ResolveReferences();
            ConfigureBehaviorParameters();
        }

        protected override void OnDisable()
        {
            ReleaseControl();
            base.OnDisable();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            ResolveReferences();
            Vector3 bodyPosition = GetBodyPosition();
            FrenetProjection projection = ProjectToPath(bodyPosition);
            float currentS = projection.IsValid ? projection.S : 0f;
            float kappa0;
            float kappa1;
            float kappa2;
            float maxKappa = GetMaxCurvatureFeatures(currentS, out kappa0, out kappa1, out kappa2);
            float deltaKappa = kappa1 - kappa0;
            Vector3 routeTangent = projection.IsValid ? projection.Tangent : GetVehicleForward();
            float headingError = GetSignedHeadingError(routeTangent);
            float allowedSpeed = GetAllowedSpeedMetersPerSecond(projection);

            sensor.AddObservation(Mathf.Clamp01(CurrentSpeedMS / Mathf.Max(0.1f, maxObservationSpeedMetersPerSecond)));
            sensor.AddObservation(Mathf.Clamp01(kappa0 / Mathf.Max(0.001f, maxObservationCurvature)));
            sensor.AddObservation(Mathf.Clamp01(kappa1 / Mathf.Max(0.001f, maxObservationCurvature)));
            sensor.AddObservation(Mathf.Clamp01(kappa2 / Mathf.Max(0.001f, maxObservationCurvature)));
            sensor.AddObservation(Mathf.Clamp(deltaKappa / Mathf.Max(0.001f, maxObservationCurvature), -1f, 1f));
            sensor.AddObservation(Mathf.Clamp01((projection.IsValid ? projection.LateralError : 0f) / Mathf.Max(0.1f, maxCrossTrackErrorMeters)));
            sensor.AddObservation(Mathf.Clamp(headingError / 180f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp01(allowedSpeed / Mathf.Max(0.1f, maxObservationSpeedMetersPerSecond)));
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            if (!driving)
            {
                return;
            }

            float rawLookaheadAction = actionBuffers.ContinuousActions.Length > 0
                ? Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f)
                : 0f;
            float normalizedLookahead = MapSignedActionToUnit(rawLookaheadAction, zeroActionLookaheadNormalized);
            targetLookaheadMeters = Mathf.Lerp(minLookaheadMeters, maxLookaheadMeters, normalizedLookahead);

            float rawSpeedAction = actionBuffers.ContinuousActions.Length > 1
                ? Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f)
                : 0f;
            float normalizedSpeed = MapSignedActionToUnit(rawSpeedAction, zeroActionSpeedNormalized);
            targetSpeedRatio = Mathf.Lerp(minTargetSpeedRatio, maxTargetSpeedRatio, normalizedSpeed);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            FrenetProjection projection = ProjectToPath(GetBodyPosition());
            float currentS = projection.IsValid ? projection.S : 0f;
            float maxKappa = GetMaxCurvatureFeatures(currentS, out _, out _, out _);
            float teacherLookahead = GetTeacherLookaheadMeters(CurrentSpeedMS, maxKappa);
            float normalized = Mathf.InverseLerp(minLookaheadMeters, maxLookaheadMeters, teacherLookahead);
            ActionSegment<float> continuousActionsOut = actionsOut.ContinuousActions;
            if (continuousActionsOut.Length > 0)
            {
                continuousActionsOut[0] = MapUnitToSignedAction(normalized, zeroActionLookaheadNormalized);
            }

            if (continuousActionsOut.Length > 1)
            {
                continuousActionsOut[1] = MapUnitToSignedAction(1f, zeroActionSpeedNormalized);
            }
        }

        private static float MapSignedActionToUnit(float action, float zeroActionNormalized)
        {
            float zero = Mathf.Clamp01(zeroActionNormalized);
            float clampedAction = Mathf.Clamp(action, -1f, 1f);
            if (clampedAction <= 0f)
            {
                return Mathf.Lerp(0f, zero, clampedAction + 1f);
            }

            return Mathf.Lerp(zero, 1f, clampedAction);
        }

        private static float MapUnitToSignedAction(float normalized, float zeroActionNormalized)
        {
            float value = Mathf.Clamp01(normalized);
            float zero = Mathf.Clamp01(zeroActionNormalized);
            if (value <= zero)
            {
                return zero <= 0.0001f
                    ? 0f
                    : Mathf.Clamp((value / zero) - 1f, -1f, 0f);
            }

            return zero >= 0.9999f
                ? 0f
                : Mathf.Clamp((value - zero) / (1f - zero), 0f, 1f);
        }

        private void FixedUpdate()
        {
            if (!driving)
            {
                return;
            }

            ResolveReferences();
            if (playerCar == null || vehicleRigidbody == null || pathPoints.Count == 0)
            {
                RegisterCriticalFault("PPO pure pursuit vehicle references missing.", referenceFaultPenalty);
                return;
            }

            RequestDecision();
            ApplyPurePursuitControlAndRewards();
        }

        private void ConfigureBehaviorParameters()
        {
            BehaviorParameters behaviorParameters = GetComponent<BehaviorParameters>();
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
            behaviorParameters.BrainParameters.ActionSpec = new ActionSpec(ContinuousActionSize, Array.Empty<int>());
        }

        private void ApplyPurePursuitControlAndRewards()
        {
            Vector3 bodyPosition = GetBodyPosition();
            FrenetProjection projection = ProjectToPath(bodyPosition);
            if (!projection.IsValid)
            {
                RegisterCriticalFault("PPO pure pursuit reference path projection failed.", referenceFaultPenalty);
                return;
            }

            Vector3 routeTangent = projection.Tangent;
            float headingErrorDegrees = Mathf.Abs(GetSignedHeadingError(routeTangent));
            float destinationDistance = GetPlanarDistance(bodyPosition, finalDestination);
            float kappa0;
            float kappa1;
            float kappa2;
            float maxKappa = GetMaxCurvatureFeatures(projection.S, out kappa0, out kappa1, out kappa2);
            float allowedSpeed = GetAllowedSpeedMetersPerSecond(projection);

            RecordStat("DRTPurePursuit/LateralError", projection.LateralError);
            RecordStat("DRTPurePursuit/HeadingErrorDeg", headingErrorDegrees);
            RecordStat("DRTPurePursuit/Kappa0", kappa0);
            RecordStat("DRTPurePursuit/Kappa1", kappa1);
            RecordStat("DRTPurePursuit/Kappa2", kappa2);
            RecordStat("DRTPurePursuit/AllowedSpeedMS", allowedSpeed);

            if (hardCrossTrackLimitMeters > 0f && projection.LateralError >= hardCrossTrackLimitMeters)
            {
                RegisterCriticalFault(
                    $"PPO pure pursuit exceeded lateral path error. e_y={projection.LateralError:0.00}m",
                    assignedRouteExitPenalty);
                return;
            }

            if (destinationDistance <= finalReachDistanceMeters)
            {
                destinationReachedPending = true;
                if (endEpisodeOnDestinationReached)
                {
                    AddReward(destinationReward);
                    RecordStat("DRTPurePursuit/DestinationReached", 1f, StatAggregationMethod.Sum);
                    EndEpisode();
                }
                else
                {
                    AddReward(waypointPassedReward);
                    RecordStat("DRTPurePursuit/IntermediateDestinationReached", 1f, StatAggregationMethod.Sum);
                }

                StopAndHold(true);
                return;
            }

            AdvancePathProgress(projection);

            previousSmoothedLookaheadMeters = smoothedLookaheadMeters;
            smoothedLookaheadMeters = Mathf.Clamp(targetLookaheadMeters, minLookaheadMeters, maxLookaheadMeters);

            Vector3 targetPoint = GetLookaheadPoint(projection.S, smoothedLookaheadMeters);
            float curvatureCommand = GetPurePursuitCurvature(bodyPosition, targetPoint, smoothedLookaheadMeters);
            smoothedCurvatureCommand = Mathf.Lerp(smoothedCurvatureCommand, curvatureCommand, curvatureSmoothingBeta);
            float steerAngleDegrees = Mathf.Atan(GetWheelBaseMeters() * smoothedCurvatureCommand) * Mathf.Rad2Deg;
            float maxSteerDegrees = GetMaxSteerDegrees();
            float targetSteerInput = Mathf.Clamp(steerAngleDegrees / Mathf.Max(1f, maxSteerDegrees), -1f, 1f);
            currentSteeringInput = Mathf.MoveTowards(
                currentSteeringInput,
                targetSteerInput,
                Mathf.Max(0.1f, steeringInputSmoothing) * Time.fixedDeltaTime);

            targetSpeedMetersPerSecond = GetTargetSpeed(allowedSpeed, targetSpeedRatio, destinationDistance);
            float targetThrottle = GetThrottleForTargetSpeed(targetSpeedMetersPerSecond, destinationDistance);
            currentThrottleInput = Mathf.MoveTowards(
                currentThrottleInput,
                targetThrottle,
                Mathf.Max(0.1f, throttleInputSmoothing) * Time.fixedDeltaTime);

            playerCar.SetExternalInput(currentSteeringInput, currentThrottleInput, true);
            ApplyStepRewards(projection, headingErrorDegrees, maxKappa, allowedSpeed, destinationDistance);
            ApplyStallPenalty();
            ApplyNoMovementTermination(bodyPosition);

            RecordStat("DRTPurePursuit/LookaheadTarget", targetLookaheadMeters);
            RecordStat("DRTPurePursuit/LookaheadApplied", smoothedLookaheadMeters);
            RecordStat("DRTPurePursuit/LookaheadDelta", Mathf.Abs(smoothedLookaheadMeters - previousSmoothedLookaheadMeters));
            RecordStat("DRTPurePursuit/TargetSpeedRatio", targetSpeedRatio);
            RecordStat("DRTPurePursuit/TargetSpeedMS", targetSpeedMetersPerSecond);
            RecordStat("DRTPurePursuit/SteerInput", currentSteeringInput);
            RecordStat("DRTPurePursuit/ThrottleInput", currentThrottleInput);
            RecordStat("DRTPurePursuit/SpeedMS", CurrentSpeedMS);
        }

        private void AdvancePathProgress(FrenetProjection projection)
        {
            int previousTargetPointIndex = targetPointIndex;
            float passedDistance = projection.S + Mathf.Max(0.01f, referenceWaypointPassDistanceMeters);
            while (targetPointIndex < pathPoints.Count - 1 &&
                   GetPathCumulativeDistance(targetPointIndex) <= passedDistance)
            {
                targetPointIndex++;
            }

            int passedCount = targetPointIndex - previousTargetPointIndex;
            if (passedCount > 0)
            {
                AddReward(waypointPassedReward * passedCount);
                RecordStat("DRTPurePursuit/WaypointPassed", passedCount, StatAggregationMethod.Sum);
            }
        }

        private void ApplyStepRewards(
            FrenetProjection projection,
            float headingErrorDegrees,
            float maxKappa,
            float allowedSpeed,
            float destinationDistance)
        {
            float dt = Time.fixedDeltaTime;
            float currentSpeed = CurrentSpeedMS;
            float pathProgressMeters = Mathf.Max(0f, projection.S - lastPathProgressMeters);
            if (pathProgressMeters > 0f)
            {
                AddReward(pathProgressMeters * progressRewardPerMeter);
                lastPathProgressMeters = Mathf.Max(lastPathProgressMeters, projection.S);
            }

            float destinationProgressMeters = Mathf.Max(0f, lastDestinationDistance - destinationDistance);
            if (destinationProgressMeters > 0f)
            {
                AddReward(destinationProgressMeters * destinationProgressRewardPerMeter);
            }
            lastDestinationDistance = destinationDistance;

            float teacherLookahead = GetTeacherLookaheadMeters(currentSpeed, maxKappa);
            float normalizedLateralError = Mathf.Clamp01(projection.LateralError / Mathf.Max(0.1f, maxCrossTrackErrorMeters));
            float localLateralVelocity = Mathf.Abs(GetLocalLateralVelocityMetersPerSecond());
            float headingErrorNorm = Mathf.Clamp01(headingErrorDegrees / 180f);
            float overspeed = Mathf.Max(0f, currentSpeed - allowedSpeed);
            float lookaheadError = Mathf.Abs(smoothedLookaheadMeters - teacherLookahead);
            float lookaheadDelta = Mathf.Abs(smoothedLookaheadMeters - previousSmoothedLookaheadMeters);
            float lateralErrorPenaltyPerMeterStep =
                lateralErrorPenaltyPerMeter * normalizedLateralError * normalizedLateralError;
            float lateralErrorPenalty = lateralErrorPenaltyPerMeterStep * pathProgressMeters;
            float localLateralVelocityCurvatureMultiplier =
                1f + Mathf.Max(0f, localLateralVelocityCurvatureGain) * Mathf.Max(0f, maxKappa);
            float localLateralVelocityPenaltyPerMeterStep =
                localLateralVelocityPenaltyPerMeter *
                localLateralVelocity *
                localLateralVelocityCurvatureMultiplier;
            float localLateralVelocityPenalty =
                localLateralVelocityPenaltyPerMeterStep * pathProgressMeters;
            float headingErrorPenaltyPerMeterStep = headingErrorPenaltyPerMeter * headingErrorNorm;
            float headingErrorPenalty = headingErrorPenaltyPerMeterStep * pathProgressMeters;

            float rewardPerSecond =
                speedRewardPerSecond * currentSpeed -
                lookaheadTeacherPenaltyPerMeter * lookaheadError -
                curvaturePenaltyPerSecond * maxKappa -
                overspeedPenaltyPerSecond * overspeed * overspeed;

            AddReward(rewardPerSecond * dt);
            AddReward(-lateralErrorPenalty);
            AddReward(-localLateralVelocityPenalty);
            AddReward(-headingErrorPenalty);
            AddReward(-lookaheadChangePenaltyPerMeter * lookaheadDelta);

            RecordStat("DRTPurePursuit/RewardPerSecond", rewardPerSecond);
            RecordStat("DRTPurePursuit/LookaheadTeacher", teacherLookahead);
            RecordStat("DRTPurePursuit/LookaheadTeacherPenalty", lookaheadTeacherPenaltyPerMeter * lookaheadError);
            RecordStat("DRTPurePursuit/LookaheadChangePenalty", lookaheadChangePenaltyPerMeter * lookaheadDelta);
            RecordStat("DRTPurePursuit/LateralErrorPenaltyPerMeter", lateralErrorPenaltyPerMeterStep);
            RecordStat("DRTPurePursuit/LateralErrorPenalty", lateralErrorPenalty);
            RecordStat("DRTPurePursuit/LocalLateralVelocityMS", localLateralVelocity);
            RecordStat("DRTPurePursuit/LocalLateralVelocityCurvatureMultiplier", localLateralVelocityCurvatureMultiplier);
            RecordStat("DRTPurePursuit/LocalLateralVelocityPenaltyPerMeter", localLateralVelocityPenaltyPerMeterStep);
            RecordStat("DRTPurePursuit/LocalLateralVelocityPenalty", localLateralVelocityPenalty);
            RecordStat("DRTPurePursuit/HeadingErrorPenaltyPerMeter", headingErrorPenaltyPerMeterStep);
            RecordStat("DRTPurePursuit/HeadingErrorPenalty", headingErrorPenalty);
            RecordStat("DRTPurePursuit/OverspeedPenalty", overspeedPenaltyPerSecond * overspeed * overspeed);
        }

        private void ApplyStallPenalty()
        {
            if (CurrentSpeedMS < stallSpeedThresholdMetersPerSecond)
            {
                stallSeconds += Time.fixedDeltaTime;
                if (stallSeconds >= stallGraceSeconds)
                {
                    AddReward(-stallPenaltyPerSecond * Time.fixedDeltaTime);
                    RecordStat("DRTPurePursuit/StallPenalty", stallPenaltyPerSecond);
                }
                return;
            }

            stallSeconds = 0f;
        }

        private void ApplyNoMovementTermination(Vector3 bodyPosition)
        {
            if (GetPlanarDistance(bodyPosition, lastMovementPosition) >= minimumMovementMeters)
            {
                lastMovementPosition = bodyPosition;
                lastMovementRealtime = Time.realtimeSinceStartup;
                return;
            }

            if (Time.realtimeSinceStartup - lastMovementRealtime < noMovementTimeoutRealSeconds)
            {
                return;
            }

            RecordStat("DRTPurePursuit/NoMovementTimeout", 1f, StatAggregationMethod.Sum);
            RegisterCriticalFault(
                $"PPO pure pursuit no movement timeout. moved<{minimumMovementMeters:0.00}m for {noMovementTimeoutRealSeconds:0.0}s.",
                0f);
        }

        private float GetTargetSpeed(float allowedSpeed, float speedRatio, float finalDistance)
        {
            float targetSpeed = Mathf.Clamp01(speedRatio) * Mathf.Max(0f, allowedSpeed);

            if (finalDistance < slowDownDistanceMeters)
            {
                float slowFactor = Mathf.Clamp01(finalDistance / Mathf.Max(1f, slowDownDistanceMeters));
                targetSpeed = Mathf.Min(targetSpeed, Mathf.Lerp(1.5f, targetSpeed, slowFactor));
            }

            return Mathf.Max(0f, targetSpeed);
        }

        private float GetThrottleForTargetSpeed(float targetSpeed, float finalDistance)
        {
            float currentSpeed = CurrentSpeedMS;
            if (finalDistance <= finalReachDistanceMeters * 2f && currentSpeed > 1f)
            {
                return destinationApproachBrakeInput;
            }

            if (targetSpeed <= 0.05f)
            {
                return currentSpeed > 0.2f ? speedLimitBrakeInput : 0f;
            }

            if (currentSpeed > targetSpeed + 1f)
            {
                return speedLimitBrakeInput;
            }

            if (targetSpeed <= destinationApproachRecoveryThrottle && currentSpeed < targetSpeed)
            {
                return destinationApproachCreepThrottle;
            }

            return Mathf.Clamp((targetSpeed - currentSpeed) / Mathf.Max(1f, targetSpeed), 0.2f, 1f);
        }

        private float GetAllowedSpeedMetersPerSecond(FrenetProjection projection)
        {
            float roadSpeed = GetGleySpeedLimitMetersPerSecond(projection);
            if (roadSpeed <= 0f && usePolicySpeedLimit && maxPolicySpeedMetersPerSecond > 0f)
            {
                roadSpeed = maxPolicySpeedMetersPerSecond;
            }

            if (roadSpeed <= 0f)
            {
                roadSpeed = baseCruiseSpeedMetersPerSecond * Mathf.Max(0.1f, speedMultiplier);
            }

            return Mathf.Max(0.1f, roadSpeed);
        }

        private float GetTeacherLookaheadMeters(float speedMetersPerSecond, float maxKappa)
        {
            float lookahead =
                teacherBaseLookaheadMeters +
                teacherSpeedGainSeconds * speedMetersPerSecond -
                teacherCurvatureGain * maxKappa;
            return Mathf.Clamp(lookahead, minLookaheadMeters, maxLookaheadMeters);
        }

        private float GetPurePursuitCurvature(Vector3 bodyPosition, Vector3 targetPoint, float lookaheadMeters)
        {
            Vector3 localTarget = InverseVehiclePoint(targetPoint);
            float denominator = Mathf.Max(0.001f, lookaheadMeters * lookaheadMeters);
            return 2f * localTarget.x / denominator;
        }

        private Vector3 GetLookaheadPoint(float currentS, float lookaheadMeters)
        {
            return TryGetPathPointAtDistance(currentS + lookaheadMeters, out Vector3 point, out _)
                ? point
                : pathPoints.Count > 0 ? pathPoints[pathPoints.Count - 1] : finalDestination;
        }

        private float GetMaxCurvatureFeatures(float currentS, out float kappa0, out float kappa1, out float kappa2)
        {
            float nearEnd = currentS + midCurvatureHorizonMeters;
            float midEnd = currentS + farCurvatureHorizonMeters;
            float farEnd = currentS + maxCurvatureHorizonMeters;

            kappa0 = GetMaxCurvatureInRange(currentS, nearEnd);
            kappa1 = GetMaxCurvatureInRange(nearEnd, midEnd);
            kappa2 = GetMaxCurvatureInRange(midEnd, farEnd);
            return Mathf.Max(kappa0, Mathf.Max(kappa1, kappa2));
        }

        private float GetMaxCurvatureInRange(float startDistance, float endDistance)
        {
            if (pathPoints.Count < 2)
            {
                return 0f;
            }

            float totalDistance = GetPathCumulativeDistance(pathPoints.Count - 1);
            float start = Mathf.Clamp(Mathf.Min(startDistance, endDistance), 0f, totalDistance);
            float end = Mathf.Clamp(Mathf.Max(startDistance, endDistance), 0f, totalDistance);
            float step = Mathf.Max(0.5f, curvatureSampleSpacingMeters);
            float maxCurvature = 0f;

            if (end <= start + 0.001f)
            {
                return Mathf.Abs(GetCurvatureAtDistance(start));
            }

            for (float sampleS = start; sampleS <= end; sampleS += step)
            {
                maxCurvature = Mathf.Max(maxCurvature, Mathf.Abs(GetCurvatureAtDistance(sampleS)));
            }

            maxCurvature = Mathf.Max(maxCurvature, Mathf.Abs(GetCurvatureAtDistance(end)));
            return maxCurvature;
        }

        private float GetCurvatureAtDistance(float pathDistance)
        {
            float spacing = Mathf.Max(0.5f, curvatureSampleSpacingMeters);
            if (!TryGetPathPointAtDistance(pathDistance, out _, out Vector3 tangentA) ||
                !TryGetPathPointAtDistance(pathDistance + spacing, out _, out Vector3 tangentB))
            {
                return 0f;
            }

            tangentA.y = 0f;
            tangentB.y = 0f;
            if (tangentA.sqrMagnitude < 0.001f || tangentB.sqrMagnitude < 0.001f)
            {
                return 0f;
            }

            float headingDeltaRadians = Mathf.Abs(Vector3.SignedAngle(tangentA.normalized, tangentB.normalized, Vector3.up)) *
                                         Mathf.Deg2Rad;
            return headingDeltaRadians / spacing;
        }

        private void RegisterCriticalFault(string reason, float penalty)
        {
            if (criticalFault)
            {
                return;
            }

            criticalFault = true;
            criticalFaultReason = string.IsNullOrWhiteSpace(reason)
                ? "PPO pure pursuit vehicle critical fault."
                : reason;
            AddReward(penalty);
            RecordStat("DRTPurePursuit/CriticalFault", 1f, StatAggregationMethod.Sum);
            EndEpisode();
            StopAndHold(false);
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (!driving || collision == null || collision.collider == null || IsOwnCollider(collision.collider))
            {
                return;
            }

            RegisterCriticalFault($"PPO pure pursuit collision with {collision.collider.name}.", collisionPenalty);
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

            cachedWheelBaseMeters = ComputeWheelBaseMeters();
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
                int lastIndex = rawPoints.Count - 1;
                rawWaypoints[lastIndex] = waypoint ?? rawWaypoints[lastIndex];
                rawWaypointIndexes[lastIndex] = waypointIndex >= 0 ? waypointIndex : rawWaypointIndexes[lastIndex];
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
                pathWaypointIndexes[lastIndex] = waypointIndex >= 0 ? waypointIndex : pathWaypointIndexes[lastIndex];
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
                Vector3 point = pathPoints[0];
                point.y = 0f;
                projection.IsValid = true;
                projection.S = 0f;
                projection.LateralError = GetPlanarDistance(planarBody, point);
                projection.SignedLateralError = projection.LateralError;
                projection.SegmentIndex = 0;
                projection.ClosestPoint = point;
                return projection;
            }

            float bestDistanceSquared = float.PositiveInfinity;
            for (int i = 0; i < pathPoints.Count - 1; i++)
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

                bestDistanceSquared = distanceSquared;
                projection.IsValid = true;
                projection.S = GetPathCumulativeDistance(i) + segmentLength * t;
                projection.LateralError = Mathf.Sqrt(distanceSquared);
                projection.SignedLateralError = Vector3.Cross(tangent, lateralOffset).y;
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

        private float GetGleySpeedLimitMetersPerSecond(FrenetProjection projection)
        {
            int waypointIndex = projection.IsValid ? projection.SegmentIndex + 1 : targetPointIndex;
            TrafficWaypoint waypoint = null;
            if (waypointIndex >= 0 && waypointIndex < pathWaypoints.Count)
            {
                waypoint = pathWaypoints[waypointIndex];
            }
            else if (pathWaypoints.Count > 0)
            {
                waypoint = pathWaypoints[Mathf.Clamp(targetPointIndex, 0, pathWaypoints.Count - 1)];
            }

            return waypoint != null && waypoint.MaxSpeed > 0f
                ? waypoint.MaxSpeed / 3.6f
                : 0f;
        }

        private void InitializeEpisodeState(Vector3 bodyPosition)
        {
            FrenetProjection projection = ProjectToPath(bodyPosition);
            lastPathProgressMeters = projection.IsValid ? projection.S : 0f;
            lastDestinationDistance = GetPlanarDistance(bodyPosition, finalDestination);
            lastMovementPosition = bodyPosition;
            lastMovementRealtime = Time.realtimeSinceStartup;
            cachedWheelBaseMeters = ComputeWheelBaseMeters();
        }

        private Vector3 GetDestinationTailPoint(Vector3 destination, Vector3 bodyPosition, List<Vector3> routePoints)
        {
            Vector3 anchor = routePoints != null && routePoints.Count > 0 ? routePoints[routePoints.Count - 1] : destination;
            Vector3 direction = Vector3.zero;
            if (routePoints != null && routePoints.Count >= 2)
            {
                direction = routePoints[routePoints.Count - 1] - routePoints[routePoints.Count - 2];
            }

            if (direction.sqrMagnitude < 0.001f)
            {
                direction = destination - bodyPosition;
            }

            direction.y = 0f;
            if (direction.sqrMagnitude < 0.001f)
            {
                direction = GetVehicleForward();
            }

            direction.y = 0f;
            if (direction.sqrMagnitude < 0.001f)
            {
                direction = Vector3.forward;
            }

            direction.Normalize();
            Vector3 toDestination = destination - anchor;
            toDestination.y = 0f;
            float destinationProjection = Mathf.Max(0f, Vector3.Dot(toDestination, direction));
            Vector3 tailPoint = anchor + direction * (destinationProjection + Mathf.Max(0f, destinationTailDistanceMeters));
            tailPoint.y = destination.y;
            return tailPoint;
        }

        private float ComputeWheelBaseMeters()
        {
            if (playerCar == null || playerCar.axleInfos == null || playerCar.axleInfos.Count == 0)
            {
                return DefaultWheelBaseMeters;
            }

            Transform root = GetVehicleRoot();
            float steeringZ = 0f;
            float nonSteeringZ = 0f;
            int steeringCount = 0;
            int nonSteeringCount = 0;

            for (int i = 0; i < playerCar.axleInfos.Count; i++)
            {
                AxleInfo axle = playerCar.axleInfos[i];
                if (axle == null)
                {
                    continue;
                }

                Vector3 center = Vector3.zero;
                int wheelCount = 0;
                if (axle.leftWheel != null)
                {
                    center += axle.leftWheel.transform.position;
                    wheelCount++;
                }

                if (axle.rightWheel != null)
                {
                    center += axle.rightWheel.transform.position;
                    wheelCount++;
                }

                if (wheelCount == 0)
                {
                    continue;
                }

                center /= wheelCount;
                float localZ = root != null ? root.InverseTransformPoint(center).z : transform.InverseTransformPoint(center).z;
                if (axle.steering)
                {
                    steeringZ += localZ;
                    steeringCount++;
                }
                else
                {
                    nonSteeringZ += localZ;
                    nonSteeringCount++;
                }
            }

            if (steeringCount == 0 || nonSteeringCount == 0)
            {
                return DefaultWheelBaseMeters;
            }

            float wheelBase = Mathf.Abs(steeringZ / steeringCount - nonSteeringZ / nonSteeringCount);
            return wheelBase > 0.1f ? wheelBase : DefaultWheelBaseMeters;
        }

        private float GetWheelBaseMeters()
        {
            return cachedWheelBaseMeters > 0.1f ? cachedWheelBaseMeters : DefaultWheelBaseMeters;
        }

        private float GetMaxSteerDegrees()
        {
            return playerCar != null && playerCar.maxSteeringAngle > 0f
                ? playerCar.maxSteeringAngle
                : Mathf.Max(1f, maxSteeringAngleForFullInput);
        }

        private float GetSignedHeadingError(Vector3 routeTangent)
        {
            Vector3 forward = GetVehicleForward();
            forward.y = 0f;
            routeTangent.y = 0f;
            if (forward.sqrMagnitude < 0.001f || routeTangent.sqrMagnitude < 0.001f)
            {
                return 0f;
            }

            return Vector3.SignedAngle(forward.normalized, routeTangent.normalized, Vector3.up);
        }

        private Vector3 GetVehicleForward()
        {
            Transform root = GetVehicleRoot();
            Vector3 forward = root != null ? root.forward : transform.forward;
            forward.y = 0f;
            return forward.sqrMagnitude > 0.001f ? forward.normalized : Vector3.forward;
        }

        private float GetLocalLateralVelocityMetersPerSecond()
        {
            ResolveReferences();
            if (vehicleRigidbody == null)
            {
                return 0f;
            }

#if UNITY_6000_0_OR_NEWER
            Vector3 worldVelocity = vehicleRigidbody.linearVelocity;
#else
            Vector3 worldVelocity = vehicleRigidbody.velocity;
#endif
            Transform root = GetVehicleRoot();
            if (root == null)
            {
                return 0f;
            }

            Vector3 localVelocity = root.InverseTransformDirection(worldVelocity);
            return localVelocity.x;
        }

        private Vector3 InverseVehiclePoint(Vector3 worldPoint)
        {
            Transform root = GetVehicleRoot();
            return root != null ? root.InverseTransformPoint(worldPoint) : transform.InverseTransformPoint(worldPoint);
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
            return vehicleRoot != null ? vehicleRoot : transform;
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

        private void RecordStat(string key, float value, StatAggregationMethod aggregationMethod = StatAggregationMethod.Average)
        {
            Academy.Instance.StatsRecorder.Add(key, value, aggregationMethod);
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
            minLookaheadMeters = Mathf.Max(0.1f, minLookaheadMeters);
            maxLookaheadMeters = Mathf.Max(minLookaheadMeters + 0.1f, maxLookaheadMeters);
            zeroActionLookaheadNormalized = Mathf.Clamp01(zeroActionLookaheadNormalized);
            teacherBaseLookaheadMeters = Mathf.Max(0f, teacherBaseLookaheadMeters);
            teacherSpeedGainSeconds = Mathf.Max(0f, teacherSpeedGainSeconds);
            teacherCurvatureGain = Mathf.Max(0f, teacherCurvatureGain);
            baseCruiseSpeedMetersPerSecond = Mathf.Max(0.5f, baseCruiseSpeedMetersPerSecond);
            maxPolicySpeedMetersPerSecond = Mathf.Max(0f, maxPolicySpeedMetersPerSecond);
            minTargetSpeedRatio = Mathf.Clamp01(minTargetSpeedRatio);
            maxTargetSpeedRatio = Mathf.Clamp(maxTargetSpeedRatio, minTargetSpeedRatio, 1f);
            zeroActionSpeedNormalized = Mathf.Clamp01(zeroActionSpeedNormalized);
            speedLimitBrakeInput = Mathf.Clamp(speedLimitBrakeInput, -1f, 0f);
            throttleInputSmoothing = Mathf.Max(0.1f, throttleInputSmoothing);
            steeringInputSmoothing = Mathf.Max(0.1f, steeringInputSmoothing);
            curvatureSmoothingBeta = Mathf.Clamp(curvatureSmoothingBeta, 0.01f, 1f);
            maxSteeringAngleForFullInput = Mathf.Max(1f, maxSteeringAngleForFullInput);
            slowDownDistanceMeters = Mathf.Max(1f, slowDownDistanceMeters);
            destinationApproachBrakeInput = Mathf.Clamp(destinationApproachBrakeInput, -1f, 0f);
            destinationApproachCreepThrottle = Mathf.Clamp01(destinationApproachCreepThrottle);
            destinationApproachRecoveryThrottle = Mathf.Clamp01(destinationApproachRecoveryThrottle);
            maxObservationSpeedMetersPerSecond = Mathf.Max(0.5f, maxObservationSpeedMetersPerSecond);
            maxObservationCurvature = Mathf.Max(0.001f, maxObservationCurvature);
            maxCrossTrackErrorMeters = Mathf.Max(0.1f, maxCrossTrackErrorMeters);
            midCurvatureHorizonMeters = Mathf.Max(0f, midCurvatureHorizonMeters);
            farCurvatureHorizonMeters = Mathf.Max(midCurvatureHorizonMeters, farCurvatureHorizonMeters);
            maxCurvatureHorizonMeters = Mathf.Max(farCurvatureHorizonMeters, maxCurvatureHorizonMeters);
            curvatureSampleSpacingMeters = Mathf.Max(0.5f, curvatureSampleSpacingMeters);
            speedRewardPerSecond = Mathf.Max(0f, speedRewardPerSecond);
            progressRewardPerMeter = Mathf.Max(0f, progressRewardPerMeter);
            waypointPassedReward = Mathf.Max(0f, waypointPassedReward);
            destinationProgressRewardPerMeter = Mathf.Max(0f, destinationProgressRewardPerMeter);
            lookaheadTeacherPenaltyPerMeter = Mathf.Max(0f, lookaheadTeacherPenaltyPerMeter);
            lookaheadChangePenaltyPerMeter = Mathf.Max(0f, lookaheadChangePenaltyPerMeter);
            curvaturePenaltyPerSecond = Mathf.Max(0f, curvaturePenaltyPerSecond);
            lateralErrorPenaltyPerMeter = Mathf.Max(0f, lateralErrorPenaltyPerMeter);
            localLateralVelocityPenaltyPerMeter = Mathf.Max(0f, localLateralVelocityPenaltyPerMeter);
            localLateralVelocityCurvatureGain = Mathf.Max(0f, localLateralVelocityCurvatureGain);
            headingErrorPenaltyPerMeter = Mathf.Max(0f, headingErrorPenaltyPerMeter);
            overspeedPenaltyPerSecond = Mathf.Max(0f, overspeedPenaltyPerSecond);
            stallPenaltyPerSecond = Mathf.Max(0f, stallPenaltyPerSecond);
            destinationReward = Mathf.Max(0f, destinationReward);
            collisionPenalty = Mathf.Min(0f, collisionPenalty);
            assignedRouteExitPenalty = Mathf.Min(0f, assignedRouteExitPenalty);
            referenceFaultPenalty = Mathf.Min(0f, referenceFaultPenalty);
            hardCrossTrackLimitMeters = Mathf.Max(0.5f, hardCrossTrackLimitMeters);
            noMovementTimeoutRealSeconds = Mathf.Max(0.5f, noMovementTimeoutRealSeconds);
            minimumMovementMeters = Mathf.Max(0.01f, minimumMovementMeters);
            stallSpeedThresholdMetersPerSecond = Mathf.Max(0f, stallSpeedThresholdMetersPerSecond);
            stallGraceSeconds = Mathf.Max(0f, stallGraceSeconds);
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
    }

    public class DRTPPOPurePursuitVehicleCollisionRelay : MonoBehaviour
    {
        private DRTPPOPurePursuitVehicleDriver owner;

        public void Configure(DRTPPOPurePursuitVehicleDriver newOwner)
        {
            owner = newOwner;
        }

        private void OnCollisionEnter(Collision collision)
        {
            owner?.SendMessage("OnCollisionEnter", collision, SendMessageOptions.DontRequireReceiver);
        }
    }
}
