using System.Collections.Generic;
using System.Text;
using Unity.Barracuda;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace DRT
{
    [RequireComponent(typeof(BehaviorParameters))]
    public class DRTNextStopSelector : Agent
    {
        private const string BehaviorName = "DRTNextStopPPO";
        private const int GlobalObservationCount = 6;
        private const int ObservationsPerStop = 8;

        [HideInInspector, SerializeField] private DRTBusController busController;
        [HideInInspector, SerializeField, Min(2)] private int maxStops = 16;
        [HideInInspector, SerializeField] private bool skipCurrentStop = true;
        [HideInInspector, SerializeField] private float episodeLengthSeconds = 3000f;
        [HideInInspector, SerializeField] private float maxDistanceForObservation = 500f;
        [HideInInspector, SerializeField] private float maxTravelSecondsForObservation = 1800f;
        [HideInInspector, SerializeField] private float maxWaitSecondsForObservation = 1800f;
        [HideInInspector, SerializeField] private float maxDecisionWaitSeconds = 1f;

        [Header("Policy")]
        [SerializeField, InspectorName("Mode")] private DRTNextStopPolicy nextStopPolicy = DRTNextStopPolicy.MLAgentsTraining;
        [Tooltip("Used only when Next Stop Policy is ONNX Inference. Import the .onnx under Assets first, then assign it here.")]
        [SerializeField, InspectorName("ONNX Model")] private NNModel onnxInferenceModel;
        [HideInInspector, SerializeField] private InferenceDevice onnxInferenceDevice = InferenceDevice.Default;

        [Header("Reward")]
        [SerializeField, Min(0f), InspectorName("Unboarded Penalty x")] private float unboardedPassengerPenaltyWeight = 1f;
        [SerializeField, Min(0f), InspectorName("Boarding Reward x")] private float boardingRewardWeight = 1f;
        [SerializeField, Min(0f), InspectorName("Dropoff Reward x")] private float dropoffRewardWeight = 1f;
        [SerializeField, Min(0f), InspectorName("No Interaction Penalty x")] private float noInteractionPenaltyWeight = 2f;
        [SerializeField, Min(1f), InspectorName("Accept Wait (s)")] private float acceptableWaitSeconds = 100f;
        [SerializeField, Min(1f), InspectorName("Accept Wait x")] private float acceptableWaitRewardMultiplier = 1f;
        [HideInInspector, SerializeField] private float networkDistanceUnitsPerMinute = 100f;
        [HideInInspector, SerializeField] private float minimumNetworkAverageReward = 0.01f;
        [HideInInspector, SerializeField] private float invalidActionPenalty = 0f;
        [HideInInspector, SerializeField] private bool logReward = true;

        [HideInInspector, SerializeField] private bool logDecision = true;
        [HideInInspector, SerializeField] private bool logPolicyAction = true;

        private IReadOnlyList<DRTStop> decisionStops;
        private DRTPassengerManager decisionPassengerManager;
        private int decisionCurrentStopId;
        private float decisionEpisodeTime;
        private bool decisionPending;
        private bool decisionReady;
        private int selectedStopId = -1;
        private int lastSelectedStopId = -1;
        private int episodeDecisionCount;
        private int episodeStopArrivalCount;
        private float episodeRewardTotal;
        private readonly List<int> allStationRoute = new List<int>();
        private readonly HashSet<string> allStationCompletedPairKeys = new HashSet<string>();
        private string allStationRouteSignature = string.Empty;
        private int allStationRouteCursor;
        private int allStationRouteStartStopId;
        private bool allStationRunComplete;
        private bool loggedDriveTrainingBypass;

        public float MaxDecisionWaitSeconds => maxDecisionWaitSeconds;
        public int LastSelectedStopId => lastSelectedStopId;
        public DRTNextStopPolicy NextStopPolicy => nextStopPolicy;
        public string NextStopPolicyName => nextStopPolicy.ToString();
        public bool UsesMlAgentsDecisionPolicy =>
            !BypassMlAgentsDuringDriveTraining &&
            nextStopPolicy != DRTNextStopPolicy.VanillaSequential &&
            nextStopPolicy != DRTNextStopPolicy.GreedyNearestFeasible &&
            nextStopPolicy != DRTNextStopPolicy.Fifo &&
            nextStopPolicy != DRTNextStopPolicy.AllStationRunner;
        public bool UsesGreedyNearestFeasible => nextStopPolicy == DRTNextStopPolicy.GreedyNearestFeasible;
        public bool UsesAllStationRunner => nextStopPolicy == DRTNextStopPolicy.AllStationRunner;
        public bool IsAllStationRunComplete => UsesAllStationRunner && allStationRunComplete;

        private int ObservationSize => GlobalObservationCount + maxStops * ObservationsPerStop;
        private bool BypassMlAgentsDuringDriveTraining =>
            nextStopPolicy == DRTNextStopPolicy.MLAgentsTraining &&
            busController != null &&
            busController.UsesPPOVehicleControl &&
            busController.PPODrivePolicy == DRTPPODrivePolicy.MLAgentsTraining;

        private void Awake()
        {
            ResolveBusController();
            ConfigureBehaviorParameters();
        }

        public override void Initialize()
        {
            ResolveBusController();
            ConfigureBehaviorParameters();
        }

        public void Configure(DRTBusController newBusController)
        {
            busController = newBusController;
            if (busController != null)
            {
                episodeLengthSeconds = busController.EpisodeLengthSeconds;
            }

            ConfigureBehaviorParameters();
        }

        public void ConfigureDiagnostics(bool newLogReward, bool newLogDecision, bool newLogPolicyAction)
        {
            logReward = newLogReward;
            logDecision = newLogDecision;
            logPolicyAction = newLogPolicyAction;
        }

        public void ConfigureAllStationResume(IEnumerable<string> completedPairKeys, int routeStartStopId)
        {
            allStationCompletedPairKeys.Clear();
            if (completedPairKeys != null)
            {
                foreach (string pairKey in completedPairKeys)
                {
                    if (!string.IsNullOrWhiteSpace(pairKey))
                    {
                        allStationCompletedPairKeys.Add(pairKey);
                    }
                }
            }

            allStationRouteStartStopId = Mathf.Max(1, routeStartStopId);
            ResetAllStationRunnerState();
        }

        public bool TryGetAllStationResumeStart(
            IReadOnlyList<DRTStop> stops,
            int fallbackStartStopId,
            out int resumeStartStopId,
            out int resumeTargetStopId,
            out int skippedEdgeCount,
            out int totalEdgeCount)
        {
            resumeStartStopId = Mathf.Max(1, fallbackStartStopId);
            resumeTargetStopId = -1;
            skippedEdgeCount = 0;
            totalEdgeCount = 0;

            List<int> route = BuildAllStationRouteForResume(stops, fallbackStartStopId);
            totalEdgeCount = Mathf.Max(0, route.Count - 1);
            if (route.Count < 2)
            {
                return false;
            }

            skippedEdgeCount = CountCompletedRoutePrefix(route);
            if (skippedEdgeCount >= route.Count - 1)
            {
                resumeStartStopId = route[route.Count - 1];
                return false;
            }

            resumeStartStopId = route[skippedEdgeCount];
            resumeTargetStopId = route[skippedEdgeCount + 1];
            return true;
        }

        public override void OnEpisodeBegin()
        {
            if (BypassMlAgentsDuringDriveTraining)
            {
                CancelDecision();
                return;
            }

            CancelDecision();
            episodeDecisionCount = 0;
            episodeStopArrivalCount = 0;
            episodeRewardTotal = 0f;
            ResetAllStationRunnerState();
            ResolveBusController();
            busController?.ResetEpisodeFromAgent();
        }

        public bool BeginDecision(
            int currentStopId,
            IReadOnlyList<DRTStop> stops,
            DRTPassengerManager passengerManager,
            float currentEpisodeTime)
        {
            if (!UsesMlAgentsDecisionPolicy)
            {
                return false;
            }

            if (stops == null || stops.Count == 0 || passengerManager == null)
            {
                return false;
            }

            ConfigureBehaviorParameters();
            decisionStops = stops;
            decisionPassengerManager = passengerManager;
            decisionCurrentStopId = currentStopId;
            decisionEpisodeTime = currentEpisodeTime;
            selectedStopId = -1;
            decisionReady = false;
            decisionPending = true;
            episodeDecisionCount++;

            RecordStat("DRT/DecisionRequested", 1f, StatAggregationMethod.Sum);
            RecordStat("DRT/EpisodeTimeAtDecision", currentEpisodeTime);
            RecordStat("DRT/EpisodeDecisionCount", episodeDecisionCount, StatAggregationMethod.MostRecent);

            RequestDecision();
            return true;
        }

        public bool TryConsumeDecision(out int stopId)
        {
            if (!decisionReady)
            {
                stopId = -1;
                return false;
            }

            stopId = selectedStopId;
            decisionReady = false;
            return true;
        }

        public void CancelDecision()
        {
            decisionPending = false;
            decisionReady = false;
            selectedStopId = -1;
        }

        public void RecordStopArrival(DRTStopProcessResult result, float currentEpisodeTime)
        {
            if (UsesAllStationRunner)
            {
                episodeStopArrivalCount++;
                if (allStationRoute.Count > 0 &&
                    allStationRouteCursor >= allStationRoute.Count - 1 &&
                    allStationRouteCursor < allStationRoute.Count &&
                    result.StopId == allStationRoute[allStationRouteCursor])
                {
                    allStationRunComplete = true;
                }

                RecordStat("DRT/StopArrivals", 1f, StatAggregationMethod.Sum);
                RecordStat("DRT/EpisodeStopArrivalCount", episodeStopArrivalCount, StatAggregationMethod.MostRecent);
                return;
            }

            if (decisionPassengerManager == null)
            {
                return;
            }

            float networkAverageReward = GetNetworkAverageAccessReward();
            float unboardedPenalty = GetUnboardedPassengerPenalty(currentEpisodeTime);
            float boardingReward = GetBoardingReward(result.StopId, currentEpisodeTime, networkAverageReward);
            float dropoffReward = GetDropoffReward(result.StopId, currentEpisodeTime, networkAverageReward);
            float noInteractionPenalty = GetNoInteractionPenalty(result, networkAverageReward);
            float reward = boardingReward + dropoffReward - unboardedPenalty - noInteractionPenalty;

            AddReward(reward);
            episodeRewardTotal += reward;
            episodeStopArrivalCount++;

            RecordStat("DRT/Reward/StopTotal", reward);
            RecordStat("DRT/Reward/Boarding", boardingReward);
            RecordStat("DRT/Reward/Dropoff", dropoffReward);
            RecordStat("DRT/Reward/UnboardedPenalty", unboardedPenalty);
            RecordStat("DRT/Reward/NoInteractionPenalty", noInteractionPenalty);
            RecordStat("DRT/Reward/EpisodeTotal", episodeRewardTotal, StatAggregationMethod.MostRecent);
            RecordStat("DRT/StopArrivals", 1f, StatAggregationMethod.Sum);
            RecordStat("DRT/EpisodeStopArrivalCount", episodeStopArrivalCount, StatAggregationMethod.MostRecent);

            if (logReward && !ShouldSuppressUnityLogs())
            {
                Debug.Log(
                    $"[NEXTSTOPSELECTOR] PaperReward stop={result.StopId} t={currentEpisodeTime:0.0}s " +
                    $"reward={reward:0.000}, board={boardingReward:0.000}, drop={dropoffReward:0.000}, " +
                    $"unboardedPenalty={unboardedPenalty:0.000}, noInteractionPenalty={noInteractionPenalty:0.000}, " +
                    $"networkAvg={networkAverageReward:0.000}");
            }
        }

        public void RecordExternalPenalty(float penalty, string reason)
        {
            if (Mathf.Approximately(penalty, 0f))
            {
                return;
            }

            AddReward(penalty);
            episodeRewardTotal += penalty;
            RecordStat("DRT/Reward/ExternalPenalty", penalty);
            RecordStat("DRT/Reward/EpisodeTotal", episodeRewardTotal, StatAggregationMethod.MostRecent);

            if (logReward && !ShouldSuppressUnityLogs())
            {
                Debug.Log($"[NEXTSTOPSELECTOR] ExternalPenalty reward={penalty:0.000}, reason={reason}");
            }
        }

        public void NotifyEpisodeFinished(
            bool completedAllRequests,
            float travelDistanceMeters,
            float averageWaitSeconds,
            float averageRideSeconds,
            float serviceRate,
            int completedPassengerCount)
        {
            RecordStat("DRT/EpisodeDecisionCount", episodeDecisionCount, StatAggregationMethod.MostRecent);
            RecordStat("DRT/EpisodeStopArrivalCount", episodeStopArrivalCount, StatAggregationMethod.MostRecent);
            RecordStat("DRT/Reward/EpisodeTotal", episodeRewardTotal, StatAggregationMethod.MostRecent);
            RecordStat("DRT/EpisodeCompletedAllRequests", completedAllRequests ? 1f : 0f, StatAggregationMethod.MostRecent);
            RecordStat("DRT/EpisodeTravelDistanceMeters", travelDistanceMeters, StatAggregationMethod.MostRecent);
            RecordStat("DRT/EpisodeAverageWaitSeconds", averageWaitSeconds, StatAggregationMethod.MostRecent);
            RecordStat("DRT/EpisodeAverageRideSeconds", averageRideSeconds, StatAggregationMethod.MostRecent);
            RecordStat("DRT/EpisodeServiceRate", serviceRate, StatAggregationMethod.MostRecent);
            RecordStat("DRT/EpisodeCompletedPassengers", completedPassengerCount, StatAggregationMethod.MostRecent);
            CancelDecision();
            if (!UsesAllStationRunner)
            {
                EndEpisode();
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            int stopCount = decisionStops != null ? Mathf.Min(decisionStops.Count, maxStops) : 0;
            int onBoardCount = decisionPassengerManager != null ? decisionPassengerManager.GetOnBoardCount() : 0;
            int waitingCount = decisionPassengerManager != null ? decisionPassengerManager.GetWaitingCount(decisionEpisodeTime) : 0;
            int capacity = decisionPassengerManager != null ? Mathf.Max(1, decisionPassengerManager.BusCapacity) : 1;

            sensor.AddObservation(NormalizeStopIndex(decisionCurrentStopId));
            sensor.AddObservation(Mathf.Clamp01(decisionEpisodeTime / Mathf.Max(1f, episodeLengthSeconds)));
            sensor.AddObservation(Mathf.Clamp01((float)waitingCount / capacity));
            sensor.AddObservation(Mathf.Clamp01((float)onBoardCount / capacity));
            sensor.AddObservation(Mathf.Clamp01((float)(capacity - onBoardCount) / capacity));
            sensor.AddObservation(decisionPassengerManager != null ? decisionPassengerManager.GetServiceRate() : 0f);

            DRTStop currentStop = FindStop(decisionStops, decisionCurrentStopId);

            for (int i = 0; i < maxStops; i++)
            {
                DRTStop stop = i < stopCount ? decisionStops[i] : null;
                bool valid = stop != null;
                bool isCurrent = valid && stop.StopId == decisionCurrentStopId;

                int waitingAtStop = valid && decisionPassengerManager != null
                    ? decisionPassengerManager.GetWaitingCountAtStop(stop.StopId, decisionEpisodeTime)
                    : 0;
                int dropOffAtStop = valid && decisionPassengerManager != null
                    ? decisionPassengerManager.GetOnBoardDestinationCount(stop.StopId)
                    : 0;
                int scheduledAtStop = valid && decisionPassengerManager != null
                    ? decisionPassengerManager.GetScheduledCountAtStop(stop.StopId, decisionEpisodeTime)
                    : 0;

                float travelFeature = valid && currentStop != null
                    ? GetCandidateTravelFeature(currentStop.StopId, stop.StopId, currentStop, stop)
                    : 1f;

                GetStopPassengerTimeFeatures(
                    valid ? stop.StopId : -1,
                    decisionEpisodeTime,
                    out float maxWaitSeconds,
                    out float maxRideSeconds);

                sensor.AddObservation(valid ? 1f : 0f);
                sensor.AddObservation(isCurrent ? 1f : 0f);
                sensor.AddObservation(travelFeature);
                sensor.AddObservation(Mathf.Clamp01((float)waitingAtStop / capacity));
                sensor.AddObservation(Mathf.Clamp01((float)dropOffAtStop / capacity));
                sensor.AddObservation(Mathf.Clamp01((float)scheduledAtStop / capacity));
                sensor.AddObservation(Mathf.Clamp01(maxWaitSeconds / Mathf.Max(1f, maxWaitSecondsForObservation)));
                sensor.AddObservation(Mathf.Clamp01(maxRideSeconds / Mathf.Max(1f, maxWaitSecondsForObservation)));
            }
        }

        public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            int stopCount = decisionStops != null ? Mathf.Min(decisionStops.Count, maxStops) : 0;
            for (int i = 0; i < maxStops; i++)
            {
                bool enabled = i < stopCount && decisionStops[i] != null;
                if (enabled && skipCurrentStop && stopCount > 1)
                {
                    enabled = decisionStops[i].StopId != decisionCurrentStopId;
                }

                actionMask.SetActionEnabled(0, i, enabled);
            }
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            if (!decisionPending)
            {
                return;
            }

            int actionIndex = actionBuffers.DiscreteActions.Length > 0 ? actionBuffers.DiscreteActions[0] : -1;
            int rawStopId = GetStopIdFromAction(actionIndex, false);
            selectedStopId = GetStopIdFromAction(actionIndex, true);

            if (selectedStopId < 1)
            {
                if (!Mathf.Approximately(invalidActionPenalty, 0f))
                {
                    AddReward(invalidActionPenalty);
                    episodeRewardTotal += invalidActionPenalty;
                    RecordStat("DRT/Reward/InvalidActionPenalty", invalidActionPenalty);
                    RecordStat("DRT/Reward/EpisodeTotal", episodeRewardTotal, StatAggregationMethod.MostRecent);
                }
            }

            LogPolicyAction(actionIndex, rawStopId, selectedStopId);
            lastSelectedStopId = selectedStopId;
            decisionPending = false;
            decisionReady = true;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActionsOut = actionsOut.DiscreteActions;
            int heuristicStopId;
            if (nextStopPolicy == DRTNextStopPolicy.AllStationRunner)
            {
                heuristicStopId = SelectAllStationRunnerStopId(
                    decisionCurrentStopId,
                    decisionStops,
                    decisionEpisodeTime);
            }
            else if (nextStopPolicy == DRTNextStopPolicy.VanillaSequential)
            {
                heuristicStopId = SelectVanillaSequentialStopId(
                    decisionCurrentStopId,
                    decisionStops,
                    decisionPassengerManager,
                    decisionEpisodeTime);
            }
            else if (nextStopPolicy == DRTNextStopPolicy.GreedyNearestFeasible)
            {
                heuristicStopId = SelectGreedyNearestFeasibleStopId(
                    decisionCurrentStopId,
                    decisionStops,
                    decisionPassengerManager,
                    decisionEpisodeTime);
            }
            else if (nextStopPolicy == DRTNextStopPolicy.Fifo)
            {
                heuristicStopId = SelectFifoStopId(
                    decisionCurrentStopId,
                    decisionStops,
                    decisionPassengerManager,
                    decisionEpisodeTime);
            }
            else
            {
                discreteActionsOut[0] = -1;
                return;
            }

            discreteActionsOut[0] = FindActionIndexForStop(heuristicStopId);
        }

        public int SelectVanillaSequentialStopId(
            int currentStopId,
            IReadOnlyList<DRTStop> stops,
            DRTPassengerManager passengerManager,
            float currentEpisodeTime)
        {
            int sequentialStopId = GetNextSequentialStopId(currentStopId, stops);
            lastSelectedStopId = sequentialStopId;
            LogSelectedStop(
                "vanilla-sequential",
                sequentialStopId,
                currentStopId,
                currentEpisodeTime,
                passengerManager,
                0f,
                "fixed stop-id loop");
            return sequentialStopId;
        }

        public int SelectGreedyNearestFeasibleStopId(
            int currentStopId,
            IReadOnlyList<DRTStop> stops,
            DRTPassengerManager passengerManager,
            float currentEpisodeTime)
        {
            if (stops == null || stops.Count == 0 || passengerManager == null)
            {
                return SelectVanillaSequentialStopId(
                    currentStopId,
                    stops,
                    passengerManager,
                    currentEpisodeTime);
            }

            DRTStop currentStop = FindStop(stops, currentStopId);
            int currentOnBoard = passengerManager.GetOnBoardCount();
            int freeSeats = Mathf.Max(0, passengerManager.BusCapacity - currentOnBoard);
            int bestStopId = -1;
            float bestTravelSeconds = float.MaxValue;
            int bestDropOffCount = 0;
            int bestInteractionCount = 0;
            float bestOldestWaitSeconds = -1f;
            var scoreSummary = new StringBuilder();

            for (int i = 0; i < stops.Count; i++)
            {
                DRTStop candidateStop = stops[i];
                if (candidateStop == null)
                {
                    continue;
                }

                int candidateStopId = candidateStop.StopId;
                if (skipCurrentStop && stops.Count > 1 && candidateStopId == currentStopId)
                {
                    continue;
                }

                int dropOffCount = passengerManager.GetOnBoardDestinationCount(candidateStopId);
                int waitingCount = passengerManager.GetWaitingCountAtStop(candidateStopId, currentEpisodeTime);
                int boardableCount = Mathf.Min(waitingCount, freeSeats + dropOffCount);
                int interactionCount = dropOffCount + boardableCount;
                if (interactionCount <= 0)
                {
                    continue;
                }

                float travelSeconds = GetCandidateTravelSeconds(
                    currentStopId,
                    candidateStopId,
                    currentStop,
                    candidateStop);
                float oldestWaitSeconds = GetOldestWaitingSecondsAtStop(
                    passengerManager,
                    candidateStopId,
                    currentEpisodeTime);

                if (scoreSummary.Length > 0)
                {
                    scoreSummary.Append("; ");
                }

                scoreSummary
                    .Append("S").Append(candidateStopId)
                    .Append("(travel=").Append(travelSeconds.ToString("0.0"))
                    .Append("s,drop=").Append(dropOffCount)
                    .Append(",board=").Append(boardableCount)
                    .Append(",oldestWait=").Append(oldestWaitSeconds.ToString("0.0"))
                    .Append("s)");

                bool shouldSelect = bestStopId < 1 ||
                                    travelSeconds < bestTravelSeconds - 0.001f;

                if (!shouldSelect && Mathf.Abs(travelSeconds - bestTravelSeconds) <= 0.001f)
                {
                    bool candidateHasDropOff = dropOffCount > 0;
                    bool bestHasDropOff = bestDropOffCount > 0;
                    shouldSelect = candidateHasDropOff != bestHasDropOff
                        ? candidateHasDropOff
                        : interactionCount > bestInteractionCount ||
                          (interactionCount == bestInteractionCount &&
                           (oldestWaitSeconds > bestOldestWaitSeconds + 0.001f ||
                            (Mathf.Abs(oldestWaitSeconds - bestOldestWaitSeconds) <= 0.001f &&
                             candidateStopId < bestStopId)));
                }

                if (!shouldSelect)
                {
                    continue;
                }

                bestStopId = candidateStopId;
                bestTravelSeconds = travelSeconds;
                bestDropOffCount = dropOffCount;
                bestInteractionCount = interactionCount;
                bestOldestWaitSeconds = oldestWaitSeconds;
            }

            if (bestStopId < 1)
            {
                if (freeSeats > 0 &&
                    passengerManager.TryGetNextScheduledOrigin(currentEpisodeTime, out int nextScheduledOriginStopId) &&
                    FindStop(stops, nextScheduledOriginStopId) != null &&
                    (!skipCurrentStop || stops.Count <= 1 || nextScheduledOriginStopId != currentStopId))
                {
                    bestStopId = nextScheduledOriginStopId;
                    bestTravelSeconds = GetCandidateTravelSeconds(
                        currentStopId,
                        bestStopId,
                        currentStop,
                        FindStop(stops, bestStopId));
                    scoreSummary.Append("no immediate interaction; fallback=next scheduled origin");
                }
                else
                {
                    bestStopId = GetNextSequentialStopId(currentStopId, stops);
                    bestTravelSeconds = GetCandidateTravelSeconds(
                        currentStopId,
                        bestStopId,
                        currentStop,
                        FindStop(stops, bestStopId));
                    scoreSummary.Append("no immediate interaction; fallback=vanilla sequential");
                }
            }

            lastSelectedStopId = bestStopId;
            LogSelectedStop(
                "greedy-nearest-feasible",
                bestStopId,
                currentStopId,
                currentEpisodeTime,
                passengerManager,
                bestTravelSeconds,
                scoreSummary.ToString());
            return bestStopId;
        }

        public int SelectFifoStopId(
            int currentStopId,
            IReadOnlyList<DRTStop> stops,
            DRTPassengerManager passengerManager,
            float currentEpisodeTime)
        {
            if (stops == null || stops.Count == 0 || passengerManager == null)
            {
                return SelectVanillaSequentialStopId(
                    currentStopId,
                    stops,
                    passengerManager,
                    currentEpisodeTime);
            }

            DRTStop currentStop = FindStop(stops, currentStopId);
            int currentOnBoard = passengerManager.GetOnBoardCount();
            int freeSeats = Mathf.Max(0, passengerManager.BusCapacity - currentOnBoard);
            int selectedStopId = -1;
            float selectedTravelSeconds = 0f;
            var scoreSummary = new StringBuilder();

            if (freeSeats > 0)
            {
                float earliestRequestTime = float.PositiveInfinity;
                int earliestPassengerId = int.MaxValue;
                var requests = passengerManager.Requests;
                for (int i = 0; i < requests.Count; i++)
                {
                    var request = requests[i];
                    if (request == null ||
                        request.Status != DRTPassengerStatus.Waiting ||
                        (skipCurrentStop && stops.Count > 1 && request.OriginStopId == currentStopId) ||
                        FindStop(stops, request.OriginStopId) == null)
                    {
                        continue;
                    }

                    if (request.RequestTimeSeconds < earliestRequestTime - 0.001f ||
                        (Mathf.Abs(request.RequestTimeSeconds - earliestRequestTime) <= 0.001f &&
                         request.PassengerId < earliestPassengerId))
                    {
                        earliestRequestTime = request.RequestTimeSeconds;
                        earliestPassengerId = request.PassengerId;
                        selectedStopId = request.OriginStopId;
                    }
                }

                if (selectedStopId >= 1)
                {
                    selectedTravelSeconds = GetCandidateTravelSeconds(
                        currentStopId,
                        selectedStopId,
                        currentStop,
                        FindStop(stops, selectedStopId));
                    scoreSummary
                        .Append("fifo pickup passenger=")
                        .Append(earliestPassengerId)
                        .Append(", requestTime=")
                        .Append(earliestRequestTime.ToString("0.0"))
                        .Append("s");
                }
            }

            if (selectedStopId < 1)
            {
                float bestTravelSeconds = float.MaxValue;
                for (int i = 0; i < stops.Count; i++)
                {
                    DRTStop candidateStop = stops[i];
                    if (candidateStop == null)
                    {
                        continue;
                    }

                    int candidateStopId = candidateStop.StopId;
                    if (skipCurrentStop && stops.Count > 1 && candidateStopId == currentStopId)
                    {
                        continue;
                    }

                    int dropOffCount = passengerManager.GetOnBoardDestinationCount(candidateStopId);
                    if (dropOffCount <= 0)
                    {
                        continue;
                    }

                    float travelSeconds = GetCandidateTravelSeconds(
                        currentStopId,
                        candidateStopId,
                        currentStop,
                        candidateStop);
                    if (travelSeconds < bestTravelSeconds - 0.001f ||
                        (Mathf.Abs(travelSeconds - bestTravelSeconds) <= 0.001f &&
                         (selectedStopId < 1 || candidateStopId < selectedStopId)))
                    {
                        selectedStopId = candidateStopId;
                        selectedTravelSeconds = travelSeconds;
                        bestTravelSeconds = travelSeconds;
                    }
                }

                if (selectedStopId >= 1)
                {
                    if (scoreSummary.Length > 0)
                    {
                        scoreSummary.Append("; ");
                    }

                    scoreSummary.Append("fallback=nearest onboard dropoff");
                }
            }

            if (selectedStopId < 1)
            {
                selectedStopId = GetNextSequentialStopId(currentStopId, stops);
                selectedTravelSeconds = GetCandidateTravelSeconds(
                    currentStopId,
                    selectedStopId,
                    currentStop,
                    FindStop(stops, selectedStopId));
                scoreSummary.Append("fallback=vanilla sequential");
            }

            lastSelectedStopId = selectedStopId;
            LogSelectedStop(
                "fifo",
                selectedStopId,
                currentStopId,
                currentEpisodeTime,
                passengerManager,
                selectedTravelSeconds,
                scoreSummary.ToString());
            return selectedStopId;
        }

        public int SelectAllStationRunnerStopId(
            int currentStopId,
            IReadOnlyList<DRTStop> stops,
            float currentEpisodeTime)
        {
            if (stops == null || stops.Count < 2)
            {
                allStationRunComplete = true;
                return -1;
            }

            EnsureAllStationRoute(currentStopId, stops);
            AlignAllStationCursor(currentStopId);

            if (allStationRunComplete || allStationRouteCursor >= allStationRoute.Count - 1)
            {
                allStationRunComplete = true;
                return -1;
            }

            int nextStopId = allStationRoute[allStationRouteCursor + 1];
            allStationRouteCursor++;
            lastSelectedStopId = nextStopId;

            if (logDecision && !ShouldSuppressUnityLogs())
            {
                Debug.Log(
                    $"[NEXTSTOPSELECTOR] AllStationRunner t={currentEpisodeTime:0.0}s " +
                    $"current={currentStopId} selected={nextStopId} " +
                    $"progress={allStationRouteCursor}/{Mathf.Max(0, allStationRoute.Count - 1)}");
            }

            return nextStopId;
        }

        private void LogSelectedStop(
            string reason,
            int selectedStopId,
            int currentStopId,
            float currentEpisodeTime,
            DRTPassengerManager passengerManager,
            float bestScore,
            string scoreSummary)
        {
            if (!logDecision || ShouldSuppressUnityLogs())
            {
                return;
            }

            if (passengerManager == null || selectedStopId < 1)
            {
                return;
            }

            int waiting = passengerManager.GetWaitingCountAtStop(selectedStopId, currentEpisodeTime);
            int dropOff = passengerManager.GetOnBoardDestinationCount(selectedStopId);
            int scheduled = passengerManager.GetScheduledCountAtStop(selectedStopId, currentEpisodeTime);

            Debug.Log(
                $"[NEXTSTOPSELECTOR] t={currentEpisodeTime:0.0}s current={currentStopId} selected={selectedStopId} " +
                $"reason={reason} selectedDemand(wait={waiting},drop={dropOff},future={scheduled}) " +
                $"bestScore={bestScore:0.00} scores=[{scoreSummary}]");
        }

        private int GetStopIdFromAction(int actionIndex, bool enforceCurrentStopMask = true)
        {
            if (decisionStops == null || actionIndex < 0 || actionIndex >= decisionStops.Count || actionIndex >= maxStops)
            {
                return -1;
            }

            DRTStop stop = decisionStops[actionIndex];
            if (stop == null)
            {
                return -1;
            }

            if (enforceCurrentStopMask && skipCurrentStop && decisionStops.Count > 1 && stop.StopId == decisionCurrentStopId)
            {
                return -1;
            }

            return stop.StopId;
        }

        private void LogPolicyAction(int actionIndex, int rawStopId, int finalStopId)
        {
            if (!logPolicyAction || ShouldSuppressUnityLogs())
            {
                return;
            }

            int waitingTotal = decisionPassengerManager != null
                ? decisionPassengerManager.GetWaitingCount(decisionEpisodeTime)
                : 0;
            int onBoard = decisionPassengerManager != null
                ? decisionPassengerManager.GetOnBoardCount()
                : 0;

            Debug.Log(
                $"[NEXTSTOPSELECTOR] PolicyAction t={decisionEpisodeTime:0.0}s current={decisionCurrentStopId} " +
                $"rawAction={actionIndex}, rawStop={FormatStopId(rawStopId)}, selected={FormatStopId(finalStopId)}, " +
                $"validAction={finalStopId >= 1}, waitingTotal={waitingTotal}, onBoard={onBoard}, " +
                $"policy={GetPolicySummary()}, demand=[{BuildDecisionDemandSummary()}]");
        }

        private string GetPolicySummary()
        {
            var behaviorParameters = GetComponent<BehaviorParameters>();
            if (behaviorParameters == null)
            {
                return "BehaviorParametersMissing";
            }

            string modelName = behaviorParameters.Model != null ? behaviorParameters.Model.name : "-";
            return $"{nextStopPolicy}/{behaviorParameters.BehaviorType}/model={modelName}";
        }

        private string BuildDecisionDemandSummary()
        {
            if (decisionStops == null || decisionPassengerManager == null)
            {
                return "-";
            }

            int stopCount = Mathf.Min(decisionStops.Count, maxStops);
            var summary = new StringBuilder();

            for (int i = 0; i < stopCount; i++)
            {
                var stop = decisionStops[i];
                if (stop == null)
                {
                    continue;
                }

                if (summary.Length > 0)
                {
                    summary.Append("; ");
                }

                summary.Append("S")
                    .Append(stop.StopId)
                    .Append("(w=").Append(decisionPassengerManager.GetWaitingCountAtStop(stop.StopId, decisionEpisodeTime))
                    .Append(",drop=").Append(decisionPassengerManager.GetOnBoardDestinationCount(stop.StopId))
                    .Append(",future=").Append(decisionPassengerManager.GetScheduledCountAtStop(stop.StopId, decisionEpisodeTime))
                    .Append(")");
            }

            return summary.Length > 0 ? summary.ToString() : "-";
        }

        private static string FormatStopId(int stopId)
        {
            return stopId >= 1 ? stopId.ToString() : "-";
        }

        private int FindActionIndexForStop(int stopId)
        {
            int stopCount = decisionStops != null ? Mathf.Min(decisionStops.Count, maxStops) : 0;
            int index = FindStopIndex(decisionStops, stopId);
            if (index >= 0 && index < stopCount)
            {
                return index;
            }

            for (int i = 0; i < stopCount; i++)
            {
                if (decisionStops[i] == null)
                {
                    continue;
                }

                if (!skipCurrentStop || stopCount <= 1 || decisionStops[i].StopId != decisionCurrentStopId)
                {
                    return i;
                }
            }

            return 0;
        }

        private float GetUnboardedPassengerPenalty(float currentEpisodeTime)
        {
            float penalty = 0f;
            var requests = decisionPassengerManager.Requests;

            for (int i = 0; i < requests.Count; i++)
            {
                var request = requests[i];
                if (request == null)
                {
                    continue;
                }

                if (request.Status == DRTPassengerStatus.Waiting)
                {
                    penalty += SecondsToMinutes(request.GetWaitTime(currentEpisodeTime));
                    continue;
                }

                if (request.Status == DRTPassengerStatus.Scheduled &&
                    request.RequestTimeSeconds <= currentEpisodeTime)
                {
                    penalty += SecondsToMinutes(currentEpisodeTime - request.RequestTimeSeconds);
                }
            }

            return penalty * unboardedPassengerPenaltyWeight;
        }

        private float GetBoardingReward(int stopId, float currentEpisodeTime, float networkAverageReward)
        {
            float reward = 0f;
            var requests = decisionPassengerManager.Requests;

            for (int i = 0; i < requests.Count; i++)
            {
                var request = requests[i];
                if (request == null ||
                    request.ActualPickupStopId != stopId ||
                    !IsSameEpisodeTime(request.PickupTimeSeconds, currentEpisodeTime))
                {
                    continue;
                }

                float multiplier = request.GetWaitTime(currentEpisodeTime) <= acceptableWaitSeconds
                    ? acceptableWaitRewardMultiplier
                    : 1f;
                reward += boardingRewardWeight * networkAverageReward * multiplier;
            }

            return reward;
        }

        private float GetDropoffReward(int stopId, float currentEpisodeTime, float networkAverageReward)
        {
            float reward = 0f;
            var requests = decisionPassengerManager.Requests;

            for (int i = 0; i < requests.Count; i++)
            {
                var request = requests[i];
                if (request == null ||
                    request.ActualDropoffStopId != stopId ||
                    !IsSameEpisodeTime(request.DropoffTimeSeconds, currentEpisodeTime))
                {
                    continue;
                }

                reward += dropoffRewardWeight * networkAverageReward;
            }

            return reward;
        }

        private float GetNoInteractionPenalty(DRTStopProcessResult result, float networkAverageReward)
        {
            if (noInteractionPenaltyWeight <= 0f ||
                result.BoardedCount > 0 ||
                result.DroppedOffCount > 0)
            {
                return 0f;
            }

            return noInteractionPenaltyWeight * networkAverageReward;
        }

        private float GetNetworkAverageAccessReward()
        {
            int stopCount = decisionStops != null ? Mathf.Min(decisionStops.Count, maxStops) : 0;
            if (stopCount < 2)
            {
                return minimumNetworkAverageReward;
            }

            if (busController != null &&
                busController.TryGetAverageStopTravelTimeMinutes(decisionStops, out float matrixAverageMinutes))
            {
                return Mathf.Max(minimumNetworkAverageReward, matrixAverageMinutes);
            }

            float totalMinutes = 0f;
            int pairCount = 0;

            for (int originIndex = 0; originIndex < stopCount; originIndex++)
            {
                DRTStop origin = decisionStops[originIndex];
                if (origin == null)
                {
                    continue;
                }

                for (int destinationIndex = 0; destinationIndex < stopCount; destinationIndex++)
                {
                    DRTStop destination = decisionStops[destinationIndex];
                    if (destination == null || destinationIndex == originIndex)
                    {
                        continue;
                    }

                    float distance = Vector3.Distance(origin.Position, destination.Position);
                    totalMinutes += distance / networkDistanceUnitsPerMinute;
                    pairCount++;
                }
            }

            if (pairCount == 0)
            {
                return minimumNetworkAverageReward;
            }

            return Mathf.Max(minimumNetworkAverageReward, totalMinutes / pairCount);
        }

        private static bool IsSameEpisodeTime(float recordedTime, float currentEpisodeTime)
        {
            return recordedTime >= 0f && Mathf.Abs(recordedTime - currentEpisodeTime) <= 0.05f;
        }

        private static float SecondsToMinutes(float seconds)
        {
            return Mathf.Max(0f, seconds) / 60f;
        }

        private float GetCandidateTravelFeature(int fromStopId, int toStopId, DRTStop origin, DRTStop destination)
        {
            if (busController != null &&
                busController.TryGetTravelTimeSeconds(fromStopId, toStopId, out float matrixSeconds))
            {
                return Mathf.Clamp01(matrixSeconds / Mathf.Max(1f, maxTravelSecondsForObservation));
            }

            if (origin == null || destination == null)
            {
                return 1f;
            }

            float distance = Vector3.Distance(origin.Position, destination.Position);
            return Mathf.Clamp01(distance / Mathf.Max(1f, maxDistanceForObservation));
        }

        private float GetCandidateTravelSeconds(int fromStopId, int toStopId, DRTStop origin, DRTStop destination)
        {
            if (fromStopId == toStopId)
            {
                return 0f;
            }

            if (busController != null &&
                busController.TryGetTravelTimeSeconds(fromStopId, toStopId, out float matrixSeconds))
            {
                return Mathf.Max(0f, matrixSeconds);
            }

            if (origin == null || destination == null)
            {
                return float.MaxValue;
            }

            float distance = Vector3.Distance(origin.Position, destination.Position);
            return 60f * distance / Mathf.Max(0.001f, networkDistanceUnitsPerMinute);
        }

        private static float GetOldestWaitingSecondsAtStop(
            DRTPassengerManager passengerManager,
            int stopId,
            float currentEpisodeTime)
        {
            if (passengerManager == null || stopId < 1)
            {
                return 0f;
            }

            float oldestWaitSeconds = 0f;
            var requests = passengerManager.Requests;
            for (int i = 0; i < requests.Count; i++)
            {
                var request = requests[i];
                if (request == null ||
                    request.OriginStopId != stopId ||
                    request.Status != DRTPassengerStatus.Waiting)
                {
                    continue;
                }

                oldestWaitSeconds = Mathf.Max(
                    oldestWaitSeconds,
                    request.GetWaitTime(currentEpisodeTime));
            }

            return oldestWaitSeconds;
        }

        private void GetStopPassengerTimeFeatures(int stopId, float currentEpisodeTime, out float maxWaitSeconds, out float maxRideSeconds)
        {
            maxWaitSeconds = 0f;
            maxRideSeconds = 0f;

            if (decisionPassengerManager == null || stopId < 1)
            {
                return;
            }

            var requests = decisionPassengerManager.Requests;
            for (int i = 0; i < requests.Count; i++)
            {
                var request = requests[i];
                if (request == null)
                {
                    continue;
                }

                if (request.OriginStopId == stopId && request.Status == DRTPassengerStatus.Waiting)
                {
                    maxWaitSeconds = Mathf.Max(maxWaitSeconds, request.GetWaitTime(currentEpisodeTime));
                }

                if (request.DestinationStopId == stopId && request.Status == DRTPassengerStatus.OnBoard)
                {
                    maxRideSeconds = Mathf.Max(maxRideSeconds, request.GetRideTime(currentEpisodeTime));
                }
            }
        }

        private float NormalizeStopIndex(int stopId)
        {
            if (stopId <= 0)
            {
                return 0f;
            }

            return Mathf.Clamp01((float)(stopId - 1) / Mathf.Max(1, maxStops - 1));
        }

        private void ConfigureBehaviorParameters()
        {
            var behaviorParameters = GetComponent<BehaviorParameters>();
            if (behaviorParameters == null)
            {
                return;
            }

            behaviorParameters.hideFlags = HideFlags.HideInInspector;

            if (BypassMlAgentsDuringDriveTraining)
            {
                behaviorParameters.enabled = false;
                behaviorParameters.Model = null;
                behaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
                if (!loggedDriveTrainingBypass)
                {
                    Debug.Log("[DRTNextStopSelector] ML-Agents next-stop policy disabled while training DRTDrivePPO.");
                    loggedDriveTrainingBypass = true;
                }
            }
            else
            {
                behaviorParameters.enabled = true;
                loggedDriveTrainingBypass = false;
                switch (nextStopPolicy)
                {
                    case DRTNextStopPolicy.ONNXInference:
                        behaviorParameters.Model = onnxInferenceModel;
                        behaviorParameters.InferenceDevice = onnxInferenceDevice;
                        behaviorParameters.BehaviorType = BehaviorType.InferenceOnly;
                        break;
                    case DRTNextStopPolicy.VanillaSequential:
                        behaviorParameters.Model = null;
                        behaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
                        break;
                    case DRTNextStopPolicy.GreedyNearestFeasible:
                        behaviorParameters.Model = null;
                        behaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
                        break;
                    case DRTNextStopPolicy.Fifo:
                        behaviorParameters.Model = null;
                        behaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
                        break;
                    case DRTNextStopPolicy.AllStationRunner:
                        behaviorParameters.Model = null;
                        behaviorParameters.BehaviorType = BehaviorType.HeuristicOnly;
                        break;
                    default:
                        behaviorParameters.Model = null;
                        behaviorParameters.BehaviorType = BehaviorType.Default;
                        break;
                }
            }

            behaviorParameters.BehaviorName = BehaviorName;
            behaviorParameters.BrainParameters.VectorObservationSize = ObservationSize;
            behaviorParameters.BrainParameters.NumStackedVectorObservations = 1;
            behaviorParameters.BrainParameters.ActionSpec = new ActionSpec(0, new[] { maxStops });
        }

        private static void RecordStat(
            string name,
            float value,
            StatAggregationMethod aggregationMethod = StatAggregationMethod.Average)
        {
            Academy.Instance.StatsRecorder.Add(name, value, aggregationMethod);
        }

        private bool ShouldSuppressUnityLogs()
        {
            return busController != null && busController.SuppressUnityLogsDuringMatrixTeleport;
        }

        private void ResolveBusController()
        {
            if (busController != null)
            {
                return;
            }

            busController = GetComponent<DRTBusController>();
            if (busController == null)
            {
                busController = FindObjectOfType<DRTBusController>();
            }
        }

        private static DRTStop FindStop(IReadOnlyList<DRTStop> stops, int stopId)
        {
            if (stops == null)
            {
                return null;
            }

            for (int i = 0; i < stops.Count; i++)
            {
                if (stops[i] != null && stops[i].StopId == stopId)
                {
                    return stops[i];
                }
            }

            return null;
        }

        private static int FindStopIndex(IReadOnlyList<DRTStop> stops, int stopId)
        {
            if (stops == null)
            {
                return -1;
            }

            for (int i = 0; i < stops.Count; i++)
            {
                if (stops[i] != null && stops[i].StopId == stopId)
                {
                    return i;
                }
            }

            return -1;
        }

        private void ResetAllStationRunnerState()
        {
            allStationRoute.Clear();
            allStationRouteSignature = string.Empty;
            allStationRouteCursor = 0;
            allStationRunComplete = false;
        }

        private void EnsureAllStationRoute(int currentStopId, IReadOnlyList<DRTStop> stops)
        {
            string signature = BuildAllStationRouteSignature(stops);
            if (allStationRoute.Count > 0 && allStationRouteSignature == signature)
            {
                return;
            }

            allStationRoute.Clear();
            allStationRouteSignature = signature;
            allStationRouteCursor = 0;
            allStationRunComplete = false;

            List<int> stopIds = GetSortedStopIds(stops);
            if (stopIds.Count < 2)
            {
                allStationRunComplete = true;
                return;
            }

            int preferredStartStopId = allStationRouteStartStopId > 0 ? allStationRouteStartStopId : currentStopId;
            int startStopId = stopIds.Contains(preferredStartStopId) ? preferredStartStopId : stopIds[0];
            BuildEulerianAllStationRoute(stopIds, startStopId, allStationRoute);
            TrimCompletedAllStationRoutePrefix(allStationRoute);
            allStationRunComplete = allStationRoute.Count <= 1;
        }

        private void AlignAllStationCursor(int currentStopId)
        {
            if (allStationRoute.Count == 0 ||
                allStationRunComplete ||
                allStationRouteCursor < allStationRoute.Count &&
                allStationRoute[allStationRouteCursor] == currentStopId)
            {
                return;
            }

            for (int index = Mathf.Max(0, allStationRouteCursor); index < allStationRoute.Count; index++)
            {
                if (allStationRoute[index] == currentStopId)
                {
                    allStationRouteCursor = index;
                    allStationRunComplete = allStationRouteCursor >= allStationRoute.Count - 1;
                    return;
                }
            }
        }

        private static void BuildEulerianAllStationRoute(List<int> stopIds, int startStopId, List<int> output)
        {
            var adjacency = new Dictionary<int, List<int>>();
            for (int originIndex = 0; originIndex < stopIds.Count; originIndex++)
            {
                int originStopId = stopIds[originIndex];
                var destinations = new List<int>();
                for (int destinationIndex = 0; destinationIndex < stopIds.Count; destinationIndex++)
                {
                    int destinationStopId = stopIds[destinationIndex];
                    if (destinationStopId != originStopId)
                    {
                        destinations.Add(destinationStopId);
                    }
                }

                adjacency[originStopId] = destinations;
            }

            var stack = new Stack<int>();
            var reversedPath = new List<int>();
            stack.Push(startStopId);

            while (stack.Count > 0)
            {
                int currentStopId = stack.Peek();
                List<int> destinations = adjacency[currentStopId];
                if (destinations.Count > 0)
                {
                    int nextStopId = destinations[0];
                    destinations.RemoveAt(0);
                    stack.Push(nextStopId);
                    continue;
                }

                reversedPath.Add(stack.Pop());
            }

            for (int index = reversedPath.Count - 1; index >= 0; index--)
            {
                output.Add(reversedPath[index]);
            }
        }

        private List<int> BuildAllStationRouteForResume(IReadOnlyList<DRTStop> stops, int fallbackStartStopId)
        {
            var route = new List<int>();
            List<int> stopIds = GetSortedStopIds(stops);
            if (stopIds.Count < 2)
            {
                return route;
            }

            int preferredStartStopId = allStationRouteStartStopId > 0 ? allStationRouteStartStopId : fallbackStartStopId;
            int startStopId = stopIds.Contains(preferredStartStopId) ? preferredStartStopId : stopIds[0];
            BuildEulerianAllStationRoute(stopIds, startStopId, route);
            return route;
        }

        private void TrimCompletedAllStationRoutePrefix(List<int> route)
        {
            int completedPrefix = CountCompletedRoutePrefix(route);
            if (completedPrefix > 0)
            {
                route.RemoveRange(0, completedPrefix);
            }
        }

        private int CountCompletedRoutePrefix(IReadOnlyList<int> route)
        {
            if (route == null || route.Count < 2 || allStationCompletedPairKeys.Count == 0)
            {
                return 0;
            }

            int completedPrefix = 0;
            while (completedPrefix < route.Count - 1 &&
                   allStationCompletedPairKeys.Contains(BuildStopPairKey(route[completedPrefix], route[completedPrefix + 1])))
            {
                completedPrefix++;
            }

            return completedPrefix;
        }

        public static string BuildStopPairKey(int fromStopId, int toStopId)
        {
            return $"{fromStopId}->{toStopId}";
        }

        private string BuildAllStationRouteSignature(IReadOnlyList<DRTStop> stops)
        {
            List<int> stopIds = GetSortedStopIds(stops);
            var builder = new StringBuilder();
            for (int i = 0; i < stopIds.Count; i++)
            {
                if (i > 0)
                {
                    builder.Append('|');
                }

                builder.Append(stopIds[i]);
            }

            builder.Append("#start=").Append(allStationRouteStartStopId);
            if (allStationCompletedPairKeys.Count > 0)
            {
                builder.Append("#done=");
                var completedPairKeys = new List<string>(allStationCompletedPairKeys);
                completedPairKeys.Sort();
                for (int i = 0; i < completedPairKeys.Count; i++)
                {
                    if (i > 0)
                    {
                        builder.Append(',');
                    }

                    builder.Append(completedPairKeys[i]);
                }
            }

            return builder.ToString();
        }

        private static List<int> GetSortedStopIds(IReadOnlyList<DRTStop> stops)
        {
            var stopIds = new List<int>();
            if (stops == null)
            {
                return stopIds;
            }

            for (int i = 0; i < stops.Count; i++)
            {
                if (stops[i] != null && !stopIds.Contains(stops[i].StopId))
                {
                    stopIds.Add(stops[i].StopId);
                }
            }

            stopIds.Sort();
            return stopIds;
        }

        private static int GetNextSequentialStopId(int currentStopId, IReadOnlyList<DRTStop> stops)
        {
            if (stops == null || stops.Count == 0)
            {
                return -1;
            }

            int currentIndex = FindStopIndex(stops, currentStopId);
            int nextIndex = currentIndex >= 0 ? (currentIndex + 1) % stops.Count : 0;
            return stops[nextIndex] != null ? stops[nextIndex].StopId : -1;
        }

        private void OnValidate()
        {
            maxStops = Mathf.Max(2, maxStops);
            episodeLengthSeconds = Mathf.Max(1f, episodeLengthSeconds);
            maxDistanceForObservation = Mathf.Max(1f, maxDistanceForObservation);
            maxTravelSecondsForObservation = Mathf.Max(1f, maxTravelSecondsForObservation);
            maxWaitSecondsForObservation = Mathf.Max(1f, maxWaitSecondsForObservation);
            maxDecisionWaitSeconds = Mathf.Max(0.05f, maxDecisionWaitSeconds);
            unboardedPassengerPenaltyWeight = Mathf.Max(0f, unboardedPassengerPenaltyWeight);
            boardingRewardWeight = Mathf.Max(0f, boardingRewardWeight);
            dropoffRewardWeight = Mathf.Max(0f, dropoffRewardWeight);
            noInteractionPenaltyWeight = Mathf.Max(0f, noInteractionPenaltyWeight);
            acceptableWaitSeconds = Mathf.Max(1f, acceptableWaitSeconds);
            acceptableWaitRewardMultiplier = Mathf.Max(1f, acceptableWaitRewardMultiplier);
            networkDistanceUnitsPerMinute = Mathf.Max(1f, networkDistanceUnitsPerMinute);
            minimumNetworkAverageReward = Mathf.Max(0.001f, minimumNetworkAverageReward);
            ConfigureBehaviorParameters();
        }
    }
}
