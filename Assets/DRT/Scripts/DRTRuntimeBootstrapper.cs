using System.Collections.Generic;
using UnityEngine;
using Gley.TrafficSystem;
using Gley.UrbanSystem;

namespace DRT
{
    public static class DRTRuntimeBootstrapper
    {
        private const string SystemObjectName = "DRTSystem";
        private const string BusStopsObjectName = "BusStops";
        private const int GleyChangeLanePenalty = 10;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Setup()
        {
            Transform busStopsRoot = FindBusStopsRoot();
            if (busStopsRoot == null)
            {
                Debug.LogWarning("[DRT] BusStops object not found. DRT auto setup skipped.");
                return;
            }

            DisableGleyTrafficExample();
            EnsurePathFindingEnabled();

            var existingBusControllers = Object.FindObjectsOfType<DRTBusController>();
            DRTBusController configuredBusController = SelectConfiguredBusController(existingBusControllers);

            GameObject systemObject = GameObject.Find(SystemObjectName);
            if (configuredBusController != null)
            {
                systemObject = configuredBusController.gameObject;
            }
            else if (systemObject == null)
            {
                systemObject = new GameObject(SystemObjectName);
            }

            var passengerManager = GetOrAdd<DRTPassengerManager>(systemObject);
            var demandGenerator = GetOrAdd<DRTDemandGenerator>(systemObject);
            var nextStopSelector = GetOrAdd<DRTNextStopSelector>(systemObject);
            var busController = GetOrAdd<DRTBusController>(systemObject);
            var dotVisualizer = GetOrAdd<DRTPassengerDotVisualizer>(systemObject);
            var debugGui = GetOrAdd<DRTDebugGUI>(systemObject);

            busController.Configure(busStopsRoot, passengerManager, demandGenerator, nextStopSelector, 0, 1);
            dotVisualizer.Configure(passengerManager, busController);
            debugGui.Configure(passengerManager, busController);

            int loadedStopCount = busController.Stops != null ? busController.Stops.Count : busStopsRoot.childCount;
            Debug.Log(
                $"[DRT] Auto setup complete. Stops={loadedStopCount}, " +
                $"systemObject={systemObject.name}, mode={busController.TravelExecutionModeName}, " +
                $"physicalDriver={busController.PhysicalDriveModeName}, ppoPolicy={busController.PPODrivePolicyName}, " +
                $"policy={nextStopSelector.NextStopPolicyName}");
        }

        private static DRTBusController SelectConfiguredBusController(DRTBusController[] busControllers)
        {
            if (busControllers == null || busControllers.Length == 0)
            {
                return null;
            }

            for (int i = 0; i < busControllers.Length; i++)
            {
                var busController = busControllers[i];
                if (busController != null && busController.gameObject.name == SystemObjectName)
                {
                    return busController;
                }
            }

            if (busControllers.Length > 1)
            {
                Debug.LogWarning(
                    $"[DRT] Multiple DRTBusController components found. " +
                    $"Using '{busControllers[0].gameObject.name}'. Remove duplicates to avoid unexpected run mode.");
            }

            return busControllers[0];
        }

        private static void EnsurePathFindingEnabled()
        {
            var trafficModules = Object.FindObjectOfType<TrafficModules>();
            var trafficWaypointsData = Object.FindObjectOfType<TrafficWaypointsData>();

            if (trafficModules == null || trafficWaypointsData == null)
            {
                Debug.LogWarning("[DRT] TrafficModules or TrafficWaypointsData not found. Path Finding setup skipped.");
                return;
            }

            trafficModules.SetModules(true);

            var pathFindingData = trafficWaypointsData.GetComponent<PathFindingData>();
            if (pathFindingData == null)
            {
                pathFindingData = trafficWaypointsData.gameObject.AddComponent<PathFindingData>();
            }

            var trafficWaypoints = trafficWaypointsData.AllTrafficWaypoints;
            if (trafficWaypoints == null || trafficWaypoints.Length == 0)
            {
                Debug.LogWarning("[DRT] Traffic waypoints missing. Path Finding data cannot be generated.");
                return;
            }

            var pathFindingWaypoints = new PathFindingWaypoint[trafficWaypoints.Length];
            int laneChangeEdges = 0;
            for (int i = 0; i < trafficWaypoints.Length; i++)
            {
                var waypoint = trafficWaypoints[i];
                int[] allowedAgents = new int[waypoint.AllowedVehicles.Length];
                for (int agentIndex = 0; agentIndex < waypoint.AllowedVehicles.Length; agentIndex++)
                {
                    allowedAgents[agentIndex] = (int)waypoint.AllowedVehicles[agentIndex];
                }

                int[] movementPenalties;
                int[] neighbours = BuildGleyPathFindingNeighbours(waypoint, out movementPenalties, out int addedLaneChangeEdges);
                laneChangeEdges += addedLaneChangeEdges;

                pathFindingWaypoints[i] = new PathFindingWaypoint(
                    waypoint.ListIndex,
                    waypoint.Position,
                    0,
                    0,
                    -1,
                    neighbours,
                    movementPenalties,
                    allowedAgents);
            }

            pathFindingData.SetPathFindingWaypoints(pathFindingWaypoints);
            Debug.Log($"[DRT] Runtime Path Finding enabled. Waypoints={pathFindingWaypoints.Length}, laneChangeEdges={laneChangeEdges}");
        }

        private static int[] BuildGleyPathFindingNeighbours(
            TrafficWaypoint waypoint,
            out int[] movementPenalties,
            out int laneChangeEdges)
        {
            var neighbours = new List<int>();
            var penalties = new List<int>();
            laneChangeEdges = 0;

            AddPathFindingNeighbours(waypoint.Neighbors, 0, neighbours, penalties, ref laneChangeEdges, false);
            AddPathFindingNeighbours(waypoint.OtherLanes, GleyChangeLanePenalty, neighbours, penalties, ref laneChangeEdges, true);

            movementPenalties = penalties.ToArray();
            return neighbours.ToArray();
        }

        private static void AddPathFindingNeighbours(
            int[] source,
            int movementPenalty,
            List<int> neighbours,
            List<int> penalties,
            ref int laneChangeEdges,
            bool countAsLaneChange)
        {
            if (source == null)
            {
                return;
            }

            for (int i = 0; i < source.Length; i++)
            {
                int neighbour = source[i];
                if (neighbour < 0 || neighbours.Contains(neighbour))
                {
                    continue;
                }

                neighbours.Add(neighbour);
                penalties.Add(movementPenalty);
                if (countAsLaneChange)
                {
                    laneChangeEdges++;
                }
            }
        }

        private static Transform FindBusStopsRoot()
        {
            GameObject exactMatch = GameObject.Find(BusStopsObjectName);
            if (exactMatch != null)
            {
                return exactMatch.transform;
            }

            var allTransforms = Object.FindObjectsOfType<Transform>();
            for (int i = 0; i < allTransforms.Length; i++)
            {
                if (allTransforms[i].name.ToLowerInvariant().Contains("busstop"))
                {
                    return allTransforms[i];
                }
            }

            return null;
        }

        private static void DisableGleyTrafficExample()
        {
            var behaviours = Object.FindObjectsOfType<MonoBehaviour>();
            for (int i = 0; i < behaviours.Length; i++)
            {
                var behaviour = behaviours[i];
                if (behaviour == null)
                {
                    continue;
                }

                var typeName = behaviour.GetType().FullName;
                if (typeName == "Gley.TrafficSystem.Internal.TrafficExample")
                {
                    if (behaviour.enabled)
                    {
                        behaviour.enabled = false;
                        Debug.Log("[DRT] Disabled Gley TrafficExample to avoid duplicate vehicle control.");
                    }
                }
            }
        }

        private static T GetOrAdd<T>(GameObject gameObject) where T : Component
        {
            var component = gameObject.GetComponent<T>();
            if (component == null)
            {
                component = gameObject.AddComponent<T>();
            }

            return component;
        }
    }
}
