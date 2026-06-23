using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace DRT
{
    public class DRTDebugGUI : MonoBehaviour
    {
        [HideInInspector, SerializeField] private DRTPassengerManager passengerManager;
        [HideInInspector, SerializeField] private DRTBusController busController;
        [SerializeField, InspectorName("Passenger Table")] private bool showPassengerTable = true;
        [SerializeField, InspectorName("Stop Overview")] private bool showStopOverview = true;
        [SerializeField, InspectorName("Max Rows")] private int maxPassengerRows = 40;

        private Vector2 passengerScroll;
        private Vector2 overviewScroll;
        private Rect passengerWindow = new Rect(12, 12, 850, 360);
        private Rect overviewWindow = new Rect(12, 384, 520, 420);
        private Rect controlWindow = new Rect(12, 812, 260, 130);
        private int resizingWindowId = -1;
        private int pendingResizeWindowId = -1;
        private Vector2 pendingResizeWindowSize;
        private GUIStyle headerStyle;
        private GUIStyle smallStyle;
        private GUIStyle cellStyle;
        private GUIStyle resizeStyle;

        private const int PassengerWindowId = 5201;
        private const int OverviewWindowId = 5202;
        private const int ControlWindowId = 5203;
        private const float ResizeHandleSize = 18f;

        public void Configure(DRTPassengerManager newPassengerManager, DRTBusController newBusController)
        {
            passengerManager = newPassengerManager;
            busController = newBusController;
        }

        private void Awake()
        {
            if (passengerManager == null)
            {
                passengerManager = FindObjectOfType<DRTPassengerManager>();
            }

            if (busController == null)
            {
                busController = FindObjectOfType<DRTBusController>();
            }
        }

        private void OnGUI()
        {
            EnsureStyles();
            ClampWindowToScreen(ref passengerWindow);
            ClampWindowToScreen(ref overviewWindow);
            ClampWindowToScreen(ref controlWindow);

            if (showPassengerTable)
            {
                passengerWindow = GUI.Window(PassengerWindowId, passengerWindow, DrawPassengerWindow, "DRT Passenger Requests");
                ApplyPendingResize(ref passengerWindow, PassengerWindowId);
            }

            if (showStopOverview)
            {
                overviewWindow = GUI.Window(OverviewWindowId, overviewWindow, DrawOverviewWindow, "DRT Stop / Bus Status");
                ApplyPendingResize(ref overviewWindow, OverviewWindowId);
            }

            controlWindow = GUI.Window(ControlWindowId, controlWindow, DrawControlWindow, "DRT GUI");
        }

        private void DrawPassengerWindow(int windowId)
        {
            if (passengerManager == null)
            {
                GUILayout.Label("PassengerManager not found.", headerStyle);
                GUI.DragWindow();
                return;
            }

            float currentTime = busController != null ? busController.EpisodeTimeSeconds : Time.time;
            GUILayout.Label($"time,{currentTime:0.0}, total,{passengerManager.Requests.Count}, waiting,{passengerManager.GetWaitingCount(currentTime)}, onboard,{passengerManager.GetOnBoardCount()}, completed,{passengerManager.GetCompletedCount()}", smallStyle);
            GUILayout.Space(4);

            DrawCsvHeader();
            passengerScroll = GUILayout.BeginScrollView(passengerScroll, GUILayout.Height(Mathf.Max(120f, passengerWindow.height - 76f)));

            int rows = 0;
            foreach (var request in passengerManager.Requests.OrderBy(request => request.PassengerId))
            {
                if (rows >= maxPassengerRows)
                {
                    GUILayout.Label($"... {passengerManager.Requests.Count - rows} more rows", smallStyle);
                    break;
                }

                DrawCsvRow(request, currentTime);
                rows++;
            }

            GUILayout.EndScrollView();
            DrawResizeHandle(ref passengerWindow, windowId, 520f, 210f);
            GUI.DragWindow(new Rect(0f, 0f, passengerWindow.width - ResizeHandleSize, 24f));
        }

        private void DrawOverviewWindow(int windowId)
        {
            if (passengerManager == null || busController == null)
            {
                GUILayout.Label("DRT runtime objects not found.", headerStyle);
                GUI.DragWindow();
                return;
            }

            float currentTime = busController.EpisodeTimeSeconds;
            GUILayout.Label($"Target Station: Stop {busController.TargetStopId} ({busController.TargetStopObjectName})", headerStyle);
            GUILayout.Label($"Route: Stop {busController.CurrentStopId} -> Stop {busController.TargetStopId}", smallStyle);
            GUILayout.Label($"Mode {busController.TravelExecutionModeName}/{busController.PhysicalDriveModeName} | PPO {busController.PPODrivePolicyName} | Policy {busController.NextStopPolicyName} | Driver {busController.ControlledDriverName} | Vehicle {busController.ControlledVehicleName} | last served stop {busController.CurrentStopId}", smallStyle);
            GUILayout.Label($"PPO route episode: {busController.PPOTrainingRouteName}", smallStyle);
            GUILayout.Label($"PPO speed limit: {(busController.UsePPOSpeedLimit ? $"{busController.PPOSpeedLimitMetersPerSecond:0.00}m/s" : "off")}", smallStyle);
            GUILayout.Label($"speed={busController.VehicleSpeedMS:0.00}m/s, targetDist={FormatDistance(busController.TargetDistanceMeters)}, arrival<= {busController.ArrivalDistanceMeters:0.00}m, waitingArrival={busController.IsWaitingForArrivalProximity}", smallStyle);
            GUILayout.Label($"blocked={busController.IsVehicleTemporarilyBlocked} {busController.TemporaryBlockReason}", smallStyle);
            GUILayout.Label($"episodeDistance={busController.EpisodeTravelDistanceMeters:0.0}m, assignedPath={busController.AssignedPathDistanceMeters:0.0}m/{busController.AssignedPathPointCount}pts", smallStyle);
            GUILayout.Label($"traffic={(busController.BackgroundTrafficEnabled ? "on" : "off")}, activeOtherVehicles={busController.ActiveBackgroundVehicleCount}", smallStyle);
            GUILayout.Space(6);

            overviewScroll = GUILayout.BeginScrollView(overviewScroll, GUILayout.Height(Mathf.Max(150f, overviewWindow.height - 122f)));
            GUILayout.Label($"In Vehicle ({passengerManager.GetOnBoardCount()}/{passengerManager.BusCapacity})", headerStyle);
            DrawInVehicleList(
                passengerManager.Requests.Where(request => request.Status == DRTPassengerStatus.OnBoard),
                currentTime);

            GUILayout.Space(8);
            GUILayout.Label("Waiting At Stops", headerStyle);

            foreach (var stop in busController.Stops)
            {
                if (stop == null)
                {
                    continue;
                }

                var waitingRequests = passengerManager.Requests
                    .Where(request => request.OriginStopId == stop.StopId && request.Status == DRTPassengerStatus.Waiting)
                    .OrderBy(request => request.RequestTimeSeconds)
                    .ToList();

                string routes = waitingRequests.Count > 0
                    ? string.Join(", ", waitingRequests.Select(FormatRouteOnly))
                    : "-";
                GUILayout.Label($"Stop {stop.StopId}: {routes}", cellStyle);
            }

            GUILayout.EndScrollView();
            DrawResizeHandle(ref overviewWindow, windowId, 360f, 250f);
            GUI.DragWindow(new Rect(0f, 0f, overviewWindow.width - ResizeHandleSize, 24f));
        }

        private void DrawControlWindow(int windowId)
        {
            GUILayout.BeginHorizontal();
            showPassengerTable = GUILayout.Toggle(showPassengerTable, "Passenger Table", GUILayout.Width(130));
            showStopOverview = GUILayout.Toggle(showStopOverview, "Bus Status", GUILayout.Width(100));
            GUILayout.EndHorizontal();

            if (busController != null)
            {
                GUILayout.Space(4);
                GUILayout.Label($"traffic={(busController.BackgroundTrafficEnabled ? "on" : "off")}, activeOtherVehicles={busController.ActiveBackgroundVehicleCount}", smallStyle);
                GUILayout.BeginHorizontal();
                if (GUILayout.Button("Traffic On", GUILayout.Width(100)))
                {
                    busController.EnableBackgroundTraffic();
                }

                if (GUILayout.Button("Traffic Off", GUILayout.Width(100)))
                {
                    busController.DisableBackgroundTraffic();
                }
                GUILayout.EndHorizontal();
            }

            GUI.DragWindow(new Rect(0f, 0f, controlWindow.width, 24f));
        }

        private void DrawCsvHeader()
        {
            GUILayout.BeginHorizontal();
            DrawCell("id", 42, true);
            DrawCell("from", 50, true);
            DrawCell("to", 50, true);
            DrawCell("req", 70, true);
            DrawCell("status", 90, true);
            DrawCell("pickup", 70, true);
            DrawCell("dropoff", 70, true);
            DrawCell("wait", 70, true);
            DrawCell("ride", 70, true);
            DrawCell("elapsed", 80, true);
            GUILayout.EndHorizontal();
        }

        private void DrawCsvRow(DRTPassengerRequest request, float currentTime)
        {
            GUILayout.BeginHorizontal();
            DrawCell(request.PassengerId.ToString(), 42);
            DrawCell(request.OriginStopId.ToString(), 50);
            DrawCell(request.DestinationStopId.ToString(), 50);
            DrawCell(request.RequestTimeSeconds.ToString("0"), 70);
            DrawCell(request.Status.ToString(), 90);
            DrawCell(FormatTime(request.PickupTimeSeconds), 70);
            DrawCell(FormatTime(request.DropoffTimeSeconds), 70);
            DrawCell(request.GetWaitTime(currentTime).ToString("0"), 70);
            DrawCell(request.GetRideTime(currentTime).ToString("0"), 70);
            DrawCell(request.GetElapsedSinceRequest(currentTime).ToString("0"), 80);
            GUILayout.EndHorizontal();
        }

        private void DrawRequestList(IEnumerable<DRTPassengerRequest> requests, float currentTime)
        {
            int count = 0;
            foreach (var request in requests.OrderBy(request => request.RequestTimeSeconds))
            {
                GUILayout.Label($"#{request.PassengerId} Stop {request.OriginStopId} -> {request.DestinationStopId}, wait {request.GetWaitTime(currentTime):0}s, ride {request.GetRideTime(currentTime):0}s", cellStyle);
                count++;
                if (count >= 12)
                {
                    GUILayout.Label("...", smallStyle);
                    break;
                }
            }

            if (count == 0)
            {
                GUILayout.Label("none", smallStyle);
            }
        }

        private void DrawRouteOnlyList(IEnumerable<DRTPassengerRequest> requests)
        {
            var routeTexts = requests
                .OrderBy(request => request.PickupTimeSeconds)
                .Select(FormatRouteOnly)
                .ToList();

            if (routeTexts.Count == 0)
            {
                GUILayout.Label("-", smallStyle);
                return;
            }

            GUILayout.Label(string.Join(", ", routeTexts), cellStyle);
        }

        private void DrawInVehicleList(IEnumerable<DRTPassengerRequest> requests, float currentTime)
        {
            var passengerTexts = requests
                .OrderBy(request => request.PickupTimeSeconds)
                .Select(request => $"#{request.PassengerId} {request.OriginStopId}->{request.DestinationStopId} ride {request.GetRideTime(currentTime):0}s")
                .ToList();

            if (passengerTexts.Count == 0)
            {
                GUILayout.Label("-", smallStyle);
                return;
            }

            GUILayout.Label(string.Join(", ", passengerTexts), cellStyle);
        }

        private string FormatRouteOnly(DRTPassengerRequest request)
        {
            return $"{request.OriginStopId}->{request.DestinationStopId}";
        }

        private void DrawCell(string value, float width, bool header = false)
        {
            GUILayout.Label(value, header ? headerStyle : cellStyle, GUILayout.Width(width));
        }

        private void DrawResizeHandle(ref Rect windowRect, int windowId, float minWidth, float minHeight)
        {
            Rect handleRect = new Rect(windowRect.width - ResizeHandleSize, windowRect.height - ResizeHandleSize, ResizeHandleSize, ResizeHandleSize);
            GUI.Box(handleRect, "R", resizeStyle);

            Event currentEvent = Event.current;
            if (currentEvent.type == EventType.MouseDown && handleRect.Contains(currentEvent.mousePosition))
            {
                resizingWindowId = windowId;
                currentEvent.Use();
            }

            if (resizingWindowId == windowId && currentEvent.type == EventType.MouseDrag)
            {
                windowRect.width = Mathf.Max(minWidth, currentEvent.mousePosition.x + ResizeHandleSize * 0.5f);
                windowRect.height = Mathf.Max(minHeight, currentEvent.mousePosition.y + ResizeHandleSize * 0.5f);
                pendingResizeWindowId = windowId;
                pendingResizeWindowSize = new Vector2(windowRect.width, windowRect.height);
                currentEvent.Use();
            }

            if (resizingWindowId == windowId && currentEvent.type == EventType.MouseUp)
            {
                resizingWindowId = -1;
                currentEvent.Use();
            }
        }

        private void ApplyPendingResize(ref Rect windowRect, int windowId)
        {
            if (pendingResizeWindowId != windowId)
            {
                return;
            }

            windowRect.width = pendingResizeWindowSize.x;
            windowRect.height = pendingResizeWindowSize.y;

            if (Event.current.type == EventType.MouseUp)
            {
                pendingResizeWindowId = -1;
            }
        }

        private void ClampWindowToScreen(ref Rect windowRect)
        {
            float maxX = Mathf.Max(0f, Screen.width - Mathf.Min(windowRect.width, Screen.width));
            float maxY = Mathf.Max(0f, Screen.height - Mathf.Min(windowRect.height, Screen.height));
            windowRect.x = Mathf.Clamp(windowRect.x, -windowRect.width + 80f, maxX);
            windowRect.y = Mathf.Clamp(windowRect.y, 0f, maxY);
        }

        private string FormatTime(float value)
        {
            return value >= 0f ? value.ToString("0") : "-";
        }

        private string FormatDistance(float value)
        {
            return float.IsInfinity(value) ? "-" : $"{value:0.00}m";
        }

        private void EnsureStyles()
        {
            if (headerStyle != null)
            {
                return;
            }

            headerStyle = new GUIStyle(GUI.skin.label)
            {
                fontStyle = FontStyle.Bold,
                fontSize = 13,
                normal = { textColor = Color.white }
            };

            smallStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                normal = { textColor = new Color(0.86f, 0.9f, 0.94f) }
            };

            cellStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                normal = { textColor = Color.white }
            };

            resizeStyle = new GUIStyle(GUI.skin.box)
            {
                fontSize = 8,
                alignment = TextAnchor.MiddleCenter,
                normal = { textColor = new Color(0.86f, 0.9f, 0.94f) }
            };
        }
    }
}
