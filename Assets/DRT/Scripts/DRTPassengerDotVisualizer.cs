using System.Collections.Generic;
using UnityEngine;

namespace DRT
{
    [AddComponentMenu("DRT/DRT Passenger Dot Visualizer")]
    public class DRTPassengerDotVisualizer : MonoBehaviour
    {
        [HideInInspector, SerializeField] private DRTPassengerManager passengerManager;
        [HideInInspector, SerializeField] private DRTBusController busController;

        [Header("Dots")]
        [SerializeField, InspectorName("Show Stops")] private bool showStopDots = true;
        [SerializeField, InspectorName("Show Bus")] private bool showBusDots = true;
        [SerializeField, InspectorName("Dot Radius")] private float dotRadius = 6f;
        [SerializeField, InspectorName("Dot Spacing")] private float dotSpacing = 18f;
        [SerializeField, InspectorName("Bus Dot Spacing")] private float busDotSpacing = 20f;
        [SerializeField, InspectorName("Dots Per Row")] private int dotsPerRow = 3;
        [SerializeField, InspectorName("Max Dots")] private int maxDotsPerGroup = 80;
        [SerializeField, InspectorName("Stop Height")] private float stopVerticalOffset = 8f;
        [SerializeField, InspectorName("Bus Height")] private float busVerticalOffset = 8f;
        [SerializeField, InspectorName("Stop Forward Offset")] private float stopForwardOffset = 0.5f;
        [SerializeField, InspectorName("Stop Side Offset")] private float stopSideOffset = 12f;
        [SerializeField, InspectorName("Bus Forward Offset")] private float busForwardOffset = 0f;
        [SerializeField, InspectorName("Bus Side Offset")] private float busSideOffset = 0f;

        public void Configure(DRTPassengerManager newPassengerManager, DRTBusController newBusController)
        {
            passengerManager = newPassengerManager;
            busController = newBusController;
        }

        private void OnDrawGizmos()
        {
            if (passengerManager == null || busController == null)
            {
                passengerManager = FindObjectOfType<DRTPassengerManager>();
                busController = FindObjectOfType<DRTBusController>();
                if (passengerManager == null || busController == null)
                {
                    return;
                }
            }

            float currentTime = busController.EpisodeTimeSeconds;

            if (showStopDots)
            {
                DrawStopDots(currentTime);
            }

            if (showBusDots)
            {
                DrawBusDots();
            }
        }

        private void DrawStopDots(float currentTime)
        {
            Vector3 right = GetScreenAlignedRight();
            Vector3 forward = Vector3.Cross(Vector3.up, right).normalized;

            foreach (var stop in busController.Stops)
            {
                if (stop == null)
                {
                    continue;
                }

                int waitingCount = passengerManager.GetWaitingCountAtStop(stop.StopId, currentTime);
                if (waitingCount <= 0)
                {
                    continue;
                }

                Vector3 labelPosition;
                if (TryGetStopLabelWorldPosition(stop, out labelPosition))
                {
                    Vector3 center = labelPosition + right * (stopSideOffset + dotRadius * 1.2f);
                    DrawDotGrid(center, right, forward, waitingCount, dotSpacing, visibleCount: waitingCount, forceSingleRow: true);
                }
                else
                {
                    Vector3 center = stop.Position + Vector3.up * stopVerticalOffset + right * stopSideOffset + forward * stopForwardOffset;
                    DrawDotGrid(center, right, forward, waitingCount, dotSpacing, visibleCount: waitingCount, forceSingleRow: true);
                }
            }
        }

        private Vector3 GetScreenAlignedRight()
        {
            Camera cam = Camera.current;
            if (cam == null)
            {
                return Vector3.right;
            }

            Vector3 screenRight = cam.transform.right;
            Vector3 horizontalRight = Vector3.ProjectOnPlane(screenRight, Vector3.up);
            if (horizontalRight.sqrMagnitude < 0.001f)
            {
                return Vector3.right;
            }

            return horizontalRight.normalized;
        }

        private bool TryGetStopLabelWorldPosition(DRTStop stop, out Vector3 labelPosition)
        {
            var textMesh = stop.GetComponentInChildren<TextMesh>();
            if (textMesh != null)
            {
                labelPosition = textMesh.transform.position;
                return true;
            }

            var allComponents = stop.GetComponentsInChildren<Component>(true);
            foreach (var component in allComponents)
            {
                if (component == null)
                {
                    continue;
                }

                string componentName = component.GetType().Name;
                if (componentName.Contains("TextMeshPro"))
                {
                    labelPosition = component.transform.position;
                    return true;
                }
            }

            labelPosition = Vector3.zero;
            return false;
        }

        private void DrawBusDots()
        {
            Transform vehicleTransform = busController.ControlledVehicleTransform;
            if (vehicleTransform == null)
            {
                return;
            }

            int onBoardCount = passengerManager.GetOnBoardCount();
            if (onBoardCount <= 0)
            {
                return;
            }

            Vector3 bodyPosition = busController.ControlledVehicleBodyPosition;
            Vector3 forward = vehicleTransform.forward.sqrMagnitude > 0.001f ? vehicleTransform.forward.normalized : Vector3.forward;
            Vector3 side = Vector3.Cross(Vector3.up, forward).normalized;
            Vector3 center = bodyPosition + Vector3.up * busVerticalOffset;
            int busColumns = onBoardCount == 4 ? 2 : dotsPerRow;
            DrawDotGrid(center, side, forward, onBoardCount, busDotSpacing, visibleCount: onBoardCount, forcedColumns: busColumns);
        }

        private void DrawDotGrid(Vector3 center, Vector3 right, Vector3 forward, int dotCount, float spacing, int visibleCount = -1, int forcedColumns = -1, bool forceSingleRow = false)
        {
            Gizmos.color = Color.green;
            int count = visibleCount < 0 ? Mathf.Clamp(dotCount, 0, maxDotsPerGroup) : Mathf.Clamp(visibleCount, 0, maxDotsPerGroup);
            int columns = forcedColumns > 0 ? forcedColumns : (forceSingleRow ? Mathf.Max(1, count) : Mathf.Max(1, dotsPerRow));

            for (int i = 0; i < count; i++)
            {
                int row = forceSingleRow ? 0 : i / columns;
                int column = i % columns;
                int rowColumns = forceSingleRow ? count : Mathf.Min(columns, count - row * columns);
                float xOffset = (column - (rowColumns - 1) * 0.5f) * spacing;
                float zOffset = row * spacing;
                Vector3 dotPos = center + right.normalized * xOffset + forward.normalized * zOffset;
                Gizmos.DrawSphere(dotPos, dotRadius);
            }
        }

        private void OnValidate()
        {
            dotRadius = Mathf.Max(0.05f, dotRadius);
            dotSpacing = Mathf.Max(0.05f, dotSpacing);
            busDotSpacing = Mathf.Max(0.05f, busDotSpacing);
            dotsPerRow = Mathf.Max(1, dotsPerRow);
            maxDotsPerGroup = Mathf.Max(1, maxDotsPerGroup);
            stopVerticalOffset = Mathf.Max(0f, stopVerticalOffset);
            busVerticalOffset = Mathf.Max(0f, busVerticalOffset);
            stopForwardOffset = Mathf.Max(0f, stopForwardOffset);
            stopSideOffset = Mathf.Max(0f, stopSideOffset);
            busForwardOffset = Mathf.Max(0f, busForwardOffset);
            busSideOffset = Mathf.Max(0f, busSideOffset);
        }
    }
}
