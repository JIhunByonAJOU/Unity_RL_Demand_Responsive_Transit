using System.Collections.Generic;
using Gley.TrafficSystem;
using Gley.UrbanSystem;
using UnityEngine;

using VehicleTypes = Gley.TrafficSystem.User.VehicleTypes;

namespace DRT
{
    [AddComponentMenu("DRT/DRT Train Vehicle Driver")]
    public class DRTTrainVehicleDriver : MonoBehaviour, IDRTVehicleDriver
    {
        [SerializeField] private Rigidbody vehicleRigidbody;
        [SerializeField] private PlayerCar playerCar;
        [SerializeField] private VehicleTypes vehicleType = VehicleTypes.Car;
        [SerializeField] private float baseCruiseSpeedMetersPerSecond = 10f;
        [SerializeField] private float speedMultiplier = 1f;
        [SerializeField] private float waypointReachDistanceMeters = 0.25f;
        [SerializeField] private float finalReachDistanceMeters = 0.25f;

        private readonly List<Vector3> pathPoints = new List<Vector3>();
        private int targetPointIndex;
        private bool driving;
        private Vector3 kinematicPosition;
        private Quaternion kinematicRotation = Quaternion.identity;
        private float currentSpeedMetersPerSecond;
        private bool hasOriginalRigidbodyState;
        private bool originalIsKinematic;

        public Transform VehicleTransform => transform;
        public string VehicleName => name;
        public float CurrentSpeedMS => currentSpeedMetersPerSecond;
        public int PathPointCount => pathPoints.Count;
        public int RemainingPathPointCount => driving ? Mathf.Max(0, pathPoints.Count - targetPointIndex) : 0;
        public Vector3 BodyPosition => vehicleRigidbody != null ? vehicleRigidbody.position : transform.position;
        public bool IsTemporarilyBlocked => false;
        public string TemporaryBlockReason => string.Empty;
        public bool HasCriticalFault => false;
        public string CriticalFaultReason => string.Empty;

        public void Configure(
            VehicleTypes newVehicleType,
            float newSpeedMultiplier,
            float newWaypointReachDistance,
            float newFinalReachDistance)
        {
            hideFlags = HideFlags.HideInInspector;
            vehicleType = newVehicleType;
            speedMultiplier = Mathf.Max(0.1f, newSpeedMultiplier);
            waypointReachDistanceMeters = Mathf.Max(0.01f, Mathf.Min(0.5f, newWaypointReachDistance));
            finalReachDistanceMeters = Mathf.Max(0.01f, Mathf.Min(0.5f, newFinalReachDistance));
            ResolveReferences();
        }

        public bool SetPath(List<int> waypointIndexes, Vector3 destination)
        {
            ResolveReferences();
            EnsureKinematicControl();

            pathPoints.Clear();
            targetPointIndex = 0;
            currentSpeedMetersPerSecond = 0f;
            kinematicPosition = BodyPosition;
            kinematicRotation = transform.rotation;

            if (waypointIndexes != null)
            {
                for (int i = 0; i < waypointIndexes.Count; i++)
                {
                    var waypoint = API.GetWaypointFromIndex(waypointIndexes[i]);
                    if (waypoint != null)
                    {
                        AddPathPoint(waypoint.Position);
                    }
                }
            }

            AddPathPoint(destination);
            driving = pathPoints.Count > 0;
            ApplyPose(kinematicPosition, kinematicRotation, true);
            return driving;
        }

        public void StopAndHold(bool zeroVelocity)
        {
            driving = false;
            pathPoints.Clear();
            targetPointIndex = 0;
            currentSpeedMetersPerSecond = 0f;

            if (zeroVelocity)
            {
                SetVelocity(Vector3.zero, Vector3.zero);
            }

            playerCar?.SetExternalInput(0f, 0f, true);
        }

        public void ReleaseControl()
        {
            StopAndHold(true);
            playerCar?.ClearExternalInput();
            RestoreRigidbodyState();
        }

        public void TeleportTo(Vector3 position, Quaternion rotation, int nextWaypointIndex)
        {
            ResolveReferences();
            EnsureKinematicControl();
            StopAndHold(true);
            kinematicPosition = position;
            kinematicRotation = rotation;
            ApplyPose(position, rotation, true);
        }

        public void ClearCriticalFault()
        {
        }

        private void Awake()
        {
            ResolveReferences();
            kinematicPosition = BodyPosition;
            kinematicRotation = transform.rotation;
        }

        private void OnDisable()
        {
            ReleaseControl();
        }

        private void Update()
        {
            if (!driving)
            {
                currentSpeedMetersPerSecond = 0f;
                return;
            }

            ResolveReferences();
            EnsureKinematicControl();

            float deltaTime = Time.deltaTime;
            float travelDistance = Mathf.Max(0f, baseCruiseSpeedMetersPerSecond) *
                                   Mathf.Max(0.1f, speedMultiplier) *
                                   deltaTime;
            if (travelDistance <= 0f)
            {
                currentSpeedMetersPerSecond = 0f;
                return;
            }

            Vector3 previousPosition = kinematicPosition;
            AdvanceAlongPath(travelDistance);
            float movedDistance = GetPlanarDistance(previousPosition, kinematicPosition);
            currentSpeedMetersPerSecond = movedDistance / Mathf.Max(0.0001f, deltaTime);
            ApplyPose(kinematicPosition, kinematicRotation, false);
        }

        private void ResolveReferences()
        {
            if (vehicleRigidbody == null)
            {
                vehicleRigidbody = GetComponent<Rigidbody>();
            }

            if (playerCar == null)
            {
                playerCar = GetComponent<PlayerCar>();
            }
        }

        private void EnsureKinematicControl()
        {
            if (vehicleRigidbody == null)
            {
                return;
            }

            if (!hasOriginalRigidbodyState)
            {
                hasOriginalRigidbodyState = true;
                originalIsKinematic = vehicleRigidbody.isKinematic;
            }

            vehicleRigidbody.isKinematic = true;
            SetVelocity(Vector3.zero, Vector3.zero);
            playerCar?.SetExternalInput(0f, 0f, true);
        }

        private void RestoreRigidbodyState()
        {
            if (vehicleRigidbody == null || !hasOriginalRigidbodyState)
            {
                return;
            }

            vehicleRigidbody.isKinematic = originalIsKinematic;
            hasOriginalRigidbodyState = false;
        }

        private void AddPathPoint(Vector3 point)
        {
            if (pathPoints.Count > 0 && GetPlanarDistance(pathPoints[pathPoints.Count - 1], point) < 0.05f)
            {
                return;
            }

            pathPoints.Add(point);
        }

        private void AdvanceAlongPath(float remainingDistance)
        {
            while (driving && remainingDistance > 0f && targetPointIndex < pathPoints.Count)
            {
                Vector3 targetPoint = pathPoints[targetPointIndex];
                float reachDistance = targetPointIndex >= pathPoints.Count - 1
                    ? finalReachDistanceMeters
                    : waypointReachDistanceMeters;
                Vector3 toTarget = targetPoint - kinematicPosition;
                float targetDistance = toTarget.magnitude;

                if (targetDistance <= reachDistance)
                {
                    SnapToTarget(targetPoint);
                    continue;
                }

                if (remainingDistance >= targetDistance)
                {
                    remainingDistance -= targetDistance;
                    SnapToTarget(targetPoint);
                    continue;
                }

                Vector3 direction = toTarget / targetDistance;
                kinematicPosition += direction * remainingDistance;
                UpdateRotation(direction);
                remainingDistance = 0f;
            }
        }

        private void SnapToTarget(Vector3 targetPoint)
        {
            Vector3 direction = targetPoint - kinematicPosition;
            kinematicPosition = targetPoint;
            UpdateRotation(direction);
            targetPointIndex++;

            if (targetPointIndex >= pathPoints.Count)
            {
                driving = false;
            }
        }

        private void UpdateRotation(Vector3 direction)
        {
            direction.y = 0f;
            if (direction.sqrMagnitude > 0.001f)
            {
                kinematicRotation = Quaternion.LookRotation(direction.normalized, Vector3.up);
            }
        }

        private void ApplyPose(Vector3 position, Quaternion rotation, bool syncTransforms)
        {
            if (vehicleRigidbody != null)
            {
                vehicleRigidbody.position = position;
                vehicleRigidbody.rotation = rotation;
            }

            transform.SetPositionAndRotation(position, rotation);
            SetVelocity(Vector3.zero, Vector3.zero);

            if (syncTransforms)
            {
                Physics.SyncTransforms();
            }
        }

        private void SetVelocity(Vector3 linearVelocity, Vector3 angularVelocity)
        {
            if (vehicleRigidbody == null)
            {
                return;
            }

#if UNITY_6000_0_OR_NEWER
            vehicleRigidbody.linearVelocity = linearVelocity;
#else
            vehicleRigidbody.velocity = linearVelocity;
#endif
            vehicleRigidbody.angularVelocity = angularVelocity;
        }

        private static float GetPlanarDistance(Vector3 a, Vector3 b)
        {
            a.y = 0f;
            b.y = 0f;
            return Vector3.Distance(a, b);
        }
    }
}
