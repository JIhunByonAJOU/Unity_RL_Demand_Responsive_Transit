using System.Collections.Generic;
using Gley.TrafficSystem;
using UnityEngine;

using VehicleTypes = Gley.TrafficSystem.User.VehicleTypes;

namespace DRT
{
    [AddComponentMenu("DRT/DRT Gley Vehicle Driver")]
    public class DRTGleyVehicleDriver : MonoBehaviour, IDRTVehicleDriver
    {
        [SerializeField] private int vehicleIndex;
        [SerializeField] private VehicleTypes vehicleType = VehicleTypes.Car;
        [SerializeField] private float speedMultiplier = 1f;
        [SerializeField] private bool useSpeedLimit;
        [SerializeField] private float speedLimitMetersPerSecond;

        private VehicleComponent vehicleComponent;

        public Transform VehicleTransform => vehicleComponent != null ? vehicleComponent.transform : null;
        public string VehicleName => vehicleComponent != null ? vehicleComponent.name : $"GleyVehicle[{vehicleIndex}]";
        public int VehicleIndex => vehicleIndex;
        public VehicleTypes VehicleType => vehicleType;
        public float SpeedMultiplier => speedMultiplier;
        public bool UseSpeedLimit => useSpeedLimit;
        public float SpeedLimitMetersPerSecond => speedLimitMetersPerSecond;
        public int PathPointCount => vehicleComponent != null ? vehicleComponent.MovementInfo.PathLength : 0;
        public int RemainingPathPointCount => vehicleComponent != null ? vehicleComponent.MovementInfo.RemainingPathLength : 0;
        public Vector3 BodyPosition => GetBodyPosition();
        public bool IsTemporarilyBlocked => TryGetTemporaryBlockReason(out _);
        public string TemporaryBlockReason => TryGetTemporaryBlockReason(out string reason) ? reason : string.Empty;
        public bool HasCriticalFault => false;
        public string CriticalFaultReason => string.Empty;

        public float CurrentSpeedMS
        {
            get
            {
                return ResolveVehicleComponent(false) && vehicleComponent != null
                    ? vehicleComponent.GetCurrentSpeedMS()
                    : 0f;
            }
        }

        public void Configure(int newVehicleIndex, VehicleTypes newVehicleType, float newSpeedMultiplier = 1f)
        {
            Configure(newVehicleIndex, newVehicleType, newSpeedMultiplier, false, 0f);
        }

        public void Configure(
            int newVehicleIndex,
            VehicleTypes newVehicleType,
            float newSpeedMultiplier,
            bool newUseSpeedLimit,
            float newSpeedLimitMetersPerSecond)
        {
            hideFlags = HideFlags.HideInInspector;
            vehicleIndex = Mathf.Max(0, newVehicleIndex);
            vehicleType = newVehicleType;
            speedMultiplier = Mathf.Clamp(newSpeedMultiplier, 0.1f, 2f);
            useSpeedLimit = newUseSpeedLimit;
            speedLimitMetersPerSecond = Mathf.Max(0f, newSpeedLimitMetersPerSecond);
            ResolveVehicleComponent(false);
            ApplySpeedMultiplier();
        }

        public bool SetPath(List<int> waypointIndexes, Vector3 destination)
        {
            if (!ResolveVehicleComponent(true) || waypointIndexes == null || waypointIndexes.Count == 0)
            {
                return false;
            }

            ApplySpeedMultiplier();
            API.DontRemoveVehicle(vehicleIndex, true);
            API.ResumeVehicleDriving(vehicleComponent.gameObject);
            API.SetVehiclePath(vehicleIndex, waypointIndexes);
            return true;
        }

        public void StopAndHold(bool zeroVelocity)
        {
            if (!ResolveVehicleComponent(false) || vehicleComponent == null)
            {
                return;
            }

            API.StopVehicleDriving(vehicleComponent.gameObject);
            API.RemoveVehiclePath(vehicleIndex);

            if (zeroVelocity)
            {
                vehicleComponent.SetVelocity(Vector3.zero, Vector3.zero);
            }
        }

        public void ReleaseControl()
        {
            if (!ResolveVehicleComponent(false) || vehicleComponent == null)
            {
                return;
            }

            API.DontRemoveVehicle(vehicleIndex, false);
            API.RemoveVehiclePath(vehicleIndex);
        }

        public void TeleportTo(Vector3 position, Quaternion rotation, int nextWaypointIndex)
        {
            if (!ResolveVehicleComponent(true))
            {
                return;
            }

            API.DontRemoveVehicle(vehicleIndex, true);
            API.InstantiateVehicleOnTheSpot(
                vehicleIndex,
                position,
                rotation,
                Vector3.zero,
                Vector3.zero,
                nextWaypointIndex);

            ResolveVehicleComponent(false);
            if (vehicleComponent != null)
            {
                vehicleComponent.SetVelocity(Vector3.zero, Vector3.zero);
                ApplySpeedMultiplier();
            }
        }

        public void ClearCriticalFault()
        {
        }

        private void ApplySpeedMultiplier()
        {
            if (vehicleComponent == null || vehicleComponent.MovementInfo == null)
            {
                return;
            }

            float multiplier = Mathf.Clamp(speedMultiplier, 0.1f, 2f);
            vehicleComponent.MovementInfo.SetSpeedVariationPercent(1f - multiplier, multiplier - 1f);
            vehicleComponent.MovementInfo.SetMaxVehicleSpeed(GetLimitedMaxSpeedMetersPerSecond(multiplier));
        }

        private float GetLimitedMaxSpeedMetersPerSecond(float multiplier)
        {
            float maxSpeedMetersPerSecond = vehicleComponent.MaxSpeed * multiplier;
            if (useSpeedLimit && speedLimitMetersPerSecond > 0f)
            {
                maxSpeedMetersPerSecond = Mathf.Min(maxSpeedMetersPerSecond, speedLimitMetersPerSecond);
            }

            return maxSpeedMetersPerSecond;
        }

        private bool TryGetTemporaryBlockReason(out string reason)
        {
            reason = string.Empty;
            if (!ResolveVehicleComponent(false) || vehicleComponent == null || vehicleComponent.MovementInfo == null)
            {
                return false;
            }

            var movementInfo = vehicleComponent.MovementInfo;
            if (movementInfo.HasStopWaypoints())
            {
                reason = "traffic stop or signal";
                return true;
            }

            if (movementInfo.HasGiveWayWaypoints())
            {
                reason = "give-way rule";
                return true;
            }

            if (movementInfo.HasObstacles())
            {
                reason = "obstacle";
                return true;
            }

            return false;
        }

        private bool ResolveVehicleComponent(bool logIfMissing)
        {
            if (!API.IsInitialized())
            {
                return false;
            }

            vehicleComponent = API.GetVehicleComponent(vehicleIndex);
            if (vehicleComponent == null && logIfMissing)
            {
                Debug.LogWarning($"[DRT] Gley vehicle component not found. vehicleIndex={vehicleIndex}");
            }

            return vehicleComponent != null;
        }

        private Vector3 GetBodyPosition()
        {
            if (!ResolveVehicleComponent(false) || vehicleComponent == null)
            {
                return Vector3.zero;
            }

            return vehicleComponent.rb != null
                ? vehicleComponent.rb.position
                : vehicleComponent.transform.position;
        }
    }
}
