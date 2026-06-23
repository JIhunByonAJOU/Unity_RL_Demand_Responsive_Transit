using System.Collections.Generic;
using Gley.TrafficSystem;
using UnityEngine;

using VehicleTypes = Gley.TrafficSystem.User.VehicleTypes;

namespace DRT
{
    [AddComponentMenu("DRT/DRT Noisy Gley Vehicle Driver")]
    public class DRTNoisyGleyVehicleDriver : MonoBehaviour, IDRTVehicleDriver
    {
        private const float MinSpeedMultiplier = 0.1f;
        private const float MaxSpeedMultiplier = 2f;
        private const float MinNoiseFrequency = 0.01f;
        private const float MaxLateralNoise = 3f;
        private const float MaxNoiseStrength = 3f;
        private const float SpeedCorrectionApplyEpsilon = 0.001f;

        private DRTGleyVehicleDriver baseDriver;
        private VehicleComponent vehicleComponent;
        private int vehicleIndex;
        private VehicleTypes vehicleType = VehicleTypes.Car;
        private float baseSpeedMultiplier = 1f;
        private float lateralNoise;
        private float speedNoise;
        private float noiseFrequency = 0.35f;
        private float noiseStrength = 1f;
        private float noiseIrregularity = 0.65f;
        private bool usePolicySpeedLimit;
        private float policySpeedLimitMetersPerSecond;
        private float lateralNoiseSeed;
        private float speedNoiseSeed;
        private float lastAppliedSpeedCorrection = float.NaN;

        public Transform VehicleTransform => baseDriver != null ? baseDriver.VehicleTransform : null;
        public string VehicleName => baseDriver != null ? $"{baseDriver.VehicleName} (Noisy Gley)" : $"NoisyGleyVehicle[{vehicleIndex}]";
        public int VehicleIndex => vehicleIndex;
        public float CurrentSpeedMS => baseDriver != null ? baseDriver.CurrentSpeedMS : 0f;
        public int PathPointCount => baseDriver != null ? baseDriver.PathPointCount : 0;
        public int RemainingPathPointCount => baseDriver != null ? baseDriver.RemainingPathPointCount : 0;
        public Vector3 BodyPosition => baseDriver != null ? baseDriver.BodyPosition : Vector3.zero;
        public bool IsTemporarilyBlocked => baseDriver != null && baseDriver.IsTemporarilyBlocked;
        public string TemporaryBlockReason => baseDriver != null ? baseDriver.TemporaryBlockReason : string.Empty;
        public bool HasCriticalFault => baseDriver != null && baseDriver.HasCriticalFault;
        public string CriticalFaultReason => baseDriver != null ? baseDriver.CriticalFaultReason : string.Empty;

        public void Configure(
            int newVehicleIndex,
            VehicleTypes newVehicleType,
            float newBaseSpeedMultiplier,
            float newLateralNoise,
            float newSpeedNoise,
            float newNoiseFrequency,
            float newNoiseStrength,
            float newNoiseIrregularity,
            bool newUsePolicySpeedLimit,
            float newPolicySpeedLimitMetersPerSecond)
        {
            hideFlags = HideFlags.HideInInspector;
            enabled = true;
            vehicleIndex = Mathf.Max(0, newVehicleIndex);
            vehicleType = newVehicleType;
            baseSpeedMultiplier = Mathf.Clamp(newBaseSpeedMultiplier, MinSpeedMultiplier, MaxSpeedMultiplier);
            lateralNoise = Mathf.Clamp(newLateralNoise, 0f, MaxLateralNoise);
            speedNoise = Mathf.Clamp(newSpeedNoise, 0f, 1f);
            noiseFrequency = Mathf.Max(MinNoiseFrequency, newNoiseFrequency);
            noiseStrength = Mathf.Clamp(newNoiseStrength, 0f, MaxNoiseStrength);
            noiseIrregularity = Mathf.Clamp01(newNoiseIrregularity);
            usePolicySpeedLimit = newUsePolicySpeedLimit;
            policySpeedLimitMetersPerSecond = Mathf.Max(0f, newPolicySpeedLimitMetersPerSecond);
            lateralNoiseSeed = 13.37f + vehicleIndex * 7.11f;
            speedNoiseSeed = 71.13f + vehicleIndex * 5.19f;

            ResolveBaseDriver();
            ResolveVehicleComponent(false);
            baseDriver.Configure(vehicleIndex, vehicleType, GetEffectiveBaseSpeedMultiplier());
            ApplyNoisyGleySafetySettings(true);
            ResetAppliedNoise();
            ApplyNoise();
        }

        public bool SetPath(List<int> waypointIndexes, Vector3 destination)
        {
            if (!ResolveBaseDriver())
            {
                return false;
            }

            ResolveVehicleComponent(false);
            baseDriver.Configure(vehicleIndex, vehicleType, GetEffectiveBaseSpeedMultiplier());
            ApplyNoisyGleySafetySettings(true);
            bool assigned = baseDriver.SetPath(waypointIndexes, destination);
            if (assigned)
            {
                ResolveVehicleComponent(false);
                ApplyNoise();
            }

            return assigned;
        }

        public void StopAndHold(bool zeroVelocity)
        {
            ResetAppliedNoise();
            if (TryResolveExistingBaseDriver())
            {
                baseDriver.StopAndHold(zeroVelocity);
            }
        }

        public void ReleaseControl()
        {
            ResetAppliedNoise();
            ApplyNoisyGleySafetySettings(false);
            if (TryResolveExistingBaseDriver())
            {
                baseDriver.ReleaseControl();
            }
        }

        public void TeleportTo(Vector3 position, Quaternion rotation, int nextWaypointIndex)
        {
            if (!ResolveBaseDriver())
            {
                return;
            }

            baseDriver.TeleportTo(position, rotation, nextWaypointIndex);
            ResolveVehicleComponent(false);
            ResetAppliedNoise();
            ApplyNoise();
        }

        public void ClearCriticalFault()
        {
            baseDriver?.ClearCriticalFault();
        }

        private void Update()
        {
            ApplyNoise();
        }

        private void FixedUpdate()
        {
            ApplySteeringNoise(Time.fixedTime * noiseFrequency);
        }

        private bool ResolveBaseDriver()
        {
            if (baseDriver == null)
            {
                baseDriver = GetComponent<DRTGleyVehicleDriver>();
                if (baseDriver == null)
                {
                    baseDriver = gameObject.AddComponent<DRTGleyVehicleDriver>();
                }
            }

            return baseDriver != null;
        }

        private bool TryResolveExistingBaseDriver()
        {
            if (baseDriver == null)
            {
                baseDriver = GetComponent<DRTGleyVehicleDriver>();
            }

            return baseDriver != null;
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
                Debug.LogWarning($"[DRT] Noisy Gley vehicle component not found. vehicleIndex={vehicleIndex}");
            }

            return vehicleComponent != null;
        }

        private void ApplyNoise()
        {
            if (!ResolveVehicleComponent(false) || vehicleComponent.MovementInfo == null)
            {
                return;
            }

            float noiseTime = Time.time * noiseFrequency;
            bool trafficSafetyActive = IsGleyTrafficSafetyActive();
            if (trafficSafetyActive)
            {
                EnsureGleyTrafficSafetyBehaviour();
            }

            ApplySteeringNoise(noiseTime, trafficSafetyActive);

            float speedCorrection = !trafficSafetyActive && speedNoise > 0f
                ? 1f + DRTNoisyVehicleNoise.SampleIrregular(speedNoiseSeed, noiseTime * 0.83f, noiseIrregularity) * speedNoise
                : 1f;
            ApplySpeedCorrection(Mathf.Clamp(speedCorrection, MinSpeedMultiplier, MaxSpeedMultiplier));
        }

        private void ApplySteeringNoise(float noiseTime)
        {
            if (!ResolveVehicleComponent(false))
            {
                return;
            }

            ApplySteeringNoise(noiseTime, IsGleyTrafficSafetyActive());
        }

        private void ApplySteeringNoise(float noiseTime, bool trafficSafetyActive)
        {
            API.SetSteeringNoisePercent(
                vehicleIndex,
                trafficSafetyActive ? 0f : GetSteeringNoisePercent(noiseTime));
        }

        private float GetSteeringNoisePercent(float noiseTime)
        {
            if (lateralNoise <= 0f || noiseStrength <= 0f)
            {
                return 0f;
            }

            float steeringSignal = DRTNoisyVehicleNoise.SampleIrregular(lateralNoiseSeed, noiseTime, noiseIrregularity);
            float steeringEnvelope = DRTNoisyVehicleNoise.SampleEnvelope(lateralNoiseSeed + 547.1f, noiseTime, noiseIrregularity);
            return Mathf.Clamp(steeringSignal * lateralNoise * noiseStrength * steeringEnvelope, -1f, 1f);
        }

        private bool IsGleyTrafficSafetyActive()
        {
            if (vehicleComponent == null || vehicleComponent.MovementInfo == null)
            {
                return false;
            }

            MovementInfo movementInfo = vehicleComponent.MovementInfo;
            return movementInfo.HasObstacles() ||
                   movementInfo.HasStopWaypoints() ||
                   movementInfo.HasGiveWayWaypoints() ||
                   movementInfo.HasSlowDownWaypoints();
        }

        private void EnsureGleyTrafficSafetyBehaviour()
        {
            if (vehicleComponent == null ||
                vehicleComponent.MovementInfo == null ||
                !vehicleComponent.MovementInfo.HasObstacles())
            {
                return;
            }

            API.StartVehicleBehaviour<FollowVehicle>(vehicleIndex);
        }

        private void ApplySpeedCorrection(float speedCorrection)
        {
            speedCorrection = GetLimitedSpeedCorrection(speedCorrection);
            if (!float.IsNaN(lastAppliedSpeedCorrection) &&
                Mathf.Abs(lastAppliedSpeedCorrection - speedCorrection) < SpeedCorrectionApplyEpsilon)
            {
                return;
            }

            vehicleComponent.MovementInfo.SetMaxSpeedCorrectionPercent(speedCorrection);
            float dynamicSpeedMultiplier = Mathf.Clamp(
                GetEffectiveBaseSpeedMultiplier() * speedCorrection,
                MinSpeedMultiplier,
                MaxSpeedMultiplier);
            vehicleComponent.MovementInfo.SetMaxVehicleSpeed(GetLimitedMaxSpeedMetersPerSecond(dynamicSpeedMultiplier));
            lastAppliedSpeedCorrection = speedCorrection;
        }

        private void ResetAppliedNoise()
        {
            if (!ResolveVehicleComponent(false) || vehicleComponent.MovementInfo == null)
            {
                lastAppliedSpeedCorrection = float.NaN;
                return;
            }

            API.SetSteeringNoisePercent(vehicleIndex, 0f);
            vehicleComponent.MovementInfo.SetMaxSpeedCorrectionPercent(1f);
            vehicleComponent.MovementInfo.SetMaxVehicleSpeed(vehicleComponent.MaxSpeed * baseSpeedMultiplier);
            lastAppliedSpeedCorrection = 1f;
        }

        private float GetEffectiveBaseSpeedMultiplier()
        {
            if (!usePolicySpeedLimit ||
                policySpeedLimitMetersPerSecond <= 0f ||
                vehicleComponent == null ||
                vehicleComponent.MaxSpeed <= 0f)
            {
                return baseSpeedMultiplier;
            }

            return Mathf.Clamp(
                Mathf.Min(baseSpeedMultiplier, policySpeedLimitMetersPerSecond / vehicleComponent.MaxSpeed),
                MinSpeedMultiplier,
                MaxSpeedMultiplier);
        }

        private float GetLimitedSpeedCorrection(float requestedCorrection)
        {
            requestedCorrection = Mathf.Clamp(requestedCorrection, MinSpeedMultiplier, MaxSpeedMultiplier);
            if (!usePolicySpeedLimit ||
                policySpeedLimitMetersPerSecond <= 0f ||
                vehicleComponent == null ||
                vehicleComponent.MovementInfo == null ||
                vehicleComponent.MovementInfo.PathLength <= 0)
            {
                return requestedCorrection;
            }

            float previousCorrection = !float.IsNaN(lastAppliedSpeedCorrection) && lastAppliedSpeedCorrection > 0f
                ? lastAppliedSpeedCorrection
                : 1f;
            float uncorrectedWaypointSpeed = vehicleComponent.MovementInfo.GetFirstWaypointSpeed() / previousCorrection;
            if (uncorrectedWaypointSpeed <= 0.001f)
            {
                return requestedCorrection;
            }

            float limitCorrection = policySpeedLimitMetersPerSecond / uncorrectedWaypointSpeed;
            return Mathf.Clamp(
                Mathf.Min(requestedCorrection, limitCorrection),
                0f,
                MaxSpeedMultiplier);
        }

        private float GetLimitedMaxSpeedMetersPerSecond(float speedMultiplier)
        {
            float maxSpeedMetersPerSecond = vehicleComponent.MaxSpeed * speedMultiplier;
            if (usePolicySpeedLimit && policySpeedLimitMetersPerSecond > 0f)
            {
                maxSpeedMetersPerSecond = Mathf.Min(maxSpeedMetersPerSecond, policySpeedLimitMetersPerSecond);
            }

            return maxSpeedMetersPerSecond;
        }

        private void ApplyNoisyGleySafetySettings(bool active)
        {
            if (!API.IsInitialized())
            {
                return;
            }

            if (API.GetVehicleBehaviourOfType<FollowVehicle>(vehicleIndex) is FollowVehicle followVehicle)
            {
                followVehicle.DisableOvertake(active);
            }
        }

    }

    internal static class DRTNoisyVehicleNoise
    {
        public static float SampleIrregular(float seed, float noiseTime, float irregularity)
        {
            irregularity = Mathf.Clamp01(irregularity);
            float regular = SampleCentered(seed, noiseTime);
            if (irregularity <= 0.001f)
            {
                return regular;
            }

            float warp = SampleCentered(seed + 503.2f, noiseTime * 0.23f) * irregularity * 2.1f;
            float warpedTime = noiseTime + warp;
            float slow = SampleCentered(seed, warpedTime * 0.61f);
            float mid = SampleCentered(seed + 113.7f, warpedTime * 1.73f + slow * irregularity);
            float fast = SampleCentered(seed + 271.1f, warpedTime * 4.97f);
            float jerkSource = SampleCentered(seed + 389.4f, warpedTime * 7.31f);
            float jerk = Mathf.Sign(jerkSource) * Mathf.Pow(Mathf.Abs(jerkSource), 2.5f);
            float mixed = slow * 0.54f + mid * 0.31f + fast * 0.1f + jerk * 0.16f;
            return Mathf.Clamp(Mathf.Lerp(regular, mixed * 1.12f, irregularity), -1f, 1f);
        }

        public static float SampleEnvelope(float seed, float noiseTime, float irregularity)
        {
            irregularity = Mathf.Clamp01(irregularity);
            if (irregularity <= 0.001f)
            {
                return 1f;
            }

            float burst = Mathf.Abs(SampleIrregular(seed, noiseTime * 0.37f, irregularity));
            return Mathf.Lerp(1f, Mathf.Lerp(0.45f, 1.65f, burst), irregularity);
        }

        private static float SampleCentered(float seed, float noiseTime)
        {
            float primary = Mathf.PerlinNoise(seed, noiseTime * 0.73f) - 0.5f;
            float secondary = Mathf.PerlinNoise(seed + 31.7f, noiseTime * 1.61f) - 0.5f;
            return Mathf.Clamp((primary * 1.6f + secondary * 1.1f) * 1.35f, -1f, 1f);
        }
    }
}
