using System.Collections.Generic;
using Gley.TrafficSystem;
using UnityEngine;

using VehicleTypes = Gley.TrafficSystem.User.VehicleTypes;

namespace DRT
{
    /// <summary>
    /// Applies the Noisy Gley scenario to the scene PlayerCar when it is not a
    /// pooled Gley VehicleComponent. Route following remains the existing
    /// PlayerCar driver; only its speed target and steering input are disturbed.
    /// </summary>
    [AddComponentMenu("DRT/DRT Noisy Player Vehicle Driver")]
    public class DRTNoisyPlayerVehicleDriver : MonoBehaviour, IDRTVehicleDriver
    {
        private const float DefaultPlayerCruiseSpeedMetersPerSecond = 10f;
        private const float MinSpeedMultiplier = 0.1f;
        private const float MaxSpeedMultiplier = 2f;
        private const float MinNoiseFrequency = 0.01f;
        private const float MaxLateralNoise = 3f;
        private const float MaxNoiseStrength = 3f;
        private const float SpeedMultiplierApplyEpsilon = 0.005f;

        private DRTPlayerVehicleDriver baseDriver;
        private VehicleTypes vehicleType = VehicleTypes.Car;
        private float baseSpeedMultiplier = 1f;
        private float waypointReachDistanceMeters = 6f;
        private float finalReachDistanceMeters = 4f;
        private float lateralNoise;
        private float speedNoise;
        private float noiseFrequency = 1f;
        private float noiseStrength = 1f;
        private float noiseIrregularity = 0.65f;
        private bool usePolicySpeedLimit;
        private float policySpeedLimitMetersPerSecond;
        private float lateralNoiseSeed;
        private float speedNoiseSeed;
        private float lastAppliedSpeedMultiplier = float.NaN;

        public Transform VehicleTransform => baseDriver != null ? baseDriver.VehicleTransform : transform;
        public string VehicleName => baseDriver != null ? $"{baseDriver.VehicleName} (Noisy Gley)" : $"{name} (Noisy Gley)";
        public float CurrentSpeedMS => baseDriver != null ? baseDriver.CurrentSpeedMS : 0f;
        public int PathPointCount => baseDriver != null ? baseDriver.PathPointCount : 0;
        public int RemainingPathPointCount => baseDriver != null ? baseDriver.RemainingPathPointCount : 0;
        public Vector3 BodyPosition => baseDriver != null ? baseDriver.BodyPosition : transform.position;
        public bool IsTemporarilyBlocked => baseDriver != null && baseDriver.IsTemporarilyBlocked;
        public string TemporaryBlockReason => baseDriver != null ? baseDriver.TemporaryBlockReason : string.Empty;
        public bool HasCriticalFault => baseDriver != null && baseDriver.HasCriticalFault;
        public string CriticalFaultReason => baseDriver != null ? baseDriver.CriticalFaultReason : string.Empty;

        public void Configure(
            VehicleTypes newVehicleType,
            float newBaseSpeedMultiplier,
            float newWaypointReachDistance,
            float newFinalReachDistance,
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
            vehicleType = newVehicleType;
            baseSpeedMultiplier = Mathf.Clamp(newBaseSpeedMultiplier, MinSpeedMultiplier, MaxSpeedMultiplier);
            waypointReachDistanceMeters = Mathf.Max(0.5f, newWaypointReachDistance);
            finalReachDistanceMeters = Mathf.Max(0.25f, newFinalReachDistance);
            lateralNoise = Mathf.Clamp(newLateralNoise, 0f, MaxLateralNoise);
            speedNoise = Mathf.Clamp(newSpeedNoise, 0f, 0.5f);
            noiseFrequency = Mathf.Max(MinNoiseFrequency, newNoiseFrequency);
            noiseStrength = Mathf.Clamp(newNoiseStrength, 0f, MaxNoiseStrength);
            noiseIrregularity = Mathf.Clamp01(newNoiseIrregularity);
            usePolicySpeedLimit = newUsePolicySpeedLimit;
            policySpeedLimitMetersPerSecond = Mathf.Max(0f, newPolicySpeedLimitMetersPerSecond);
            lateralNoiseSeed = 13.37f + transform.GetInstanceID() * 0.001f;
            speedNoiseSeed = 71.13f + transform.GetInstanceID() * 0.001f;

            ResolveReferences();
            ApplySpeedNoise(true);
            ApplySteeringNoise(Time.time * noiseFrequency);
        }

        public bool SetPath(List<int> waypointIndexes, Vector3 destination)
        {
            if (!ResolveReferences())
            {
                return false;
            }

            ApplySpeedNoise(true);
            ApplySteeringNoise(Time.time * noiseFrequency);
            return baseDriver.SetPath(waypointIndexes, destination);
        }

        public void StopAndHold(bool zeroVelocity)
        {
            baseDriver?.SetSteeringNoiseInput(0f);
            baseDriver?.StopAndHold(zeroVelocity);
        }

        public void ReleaseControl()
        {
            baseDriver?.SetSteeringNoiseInput(0f);
            baseDriver?.ReleaseControl();
            lastAppliedSpeedMultiplier = float.NaN;
        }

        public void TeleportTo(Vector3 position, Quaternion rotation, int nextWaypointIndex)
        {
            if (ResolveReferences())
            {
                baseDriver.TeleportTo(position, rotation, nextWaypointIndex);
            }
        }

        public void ClearCriticalFault()
        {
            baseDriver?.ClearCriticalFault();
        }

        private void Awake()
        {
            ResolveReferences();
        }

        private void OnDisable()
        {
            ReleaseControl();
        }

        private void Update()
        {
            ApplySpeedNoise(false);
        }

        private void FixedUpdate()
        {
            if (baseDriver == null || !baseDriver.IsDriving)
            {
                baseDriver?.SetSteeringNoiseInput(0f);
                return;
            }

            ApplySteeringNoise(Time.fixedTime * noiseFrequency);
        }

        private bool ResolveReferences()
        {
            if (baseDriver == null)
            {
                baseDriver = GetComponent<DRTPlayerVehicleDriver>();
                if (baseDriver == null)
                {
                    baseDriver = gameObject.AddComponent<DRTPlayerVehicleDriver>();
                }
            }

            if (baseDriver != null)
            {
                baseDriver.enabled = true;
            }

            return baseDriver != null;
        }

        private void ApplySpeedNoise(bool force)
        {
            if (!ResolveReferences())
            {
                return;
            }

            float noiseTime = Time.time * noiseFrequency;
            float speedCorrection = speedNoise > 0f
                ? 1f + DRTNoisyVehicleNoise.SampleIrregular(speedNoiseSeed, noiseTime * 0.83f, noiseIrregularity) * speedNoise
                : 1f;
            float effectiveSpeedMultiplier = GetEffectiveSpeedMultiplier(speedCorrection);
            if (!force &&
                !float.IsNaN(lastAppliedSpeedMultiplier) &&
                Mathf.Abs(lastAppliedSpeedMultiplier - effectiveSpeedMultiplier) < SpeedMultiplierApplyEpsilon)
            {
                return;
            }

            baseDriver.Configure(
                vehicleType,
                effectiveSpeedMultiplier,
                waypointReachDistanceMeters,
                finalReachDistanceMeters);
            lastAppliedSpeedMultiplier = effectiveSpeedMultiplier;
        }

        private void ApplySteeringNoise(float noiseTime)
        {
            if (!ResolveReferences())
            {
                return;
            }

            baseDriver.SetSteeringNoiseInput(GetSteeringNoisePercent(noiseTime));
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

        private float GetEffectiveSpeedMultiplier(float speedCorrection)
        {
            float multiplier = Mathf.Clamp(
                baseSpeedMultiplier * speedCorrection,
                MinSpeedMultiplier,
                MaxSpeedMultiplier);
            if (usePolicySpeedLimit && policySpeedLimitMetersPerSecond > 0f)
            {
                multiplier = Mathf.Min(
                    multiplier,
                    policySpeedLimitMetersPerSecond / DefaultPlayerCruiseSpeedMetersPerSecond);
            }

            return Mathf.Clamp(multiplier, MinSpeedMultiplier, MaxSpeedMultiplier);
        }

        private void OnValidate()
        {
            baseSpeedMultiplier = Mathf.Clamp(baseSpeedMultiplier, MinSpeedMultiplier, MaxSpeedMultiplier);
            waypointReachDistanceMeters = Mathf.Max(0.5f, waypointReachDistanceMeters);
            finalReachDistanceMeters = Mathf.Max(0.25f, finalReachDistanceMeters);
            lateralNoise = Mathf.Clamp(lateralNoise, 0f, MaxLateralNoise);
            speedNoise = Mathf.Clamp(speedNoise, 0f, 0.5f);
            noiseFrequency = Mathf.Max(MinNoiseFrequency, noiseFrequency);
            noiseStrength = Mathf.Clamp(noiseStrength, 0f, MaxNoiseStrength);
            noiseIrregularity = Mathf.Clamp01(noiseIrregularity);
            policySpeedLimitMetersPerSecond = Mathf.Max(0f, policySpeedLimitMetersPerSecond);
        }
    }
}
