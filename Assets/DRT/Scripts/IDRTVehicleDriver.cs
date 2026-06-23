using System.Collections.Generic;
using UnityEngine;

namespace DRT
{
    public interface IDRTVehicleDriver
    {
        Transform VehicleTransform { get; }
        string VehicleName { get; }
        float CurrentSpeedMS { get; }
        int PathPointCount { get; }
        int RemainingPathPointCount { get; }
        Vector3 BodyPosition { get; }
        bool IsTemporarilyBlocked { get; }
        string TemporaryBlockReason { get; }
        bool HasCriticalFault { get; }
        string CriticalFaultReason { get; }

        bool SetPath(List<int> waypointIndexes, Vector3 destination);
        void StopAndHold(bool zeroVelocity);
        void ReleaseControl();
        void TeleportTo(Vector3 position, Quaternion rotation, int nextWaypointIndex);
        void ClearCriticalFault();
    }
}
