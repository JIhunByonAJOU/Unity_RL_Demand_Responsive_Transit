using System;
using UnityEngine;

namespace DRT
{
    [Serializable]
    public class DRTPassengerRequest
    {
        [SerializeField] private int passengerId;
        [SerializeField] private int originStopId;
        [SerializeField] private int destinationStopId;
        [SerializeField] private float requestTimeSeconds;
        [SerializeField] private DRTPassengerStatus status;
        [SerializeField] private float pickupTimeSeconds = -1f;
        [SerializeField] private float dropoffTimeSeconds = -1f;
        [SerializeField] private int actualPickupStopId = -1;
        [SerializeField] private int actualDropoffStopId = -1;

        public int PassengerId => passengerId;
        public int OriginStopId => originStopId;
        public int DestinationStopId => destinationStopId;
        public float RequestTimeSeconds => requestTimeSeconds;
        public DRTPassengerStatus Status => status;
        public float PickupTimeSeconds => pickupTimeSeconds;
        public float DropoffTimeSeconds => dropoffTimeSeconds;
        public int ActualPickupStopId => actualPickupStopId;
        public int ActualDropoffStopId => actualDropoffStopId;

        public DRTPassengerRequest(
            int passengerId,
            int originStopId,
            int destinationStopId,
            float requestTimeSeconds,
            DRTPassengerStatus initialStatus = DRTPassengerStatus.Scheduled)
        {
            this.passengerId = passengerId;
            this.originStopId = originStopId;
            this.destinationStopId = destinationStopId;
            this.requestTimeSeconds = Mathf.Max(0f, requestTimeSeconds);
            status = initialStatus == DRTPassengerStatus.Waiting
                ? DRTPassengerStatus.Waiting
                : DRTPassengerStatus.Scheduled;
        }

        public void SetPassengerId(int newPassengerId)
        {
            passengerId = newPassengerId;
        }

        public bool ShouldStartWaiting(float currentEpisodeTime)
        {
            return status == DRTPassengerStatus.Scheduled && currentEpisodeTime >= requestTimeSeconds;
        }

        public void StartWaiting()
        {
            if (status == DRTPassengerStatus.Scheduled)
            {
                status = DRTPassengerStatus.Waiting;
            }
        }

        public bool CanBoardAtStop(int stopId, float currentEpisodeTime)
        {
            return status == DRTPassengerStatus.Waiting &&
                   currentEpisodeTime >= requestTimeSeconds &&
                   originStopId == stopId;
        }

        public bool CanDropOffAtStop(int stopId)
        {
            return status == DRTPassengerStatus.OnBoard && destinationStopId == stopId;
        }

        public void MarkBoarded(float currentEpisodeTime, int stopId)
        {
            status = DRTPassengerStatus.OnBoard;
            pickupTimeSeconds = currentEpisodeTime;
            actualPickupStopId = stopId;
        }

        public void MarkDroppedOff(float currentEpisodeTime, int stopId)
        {
            status = DRTPassengerStatus.Completed;
            dropoffTimeSeconds = currentEpisodeTime;
            actualDropoffStopId = stopId;
        }

        public float GetElapsedSinceRequest(float currentEpisodeTime)
        {
            if (dropoffTimeSeconds >= 0f)
            {
                return Mathf.Max(0f, dropoffTimeSeconds - requestTimeSeconds);
            }

            return Mathf.Max(0f, currentEpisodeTime - requestTimeSeconds);
        }

        public float GetWaitTime(float currentEpisodeTime)
        {
            if (pickupTimeSeconds >= 0f)
            {
                return Mathf.Max(0f, pickupTimeSeconds - requestTimeSeconds);
            }

            if (status == DRTPassengerStatus.Waiting)
            {
                return Mathf.Max(0f, currentEpisodeTime - requestTimeSeconds);
            }

            return 0f;
        }

        public float GetRideTime(float currentEpisodeTime)
        {
            if (pickupTimeSeconds < 0f)
            {
                return 0f;
            }

            if (dropoffTimeSeconds >= 0f)
            {
                return Mathf.Max(0f, dropoffTimeSeconds - pickupTimeSeconds);
            }

            if (status == DRTPassengerStatus.OnBoard)
            {
                return Mathf.Max(0f, currentEpisodeTime - pickupTimeSeconds);
            }

            return 0f;
        }

        public float GetTotalServiceTime(float currentEpisodeTime)
        {
            if (dropoffTimeSeconds >= 0f)
            {
                return Mathf.Max(0f, dropoffTimeSeconds - requestTimeSeconds);
            }

            return GetElapsedSinceRequest(currentEpisodeTime);
        }
    }
}
