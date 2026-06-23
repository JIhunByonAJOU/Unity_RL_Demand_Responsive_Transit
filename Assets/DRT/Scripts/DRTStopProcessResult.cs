namespace DRT
{
    public readonly struct DRTStopProcessResult
    {
        public readonly int StopId;
        public readonly int BoardedCount;
        public readonly int DroppedOffCount;
        public readonly int WaitingCount;
        public readonly int OnBoardCount;
        public readonly int CompletedCount;

        public DRTStopProcessResult(
            int stopId,
            int boardedCount,
            int droppedOffCount,
            int waitingCount,
            int onBoardCount,
            int completedCount)
        {
            StopId = stopId;
            BoardedCount = boardedCount;
            DroppedOffCount = droppedOffCount;
            WaitingCount = waitingCount;
            OnBoardCount = onBoardCount;
            CompletedCount = completedCount;
        }
    }
}
