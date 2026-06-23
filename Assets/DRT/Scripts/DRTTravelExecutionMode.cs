using UnityEngine;

namespace DRT
{
    public enum DRTTravelExecutionMode
    {
        [InspectorName("Matrix Teleport")]
        MatrixTeleport,

        [InspectorName("Physical Drive")]
        PhysicalDrive,

        [InspectorName("Train")]
        Train
    }

    public enum DRTPhysicalDriveMode
    {
        [InspectorName("Gley")]
        Gley,

        [InspectorName("PPO Autonomous")]
        PPOAutonomous,

        [InspectorName("PPO PurePursuit")]
        PPOPurePursuit,

        [InspectorName("Noisy Gley")]
        NoisyGley
    }

    public enum DRTPPODrivePolicy
    {
        [InspectorName("ML-Agents Training")]
        MLAgentsTraining,

        [InspectorName("ONNX Inference")]
        ONNXInference,

        [InspectorName("Heuristic Pure Pursuit")]
        HeuristicPurePursuit
    }

    public enum DRTNextStopPolicy
    {
        [InspectorName("ML-Agents Training")]
        MLAgentsTraining,

        [InspectorName("ONNX Inference")]
        ONNXInference,

        [InspectorName("Vanilla Sequential")]
        VanillaSequential,

        [InspectorName("All Station Runner")]
        AllStationRunner,

        [InspectorName("Greedy 1 - Nearest Feasible")]
        GreedyNearestFeasible,

        [InspectorName("FIFO")]
        Fifo
    }
}
