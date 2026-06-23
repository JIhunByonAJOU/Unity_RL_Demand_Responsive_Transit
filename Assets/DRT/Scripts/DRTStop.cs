using UnityEngine;

namespace DRT
{
    public class DRTStop : MonoBehaviour
    {
        [SerializeField] private int stopId = 1;

        public int StopId => stopId;
        public Vector3 Position => transform.position;

        public void SetStopId(int newStopId)
        {
            stopId = Mathf.Max(1, newStopId);
        }

        private void OnValidate()
        {
            stopId = Mathf.Max(1, stopId);
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = new Color(0.1f, 0.55f, 1f, 0.85f);
            Gizmos.DrawSphere(transform.position + Vector3.up * 0.4f, 1.4f);
        }
    }
}
