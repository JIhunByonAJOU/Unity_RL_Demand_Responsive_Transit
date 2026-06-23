using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace DRT
{
    public sealed class DRTStopTravelTimeMatrix
    {
        private readonly Dictionary<int, int> stopIdToIndex = new Dictionary<int, int>();
        private float[,] travelTimeSeconds = new float[0, 0];
        private int stopCount;

        public int StopCount => stopCount;
        public bool IsLoaded => stopCount > 0;

        public bool LoadFromCsv(string csvText, IReadOnlyList<DRTStop> stops, out string error)
        {
            Clear();
            error = null;

            if (string.IsNullOrWhiteSpace(csvText))
            {
                error = "CSV text is empty.";
                return false;
            }

            List<DRTStop> sortedStops = GetSortedStops(stops);
            if (sortedStops.Count == 0)
            {
                error = "No stops are available.";
                return false;
            }

            string[] rawLines = csvText.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var lines = new List<string>();
            for (int i = 0; i < rawLines.Length; i++)
            {
                string line = rawLines[i].Trim();
                line = line.TrimStart('\uFEFF');
                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                lines.Add(line);
            }

            if (lines.Count != sortedStops.Count)
            {
                error = $"CSV must have {sortedStops.Count} rows, but has {lines.Count}.";
                return false;
            }

            var parsed = new float[sortedStops.Count, sortedStops.Count];
            for (int row = 0; row < lines.Count; row++)
            {
                string[] columns = lines[row].Split(',');
                if (columns.Length != sortedStops.Count)
                {
                    error = $"CSV row {row + 1} must have {sortedStops.Count} columns, but has {columns.Length}.";
                    return false;
                }

                for (int column = 0; column < columns.Length; column++)
                {
                    if (!float.TryParse(columns[column].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float value))
                    {
                        error = $"CSV row {row + 1}, column {column + 1} is not a number.";
                        return false;
                    }

                    if (row == column)
                    {
                        value = 0f;
                    }
                    else if (value <= 0f)
                    {
                        error = $"CSV row {row + 1}, column {column + 1} must be positive for non-diagonal travel.";
                        return false;
                    }

                    parsed[row, column] = value;
                }
            }

            Apply(sortedStops, parsed);
            return true;
        }

        public bool TryGetTravelTimeSeconds(int fromStopId, int toStopId, out float seconds)
        {
            seconds = 0f;

            if (fromStopId == toStopId)
            {
                return true;
            }

            if (!stopIdToIndex.TryGetValue(fromStopId, out int fromIndex) ||
                !stopIdToIndex.TryGetValue(toStopId, out int toIndex))
            {
                return false;
            }

            seconds = travelTimeSeconds[fromIndex, toIndex];
            return seconds > 0f;
        }

        public bool TryGetAverageTravelTimeMinutes(IReadOnlyList<DRTStop> stops, out float averageMinutes)
        {
            averageMinutes = 0f;

            if (!IsLoaded || stops == null)
            {
                return false;
            }

            float totalSeconds = 0f;
            int pairCount = 0;

            for (int originIndex = 0; originIndex < stops.Count; originIndex++)
            {
                DRTStop origin = stops[originIndex];
                if (origin == null)
                {
                    continue;
                }

                for (int destinationIndex = 0; destinationIndex < stops.Count; destinationIndex++)
                {
                    DRTStop destination = stops[destinationIndex];
                    if (destination == null || destination.StopId == origin.StopId)
                    {
                        continue;
                    }

                    if (TryGetTravelTimeSeconds(origin.StopId, destination.StopId, out float seconds))
                    {
                        totalSeconds += seconds;
                        pairCount++;
                    }
                }
            }

            if (pairCount == 0)
            {
                return false;
            }

            averageMinutes = totalSeconds / pairCount / 60f;
            return true;
        }

        public string ToCsv()
        {
            if (!IsLoaded)
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            for (int row = 0; row < stopCount; row++)
            {
                if (row > 0)
                {
                    builder.AppendLine();
                }

                for (int column = 0; column < stopCount; column++)
                {
                    if (column > 0)
                    {
                        builder.Append(',');
                    }

                    builder.Append(travelTimeSeconds[row, column].ToString("0.##", CultureInfo.InvariantCulture));
                }
            }

            return builder.ToString();
        }

        private void Apply(List<DRTStop> sortedStops, float[,] matrix)
        {
            stopIdToIndex.Clear();
            stopCount = sortedStops.Count;
            travelTimeSeconds = matrix;

            for (int i = 0; i < sortedStops.Count; i++)
            {
                stopIdToIndex[sortedStops[i].StopId] = i;
            }
        }

        public void Clear()
        {
            stopIdToIndex.Clear();
            travelTimeSeconds = new float[0, 0];
            stopCount = 0;
        }

        private static List<DRTStop> GetSortedStops(IReadOnlyList<DRTStop> stops)
        {
            var sortedStops = new List<DRTStop>();
            if (stops == null)
            {
                return sortedStops;
            }

            for (int i = 0; i < stops.Count; i++)
            {
                if (stops[i] != null)
                {
                    sortedStops.Add(stops[i]);
                }
            }

            sortedStops.Sort((a, b) => a.StopId.CompareTo(b.StopId));
            return sortedStops;
        }
    }
}
