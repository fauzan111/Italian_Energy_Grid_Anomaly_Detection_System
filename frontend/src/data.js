export const demoSeries = [
  { label: "00:00", score: 0.12, load: 18.2, generation: 17.4 },
  { label: "02:00", score: 0.13, load: 17.8, generation: 17.2 },
  { label: "04:00", score: 0.16, load: 17.1, generation: 16.7 },
  { label: "06:00", score: 0.18, load: 18.0, generation: 17.3 },
  { label: "08:00", score: 0.22, load: 21.3, generation: 20.2 },
  { label: "10:00", score: 0.31, load: 24.7, generation: 23.4 },
  { label: "12:00", score: 0.41, load: 27.2, generation: 25.8 },
  { label: "14:00", score: 0.36, load: 26.8, generation: 25.3 },
  { label: "16:00", score: 0.24, load: 24.6, generation: 23.9 },
  { label: "18:00", score: 0.21, load: 23.4, generation: 22.7 },
  { label: "20:00", score: 0.17, load: 21.8, generation: 21.2 },
  { label: "22:00", score: 0.14, load: 19.9, generation: 19.3 },
];

export function buildPredictionRecords(points = demoSeries) {
  return points.map((point, index) => ({
    timestamp: `2025-01-01T${point.label}:00`,
    total_load_mw: Math.round(point.load * 1000),
    forecast_total_load_mw: Math.round(point.load * 990),
    actual_generation_mw: Math.round(point.generation * 1000),
    load_gap_mw: Math.round((point.load - point.generation) * 10),
    load_gap_pct: 0.05,
    hour: index * 2,
    day_of_week: 2,
    day_of_year: 1,
    month: 1,
    is_weekend: 0,
    hour_sin: Math.sin((index * 2 * Math.PI) / 24),
    hour_cos: Math.cos((index * 2 * Math.PI) / 24),
  }));
}
