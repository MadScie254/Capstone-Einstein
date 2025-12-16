
'use client'
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, Tooltip } from 'recharts'

export default function ModelComparisonChart() {
  const data = [
    { subject: 'Precision', XGBoost: 0.85, IsolationForest: 0.65, fullMark: 1 },
    { subject: 'Recall', XGBoost: 0.72, IsolationForest: 0.88, fullMark: 1 },
    { subject: 'F1 Score', XGBoost: 0.78, IsolationForest: 0.75, fullMark: 1 },
    { subject: 'Inference Speed', XGBoost: 0.95, IsolationForest: 0.90, fullMark: 1 },
    { subject: 'Interpretability', XGBoost: 0.80, IsolationForest: 0.40, fullMark: 1 },
    { subject: 'Robustness', XGBoost: 0.88, IsolationForest: 0.82, fullMark: 1 },
  ]

  return (
    <div className="h-full w-full min-h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
          
          <Radar
            name="XGBoost (Champion)"
            dataKey="XGBoost"
            stroke="#0ea5e9"
            strokeWidth={2}
            fill="#0ea5e9"
            fillOpacity={0.3}
          />
          <Radar
            name="Isolation Forest"
            dataKey="IsolationForest"
            stroke="#a855f7"
            strokeWidth={2}
            fill="#a855f7"
            fillOpacity={0.3}
          />
          
          <Legend wrapperStyle={{ color: '#fff' }} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
            itemStyle={{ color: '#e2e8f0' }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
