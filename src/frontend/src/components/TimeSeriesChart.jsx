'use client'

import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart
} from 'recharts'

/**
 * TimeSeriesChart Component
 * 
 * Interactive consumption chart with anomaly highlighting.
 * Uses Recharts for smooth, responsive visualizations.
 */
export default function TimeSeriesChart({ data, anomalyThreshold }) {
  // Transform data for Recharts
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []
    
    const mean = data.reduce((a, b) => a + (b || 0), 0) / data.length
    const threshold = anomalyThreshold || mean * 0.5
    
    return data.map((value, index) => ({
      day: `Day ${index + 1}`,
      dayNum: index + 1,
      consumption: value || 0,
      isAnomaly: value < threshold,
      anomalyValue: value < threshold ? value : null,
      normalValue: value >= threshold ? value : null,
      threshold: threshold
    }))
  }, [data, anomalyThreshold])
  
  // Calculate statistics
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null
    
    const validData = data.filter(v => v != null && !isNaN(v))
    const mean = validData.reduce((a, b) => a + b, 0) / validData.length
    const min = Math.min(...validData)
    const max = Math.max(...validData)
    const anomalies = data.filter(v => v < (anomalyThreshold || mean * 0.5)).length
    
    return { mean, min, max, anomalies }
  }, [data, anomalyThreshold])
  
  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const value = payload[0].value
      const isAnomaly = value < (anomalyThreshold || stats?.mean * 0.5)
      
      return (
        <div className="glass-card p-3 border-neon-cyan/30">
          <p className="text-neon-cyan font-semibold">{label}</p>
          <p className={`text-lg font-bold ${isAnomaly ? 'text-red-400' : 'text-white'}`}>
            {value?.toFixed(1)} kWh
            {isAnomaly && <span className="ml-2 text-xs">⚠️ Low</span>}
          </p>
        </div>
      )
    }
    return null
  }
  
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No consumption data available
      </div>
    )
  }
  
  return (
    <div className="space-y-4">
      {/* Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="consumptionGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d9ff" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00d9ff" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="anomalyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ff4444" stopOpacity={0.5} />
                <stop offset="95%" stopColor="#ff4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            
            <XAxis 
              dataKey="day" 
              tick={{ fill: '#888', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.2)' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.2)' }}
            />
            
            <YAxis 
              tick={{ fill: '#888', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.2)' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.2)' }}
              label={{ 
                value: 'kWh', 
                angle: -90, 
                position: 'insideLeft',
                fill: '#888',
                fontSize: 12
              }}
            />
            
            <Tooltip content={<CustomTooltip />} />
            
            {/* Threshold line */}
            {stats && (
              <ReferenceLine 
                y={anomalyThreshold || stats.mean * 0.5} 
                stroke="#ff4444" 
                strokeDasharray="5 5"
                label={{ 
                  value: 'Anomaly Threshold', 
                  fill: '#ff4444', 
                  fontSize: 10,
                  position: 'right'
                }}
              />
            )}
            
            {/* Mean line */}
            {stats && (
              <ReferenceLine 
                y={stats.mean} 
                stroke="#00ff88" 
                strokeDasharray="3 3"
                strokeOpacity={0.5}
              />
            )}
            
            {/* Area fill */}
            <Area
              type="monotone"
              dataKey="consumption"
              stroke="none"
              fill="url(#consumptionGradient)"
            />
            
            {/* Main line */}
            <Line
              type="monotone"
              dataKey="consumption"
              stroke="#00d9ff"
              strokeWidth={2}
              dot={(props) => {
                const { cx, cy, payload } = props
                if (payload.isAnomaly) {
                  return (
                    <circle
                      key={`dot-${payload.dayNum}`}
                      cx={cx}
                      cy={cy}
                      r={6}
                      fill="#ff4444"
                      stroke="#fff"
                      strokeWidth={2}
                      className="animate-pulse"
                    />
                  )
                }
                return (
                  <circle
                    key={`dot-${payload.dayNum}`}
                    cx={cx}
                    cy={cy}
                    r={4}
                    fill="#00d9ff"
                    stroke="#fff"
                    strokeWidth={1}
                  />
                )
              }}
              activeDot={{ r: 8, fill: '#00d9ff', stroke: '#fff', strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      {/* Statistics Bar */}
      {stats && (
        <div className="grid grid-cols-4 gap-2 text-center">
          <div className="p-2 bg-white/5 rounded-lg">
            <p className="text-xs text-gray-400">Mean</p>
            <p className="text-sm font-semibold text-neon-cyan">
              {stats.mean.toFixed(1)} kWh
            </p>
          </div>
          <div className="p-2 bg-white/5 rounded-lg">
            <p className="text-xs text-gray-400">Min</p>
            <p className="text-sm font-semibold text-gray-300">
              {stats.min.toFixed(1)} kWh
            </p>
          </div>
          <div className="p-2 bg-white/5 rounded-lg">
            <p className="text-xs text-gray-400">Max</p>
            <p className="text-sm font-semibold text-gray-300">
              {stats.max.toFixed(1)} kWh
            </p>
          </div>
          <div className="p-2 bg-white/5 rounded-lg">
            <p className="text-xs text-gray-400">Anomalies</p>
            <p className={`text-sm font-semibold ${stats.anomalies > 0 ? 'text-red-400' : 'text-green-400'}`}>
              {stats.anomalies} days
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
