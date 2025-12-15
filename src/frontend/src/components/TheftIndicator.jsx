'use client'

import { motion } from 'framer-motion'
import { AlertTriangle, ShieldCheck, ShieldAlert } from 'lucide-react'

/**
 * TheftIndicator Component
 * 
 * Visual theft probability indicator with animated effects.
 * Shows a pulsing red indicator for high-risk predictions.
 */
export default function TheftIndicator({ probability, riskLevel }) {
  // Determine colors and icons based on risk
  const getRiskConfig = () => {
    if (riskLevel === 'HIGH' || probability >= 0.8) {
      return {
        bgColor: 'from-red-500/20 to-red-600/20',
        borderColor: 'border-red-500/50',
        textColor: 'text-red-400',
        glowColor: 'shadow-glow-red',
        icon: ShieldAlert,
        label: 'HIGH RISK',
        description: 'Immediate investigation recommended',
        animate: true
      }
    } else if (riskLevel === 'MEDIUM' || probability >= 0.5) {
      return {
        bgColor: 'from-orange-500/20 to-yellow-500/20',
        borderColor: 'border-orange-500/50',
        textColor: 'text-orange-400',
        glowColor: '',
        icon: AlertTriangle,
        label: 'MEDIUM RISK',
        description: 'Further review recommended',
        animate: false
      }
    } else {
      return {
        bgColor: 'from-green-500/20 to-emerald-500/20',
        borderColor: 'border-green-500/50',
        textColor: 'text-green-400',
        glowColor: 'shadow-glow-green',
        icon: ShieldCheck,
        label: 'LOW RISK',
        description: 'No immediate action required',
        animate: false
      }
    }
  }
  
  const config = getRiskConfig()
  const Icon = config.icon
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`
        glass-card p-6 
        bg-gradient-to-br ${config.bgColor}
        border ${config.borderColor}
        ${config.glowColor}
        ${config.animate ? 'animate-pulse-glow' : ''}
      `}
    >
      <div className="flex items-center justify-between">
        {/* Left side - Icon and Label */}
        <div className="flex items-center gap-4">
          <motion.div
            animate={config.animate ? {
              scale: [1, 1.1, 1],
            } : {}}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
            className={`
              p-3 rounded-2xl 
              ${riskLevel === 'HIGH' ? 'bg-red-500/30' : 
                riskLevel === 'MEDIUM' ? 'bg-orange-500/30' : 'bg-green-500/30'}
            `}
          >
            <Icon className={`w-8 h-8 ${config.textColor}`} />
          </motion.div>
          
          <div>
            <p className="text-sm text-gray-400">Theft Probability</p>
            <div className="flex items-center gap-3">
              <span className={`text-3xl font-bold ${config.textColor}`}>
                {(probability * 100).toFixed(1)}%
              </span>
              <span className={`
                px-3 py-1 rounded-full text-sm font-semibold
                ${riskLevel === 'HIGH' ? 'bg-red-500 text-white' :
                  riskLevel === 'MEDIUM' ? 'bg-orange-500 text-white' : 
                  'bg-green-500 text-white'}
              `}>
                {config.label}
              </span>
            </div>
            <p className="text-sm text-gray-400 mt-1">{config.description}</p>
          </div>
        </div>
        
        {/* Right side - Probability Gauge */}
        <div className="hidden md:block">
          <ProbabilityGauge probability={probability} riskLevel={riskLevel} />
        </div>
      </div>
      
      {/* Alert banner for HIGH risk */}
      {riskLevel === 'HIGH' && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ delay: 0.3 }}
          className="mt-4 p-3 bg-red-500/20 rounded-xl border border-red-500/30"
        >
          <p className="text-sm text-red-300 flex items-center gap-2">
            <span className="animate-pulse">🚨</span>
            <strong>Alert:</strong> This meter shows strong indicators of potential theft. 
            Immediate field inspection is recommended.
          </p>
        </motion.div>
      )}
    </motion.div>
  )
}

/**
 * Circular probability gauge
 */
function ProbabilityGauge({ probability, riskLevel }) {
  const radius = 40
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference * (1 - probability)
  
  const getColor = () => {
    if (riskLevel === 'HIGH') return '#ff4444'
    if (riskLevel === 'MEDIUM') return '#ff8800'
    return '#00cc66'
  }
  
  return (
    <div className="relative w-24 h-24">
      <svg className="w-full h-full transform -rotate-90">
        {/* Background circle */}
        <circle
          cx="48"
          cy="48"
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="8"
        />
        
        {/* Progress circle */}
        <motion.circle
          cx="48"
          cy="48"
          r={radius}
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: 'easeOut' }}
          style={{
            filter: `drop-shadow(0 0 10px ${getColor()})`
          }}
        />
      </svg>
      
      {/* Center text */}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold" style={{ color: getColor() }}>
          {(probability * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}
