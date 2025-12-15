'use client'

import { motion } from 'framer-motion'
import { Info, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react'

/**
 * ExplanationPanel Component
 * 
 * Displays SHAP-based feature explanations and human-readable
 * interpretation of the theft prediction.
 */
export default function ExplanationPanel({ explanation, probability }) {
  if (!explanation) return null
  
  const { top_features, explanation_text, consumption_stats } = explanation
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Info className="w-5 h-5 text-neon-pink" />
        Why This Prediction?
      </h3>
      
      {/* Explanation Text */}
      {explanation_text && (
        <div className="mb-6 p-4 bg-white/5 rounded-xl border-l-4 border-neon-cyan">
          <p className="text-gray-300">{explanation_text}</p>
        </div>
      )}
      
      {/* Feature Importance */}
      {top_features && top_features.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-400 mb-3">
            Top Contributing Features
          </h4>
          
          <div className="space-y-3">
            {top_features.slice(0, 5).map((feat, index) => {
              const isPositive = feat.importance > 0 || feat.direction === 'increases'
              const importance = Math.abs(feat.importance)
              const maxImportance = Math.max(...top_features.map(f => Math.abs(f.importance)))
              const barWidth = (importance / maxImportance) * 100
              
              return (
                <motion.div
                  key={feat.feature}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="relative"
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      {isPositive ? (
                        <TrendingUp className="w-4 h-4 text-red-400" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-green-400" />
                      )}
                      <span className="text-sm font-medium text-gray-300">
                        {formatFeatureName(feat.feature)}
                      </span>
                    </div>
                    <span className={`text-sm font-mono ${isPositive ? 'text-red-400' : 'text-green-400'}`}>
                      {isPositive ? '+' : '-'}{importance.toFixed(3)}
                    </span>
                  </div>
                  
                  {/* Importance bar */}
                  <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${barWidth}%` }}
                      transition={{ delay: index * 0.1, duration: 0.5 }}
                      className={`h-full rounded-full ${
                        isPositive 
                          ? 'bg-gradient-to-r from-red-500 to-orange-500' 
                          : 'bg-gradient-to-r from-green-500 to-emerald-500'
                      }`}
                    />
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      )}
      
      {/* Consumption Stats */}
      {consumption_stats && (
        <div className="border-t border-white/10 pt-4">
          <h4 className="text-sm font-medium text-gray-400 mb-3">
            Consumption Statistics
          </h4>
          
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center p-2 bg-white/5 rounded-lg">
              <p className="text-xs text-gray-400">Mean</p>
              <p className="text-lg font-semibold text-neon-cyan">
                {consumption_stats.mean?.toFixed(1) || 'N/A'}
              </p>
            </div>
            
            <div className="text-center p-2 bg-white/5 rounded-lg">
              <p className="text-xs text-gray-400">Std Dev</p>
              <p className="text-lg font-semibold text-gray-300">
                {consumption_stats.std?.toFixed(1) || 'N/A'}
              </p>
            </div>
            
            <div className="text-center p-2 bg-white/5 rounded-lg">
              <p className="text-xs text-gray-400">Zero Days</p>
              <p className={`text-lg font-semibold ${
                consumption_stats.zero_days > 0 ? 'text-red-400' : 'text-green-400'
              }`}>
                {consumption_stats.zero_days ?? 'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Alert for high probability */}
      {probability >= 0.7 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-start gap-3"
        >
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-red-400">Recommended Action</p>
            <p className="text-sm text-gray-300 mt-1">
              Schedule field inspection within 48 hours. Review meter installation 
              and check for tampering indicators.
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

/**
 * Format feature names for display
 */
function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase())
    .replace('Pct', '%')
    .replace('Cv', 'CV')
    .replace('Std', 'Std Dev')
}
