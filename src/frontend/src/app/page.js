'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Zap, 
  Upload, 
  BarChart3, 
  Shield, 
  AlertTriangle,
  TrendingUp,
  Users,
  Activity
} from 'lucide-react'
import UploadMeter from '@/components/UploadMeter'
import TimeSeriesChart from '@/components/TimeSeriesChart'
import ExplanationPanel from '@/components/ExplanationPanel'
import TheftIndicator from '@/components/TheftIndicator'

// API base URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Home() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  
  // Check API health on mount
  useEffect(() => {
    checkApiHealth()
  }, [])
  
  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`)
      if (response.ok) {
        setApiStatus('connected')
      } else {
        setApiStatus('error')
      }
    } catch {
      setApiStatus('offline')
    }
  }
  
  // Handle scoring
  const handleScore = async (consumption, customerId) => {
    setLoading(true)
    setError(null)
    
    try {
      // Score endpoint
      const scoreResponse = await fetch(`${API_URL}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          consumption: consumption,
          customer_id: customerId || 'DASHBOARD_USER'
        })
      })
      
      if (!scoreResponse.ok) {
        throw new Error('Scoring failed')
      }
      
      const scoreData = await scoreResponse.json()
      
      // Explanation endpoint
      const explainResponse = await fetch(`${API_URL}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          consumption: consumption,
          customer_id: customerId || 'DASHBOARD_USER'
        })
      })
      
      let explainData = null
      if (explainResponse.ok) {
        explainData = await explainResponse.json()
      }
      
      setResults({
        ...scoreData,
        consumption: consumption,
        explanation: explainData
      })
      
    } catch (err) {
      setError(err.message)
      // Use mock data for demo
      setResults({
        probability: 0.73,
        risk_level: 'HIGH',
        customer_id: 'DEMO_USER',
        consumption: consumption,
        xgb_score: 0.78,
        isolation_score: 0.65,
        explanation: {
          top_features: [
            { feature: 'sudden_drop_count', importance: 0.35, direction: 'increases' },
            { feature: 'zero_ratio', importance: 0.28, direction: 'increases' },
            { feature: 'consumption_cv', importance: 0.18, direction: 'increases' },
          ],
          explanation_text: 'High number of sudden consumption drops detected. Multiple zero-reading days indicate possible meter tampering.',
          consumption_stats: {
            mean: 85.5,
            std: 45.2,
            zero_days: 3
          }
        }
      })
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="min-h-screen p-6 lg:p-8">
      {/* Header */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-2xl bg-gradient-to-br from-neon-cyan to-neon-green">
              <Zap className="w-8 h-8 text-dark-300" />
            </div>
            <div>
              <h1 className="text-3xl font-bold neon-text">Einstein</h1>
              <p className="text-gray-400">Electricity Theft Detection System</p>
            </div>
          </div>
          
          {/* API Status */}
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              apiStatus === 'connected' ? 'bg-green-500' :
              apiStatus === 'checking' ? 'bg-yellow-500 animate-pulse' :
              'bg-red-500'
            }`} />
            <span className="text-sm text-gray-400">
              {apiStatus === 'connected' ? 'API Connected' :
               apiStatus === 'checking' ? 'Connecting...' :
               'Demo Mode'}
            </span>
          </div>
        </div>
      </motion.header>
      
      {/* KPI Cards */}
      <motion.section 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
      >
        <div className="metric-card">
          <Users className="w-8 h-8 text-neon-cyan mb-2" />
          <span className="metric-value">12,847</span>
          <span className="metric-label">Monitored Meters</span>
        </div>
        
        <div className="metric-card">
          <AlertTriangle className="w-8 h-8 text-neon-red mb-2" />
          <span className="metric-value">127</span>
          <span className="metric-label">Active Alarms</span>
        </div>
        
        <div className="metric-card">
          <TrendingUp className="w-8 h-8 text-neon-green mb-2" />
          <span className="metric-value">2.8%</span>
          <span className="metric-label">Theft Rate</span>
        </div>
        
        <div className="metric-card">
          <Activity className="w-8 h-8 text-neon-pink mb-2" />
          <span className="metric-value">$1.2M</span>
          <span className="metric-label">Revenue Protected</span>
        </div>
      </motion.section>
      
      {/* Main Content */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left Panel - Upload */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <UploadMeter onScore={handleScore} loading={loading} />
          
          {/* Error display */}
          {error && apiStatus !== 'connected' && (
            <div className="mt-4 p-4 glass-card border-yellow-500/30">
              <p className="text-yellow-400 text-sm">
                ⚠️ API not available. Showing demo results.
              </p>
            </div>
          )}
        </motion.div>
        
        {/* Right Panel - Results */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-6"
        >
          {results ? (
            <>
              {/* Theft Indicator */}
              <TheftIndicator 
                probability={results.probability} 
                riskLevel={results.risk_level}
              />
              
              {/* Time Series Chart */}
              <div className="glass-card p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-neon-cyan" />
                  Consumption Pattern
                </h3>
                <TimeSeriesChart 
                  data={results.consumption}
                  anomalyThreshold={results.explanation?.consumption_stats?.mean * 0.5}
                />
              </div>
              
              {/* Explanation Panel */}
              {results.explanation && (
                <ExplanationPanel 
                  explanation={results.explanation}
                  probability={results.probability}
                />
              )}
            </>
          ) : (
            <div className="glass-card p-12 text-center">
              <Shield className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-400 mb-2">
                Ready to Analyze
              </h3>
              <p className="text-gray-500">
                Upload consumption data to detect potential electricity theft
              </p>
            </div>
          )}
        </motion.div>
      </div>
      
      {/* Footer */}
      <footer className="mt-12 text-center text-gray-500 text-sm">
        <p>⚡ Einstein Theft Detection System v1.0.0</p>
        <p className="mt-1">Built with Next.js, Tailwind CSS, and FastAPI</p>
      </footer>
    </div>
  )
}
