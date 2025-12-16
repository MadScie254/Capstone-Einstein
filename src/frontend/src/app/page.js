import HolographicCard from '@/components/HolographicCard'
import ModelComparisonChart from '@/components/ModelComparisonChart'
import LiveAnomalyScanner from '@/components/LiveAnomalyScanner'

// ... imports remain the same

export default function Home() {
  // ... existing state and logic

  return (
    <div className="min-h-screen p-6 lg:p-8 bg-dark-900 text-white selection:bg-neon-cyan/30">
      {/* Background Ambience */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-dark-800 via-dark-950 to-black -z-10" />
      <div className="fixed top-0 left-0 w-full h-[500px] bg-gradient-to-b from-neon-cyan/5 to-transparent -z-10 pointer-events-none" />

      {/* Header */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-6"
      >
        <div className="flex items-center gap-4">
          <div className="relative group">
            <div className="absolute inset-0 bg-neon-cyan blur-xl opacity-20 group-hover:opacity-40 transition-opacity" />
            <div className="relative p-3 rounded-2xl bg-dark-100 border border-white/10">
              <Zap className="w-8 h-8 text-neon-cyan" />
            </div>
          </div>
          <div>
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              Einstein
            </h1>
            <p className="text-neon-cyan font-mono text-xs tracking-widest uppercase">
              Theft Detection System v2.0
            </p>
          </div>
        </div>
        
        {/* API Status */}
        <div className="flex items-center gap-3 bg-dark-100/50 backdrop-blur px-4 py-2 rounded-full border border-white/5">
          <div className={`w-2 h-2 rounded-full shadow-[0_0_10px] ${
            apiStatus === 'connected' ? 'bg-neon-green shadow-neon-green' :
            apiStatus === 'checking' ? 'bg-neon-yellow shadow-neon-yellow animate-pulse' :
            'bg-neon-red shadow-neon-red'
          }`} />
          <span className="text-xs font-medium tracking-wide">
            {apiStatus === 'connected' ? 'SYSTEM ONLINE' :
             apiStatus === 'checking' ? 'CONNECTING...' :
             'DEMO MODE: ACTIVE'}
          </span>
        </div>
      </motion.header>
      
      {/* Command Center Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8">
        {/* KPI Cards (Top Row) */}
        <div className="lg:col-span-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <HolographicCard glowColor="cyan">
            <Users className="w-6 h-6 text-neon-cyan mb-2 opacity-80" />
            <div className="text-2xl font-bold">12,847</div>
            <div className="text-xs text-gray-400 font-mono">MONITORED METERS</div>
          </HolographicCard>
          
          <HolographicCard glowColor="red">
            <AlertTriangle className="w-6 h-6 text-neon-red mb-2 opacity-80" />
            <div className="text-2xl font-bold">127</div>
            <div className="text-xs text-gray-400 font-mono">ACTIVE ALERTS</div>
          </HolographicCard>
          
          <HolographicCard glowColor="purple">
            <TrendingUp className="w-6 h-6 text-neon-purple mb-2 opacity-80" />
            <div className="text-2xl font-bold">2.8%</div>
            <div className="text-xs text-gray-400 font-mono">THEFT RATE</div>
          </HolographicCard>
          
          <HolographicCard glowColor="green">
            <Activity className="w-6 h-6 text-neon-green mb-2 opacity-80" />
            <div className="text-2xl font-bold">$1.2M</div>
            <div className="text-xs text-gray-400 font-mono">REVENUE SAVED</div>
          </HolographicCard>

          {/* Model Arena (Mid Row) */}
          <div className="col-span-2 md:col-span-4 h-[300px] mt-4">
             <HolographicCard className="h-full flex flex-col" glowColor="purple">
               <div className="flex items-center justify-between mb-2">
                 <h3 className="font-semibold text-gray-200 flex items-center gap-2">
                   <Shield className="w-4 h-4 text-neon-purple" />
                   Model Arena: Champion vs Challenger
                 </h3>
                 <span className="text-xs bg-neon-purple/10 text-neon-purple px-2 py-1 rounded border border-neon-purple/20">
                   XGBoost Leading (+2.4%)
                 </span>
               </div>
               <div className="flex-1 w-full relative">
                 <ModelComparisonChart />
               </div>
             </HolographicCard>
          </div>
        </div>

        {/* Live Feed (Right Column) */}
        <div className="lg:col-span-4">
          <HolographicCard className="h-full" glowColor="green">
            <LiveAnomalyScanner />
          </HolographicCard>
        </div>
      </div>
      
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
