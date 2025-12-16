
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ShieldAlert, CheckCircle2 } from 'lucide-react'

export default function LiveAnomalyScanner() {
  const [logs, setLogs] = useState([])

  // Simulate live incoming data
  useEffect(() => {
    const interval = setInterval(() => {
      const newLog = generateRandomLog()
      setLogs(prev => [newLog, ...prev].slice(0, 7)) // Keep last 7 logs
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  const generateRandomLog = () => {
    const isAnomaly = Math.random() > 0.8
    return {
      id: Date.now(),
      meterId: `M-${Math.floor(Math.random() * 90000) + 10000}`,
      timestamp: new Date().toLocaleTimeString(),
      reading: (Math.random() * 50).toFixed(1),
      status: isAnomaly ? 'SUSPICIOUS' : 'NORMAL'
    }
  }

  return (
    <div className="h-full flex flex-col">
      <h3 className="text-sm font-mono text-gray-400 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 bg-neon-green rounded-full animate-pulse" />
        LIVE_METER_STREAM
      </h3>
      
      <div className="flex-1 overflow-hidden space-y-2 font-mono text-xs">
        <AnimatePresence initial={false}>
          {logs.map((log) => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: -20, height: 0 }}
              animate={{ opacity: 1, x: 0, height: 'auto' }}
              exit={{ opacity: 0 }}
              className={`
                flex items-center justify-between p-2 rounded border-l-2
                ${log.status === 'SUSPICIOUS' 
                  ? 'bg-red-500/10 border-red-500 text-red-200' 
                  : 'bg-green-500/5 border-green-500/50 text-green-200'}
              `}
            >
              <div className="flex items-center gap-3">
                <span>{log.timestamp}</span>
                <span className="opacity-70">{log.meterId}</span>
              </div>
              <div className="flex items-center gap-2">
                <span>{log.reading} kWh</span>
                {log.status === 'SUSPICIOUS' ? (
                  <ShieldAlert className="w-3 h-3 text-red-500" />
                ) : (
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  )
}
