'use client'

import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileText, X, Loader2 } from 'lucide-react'

/**
 * UploadMeter Component
 * 
 * Drag & drop file upload for consumption data CSV files,
 * with manual input option and demo data.
 */
export default function UploadMeter({ onScore, loading }) {
  const [dragActive, setDragActive] = useState(false)
  const [file, setFile] = useState(null)
  const [manualInput, setManualInput] = useState('')
  const [inputMode, setInputMode] = useState('upload') // 'upload' | 'manual' | 'demo'
  const [customerId, setCustomerId] = useState('')
  
  // Demo consumption data (26 days with some anomalies)
  const demoData = [
    100.5, 98.2, 105.3, 95.7, 92.1, 88.5, 110.2,
    107.8, 45.2, 48.3, 95.0, 97.8, 102.1, 99.5,
    101.2, 15.0, 8.5, 5.2, 95.7, 94.2, 96.5, 100.1,
    105.5, 99.8, 97.2, 101.1
  ]
  
  // Handle drag events
  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])
  
  // Handle file drop
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])
  
  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }
  
  // Process uploaded file
  const handleFile = async (uploadedFile) => {
    if (!uploadedFile.name.endsWith('.csv')) {
      alert('Please upload a CSV file')
      return
    }
    
    setFile(uploadedFile)
    
    // Read file content
    const text = await uploadedFile.text()
    const lines = text.split('\n')
    
    if (lines.length >= 2) {
      // Assume first row is header, second row is data
      const values = lines[1].split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
      if (values.length > 0) {
        setManualInput(values.join(', '))
      }
    }
  }
  
  // Handle score submission
  const handleSubmit = () => {
    let consumption = []
    
    if (inputMode === 'demo') {
      consumption = demoData
    } else if (manualInput) {
      try {
        consumption = manualInput.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
      } catch {
        alert('Invalid input format')
        return
      }
    }
    
    if (consumption.length < 3) {
      alert('Please provide at least 3 days of consumption data')
      return
    }
    
    onScore(consumption, customerId || 'DASHBOARD_USER')
  }
  
  // Clear all inputs
  const handleClear = () => {
    setFile(null)
    setManualInput('')
    setCustomerId('')
  }
  
  return (
    <div className="glass-card p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <Upload className="w-5 h-5 text-neon-cyan" />
        Upload Consumption Data
      </h2>
      
      {/* Input Mode Tabs */}
      <div className="flex gap-2 mb-4">
        {['upload', 'manual', 'demo'].map((mode) => (
          <button
            key={mode}
            onClick={() => setInputMode(mode)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              inputMode === mode
                ? 'bg-neon-cyan/20 text-neon-cyan border border-neon-cyan/30'
                : 'bg-white/5 text-gray-400 hover:bg-white/10'
            }`}
          >
            {mode === 'upload' ? '📁 Upload CSV' : mode === 'manual' ? '✏️ Manual' : '🎲 Demo Data'}
          </button>
        ))}
      </div>
      
      {/* Upload Mode */}
      {inputMode === 'upload' && (
        <div
          className={`dropzone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {file ? (
            <div className="flex items-center gap-3">
              <FileText className="w-8 h-8 text-neon-green" />
              <div>
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-gray-400">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); handleClear() }}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="w-12 h-12 text-gray-500" />
              <div className="text-center">
                <p className="font-medium">Drag & drop your CSV file here</p>
                <p className="text-sm text-gray-400 mt-1">or click to browse</p>
              </div>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
            </>
          )}
        </div>
      )}
      
      {/* Manual Mode */}
      {inputMode === 'manual' && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Daily Consumption Values (comma-separated)
            </label>
            <textarea
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="100.5, 98.2, 105.3, 95.7, ..."
              className="input-glass h-32 resize-none"
            />
          </div>
        </div>
      )}
      
      {/* Demo Mode */}
      {inputMode === 'demo' && (
        <div className="p-4 bg-neon-cyan/5 rounded-xl border border-neon-cyan/20">
          <p className="text-sm text-gray-300 mb-2">
            🎲 Demo data with suspicious patterns:
          </p>
          <p className="text-xs text-gray-400 font-mono">
            {demoData.slice(0, 10).join(', ')}... ({demoData.length} days)
          </p>
          <p className="text-xs text-yellow-400 mt-2">
            ⚠️ Includes sudden drops (days 9-10, 16-18) to simulate theft behavior
          </p>
        </div>
      )}
      
      {/* Customer ID Input */}
      <div className="mt-4">
        <label className="block text-sm text-gray-400 mb-2">
          Customer ID (optional)
        </label>
        <input
          type="text"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          placeholder="CUST_001"
          className="input-glass"
        />
      </div>
      
      {/* Action Buttons */}
      <div className="flex gap-3 mt-6">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleSubmit}
          disabled={loading}
          className="glow-button flex-1 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              🔮 Detect Theft
            </>
          )}
        </motion.button>
        
        {(file || manualInput) && (
          <button
            onClick={handleClear}
            className="px-4 py-3 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
          >
            Clear
          </button>
        )}
      </div>
      
      {/* Format hint */}
      <p className="text-xs text-gray-500 mt-4">
        📋 CSV format: Daily consumption values in kWh. Multiple columns supported.
      </p>
    </div>
  )
}
