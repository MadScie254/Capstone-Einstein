
import { motion } from 'framer-motion'

export default function HolographicCard({ children, className = '', glowColor = 'cyan' }) {
  // Map colors to tailwind classes
  const glowMap = {
    cyan: 'shadow-neon-cyan/20 border-neon-cyan/30',
    purple: 'shadow-neon-purple/20 border-neon-purple/30',
    red: 'shadow-neon-red/20 border-neon-red/30',
    green: 'shadow-neon-green/20 border-neon-green/30',
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className={`
        relative overflow-hidden backdrop-blur-xl bg-dark-200/40 
        border border-white/10 rounded-xl p-6
        hover:shadow-lg transition-all duration-300
        ${glowMap[glowColor] || glowMap.cyan}
        ${className}
      `}
    >
      {/* Glossy overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
      
      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  )
}
