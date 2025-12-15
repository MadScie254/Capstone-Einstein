import './globals.css'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: '⚡ Einstein - Electricity Theft Detection',
  description: 'AI-Powered Electricity Theft Detection System with Real-time Scoring and Explainability',
  keywords: ['electricity', 'theft detection', 'machine learning', 'smart meter', 'anomaly detection'],
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.ico" />
        <meta name="theme-color" content="#1a1a2e" />
      </head>
      <body className={`${inter.className} antialiased`}>
        {/* Background gradient */}
        <div className="fixed inset-0 bg-dark-gradient -z-10" />
        
        {/* Animated background particles (optional) */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none -z-5">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-neon-cyan/5 rounded-full blur-3xl animate-float" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-neon-pink/5 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }} />
        </div>
        
        {/* Main content */}
        <main className="min-h-screen">
          {children}
        </main>
      </body>
    </html>
  )
}
