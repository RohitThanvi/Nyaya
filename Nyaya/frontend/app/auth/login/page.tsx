'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Scale, Eye, EyeOff, Loader2 } from 'lucide-react'
import { authApi } from '@/lib/api'
import { useAuthStore } from '@/lib/store'
import { toast } from 'sonner'
import Link from 'next/link'

export default function LoginPage() {
  const router = useRouter()
  const { setUser } = useAuthStore()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPwd, setShowPwd] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [fullName, setFullName] = useState('')

  const handleSubmit = async () => {
    if (!email || !password || isLoading) return
    if (mode === 'register' && !fullName) { toast.error('Please enter your full name'); return }
    setIsLoading(true)
    try {
      if (mode === 'login') {
        await authApi.login(email, password)
      } else {
        await authApi.register({ email, password, full_name: fullName })
      }
      const user = await authApi.me()
      setUser(user)
      toast.success(`Welcome, ${user.full_name}`)
      router.push('/dashboard')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || `${mode === 'login' ? 'Login' : 'Registration'} failed`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-12 h-12 rounded-xl bg-primary/10 border border-primary/30 flex items-center justify-center mx-auto mb-4">
            <Scale className="w-6 h-6 text-primary" />
          </div>
          <h1 className="text-xl font-bold">NyayaAI</h1>
          <p className="text-sm text-muted-foreground mt-1">Indian Legal Research Platform</p>
        </div>

        <div className="bg-card border border-border rounded-2xl p-6 space-y-4">
          {/* Toggle */}
          <div className="flex rounded-lg bg-background p-0.5 border border-border">
            {(['login', 'register'] as const).map(m => (
              <button key={m} onClick={() => setMode(m)}
                className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-all capitalize ${
                  mode === m ? 'bg-card shadow text-foreground' : 'text-muted-foreground hover:text-foreground'
                }`}>
                {m === 'login' ? 'Sign In' : 'Register'}
              </button>
            ))}
          </div>

          {mode === 'register' && (
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Full Name</label>
              <input value={fullName} onChange={e => setFullName(e.target.value)}
                placeholder="Your full name"
                className="w-full bg-background border border-border rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground" />
            </div>
          )}

          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Email</label>
            <input value={email} onChange={e => setEmail(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
              type="email" placeholder="you@example.com"
              className="w-full bg-background border border-border rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground" />
          </div>

          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Password</label>
            <div className="relative">
              <input value={password} onChange={e => setPassword(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                type={showPwd ? 'text' : 'password'} placeholder="••••••••"
                className="w-full bg-background border border-border rounded-lg px-3 py-2.5 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground" />
              <button onClick={() => setShowPwd(!showPwd)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <button onClick={handleSubmit} disabled={isLoading || !email || !password}
            className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground py-2.5 rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </div>

        <p className="text-xs text-center text-muted-foreground mt-4">
          By continuing, you agree to use NyayaAI for research purposes only.
          <br />Not a substitute for professional legal advice.
        </p>
      </div>
    </div>
  )
}
