/**
 * AuthGuard — route-level authentication wrapper.
 *
 * Wraps any page component. Redirects to /auth/login if no access_token cookie.
 * Renders nothing (null) while checking, avoiding flash of protected content.
 *
 * Usage:
 *   export default function ChatPage() {
 *     return <AuthGuard><ChatPageContent /></AuthGuard>
 *   }
 */
'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Cookies from 'js-cookie'

interface AuthGuardProps {
  children: React.ReactNode
  redirectTo?: string
}

export function AuthGuard({ children, redirectTo = '/auth/login' }: AuthGuardProps) {
  const router = useRouter()
  const [checked, setChecked] = useState(false)
  const [authenticated, setAuthenticated] = useState(false)

  useEffect(() => {
    const token = Cookies.get('access_token')
    if (!token) {
      router.replace(redirectTo)
    } else {
      setAuthenticated(true)
    }
    setChecked(true)
  }, [router, redirectTo])

  if (!checked) return null
  if (!authenticated) return null
  return <>{children}</>
}
