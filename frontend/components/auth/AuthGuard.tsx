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
  const [ready, setReady] = useState(false)

  useEffect(() => {
    // Cookies.get returns undefined if the cookie doesn't exist, or '' if it
    // exists but is empty. Both are falsy, but an empty string also means the
    // cookie was set incorrectly (e.g. from a failed secure-cookie write on
    // localhost HTTP). Treat either as unauthenticated.
    const token = Cookies.get('access_token')
    if (!token || token.trim() === '') {
      router.replace(redirectTo)
    } else {
      setReady(true)
    }
  }, [router, redirectTo])

  if (!ready) return null
  return <>{children}</>
}
