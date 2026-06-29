/**
 * Protected app layout.
 * All routes under /(app)/ require authentication.
 * AppLayout (sidebar + nav) is rendered once here — not duplicated per page.
 */
import { AuthGuard } from '@/components/auth/AuthGuard'
import AppLayout from '@/components/layout/AppLayout'

export default function ProtectedLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthGuard>
      <AppLayout>{children}</AppLayout>
    </AuthGuard>
  )
}
