import type { Metadata } from 'next'
import { ThemeProvider } from 'next-themes'
import { Toaster } from 'sonner'
import './globals.css'

export const metadata: Metadata = {
  title: 'NyayaAI — Indian Legal Research Platform',
  description: 'AI-powered legal research for BNS, BNSS, BSA, and Supreme Court judgments',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              classNames: {
                toast: 'bg-zinc-900 border border-zinc-800 text-zinc-100',
                error: 'border-red-500/50',
                success: 'border-emerald-500/50',
              },
            }}
          />
        </ThemeProvider>
      </body>
    </html>
  )
}
