import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import ErrorBoundary from './utils/ErrorBoundary'

const rootEl = document.getElementById('root')
if (!rootEl) {
  const msg = document.createElement('div')
  msg.style.padding = '20px'
  msg.style.color = '#fff'
  msg.style.background = '#b91c1c'
  msg.textContent = 'Root element not found. Check index.html contains <div id="root"></div>.'
  document.body.appendChild(msg)
  console.error('Root element "root" not found in document')
} else {
  createRoot(rootEl).render(
    <StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </StrictMode>
  )
}
