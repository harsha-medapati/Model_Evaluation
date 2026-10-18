/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        studio: {
          dark: '#0f172a',
          sidebar: '#111827',
          card: '#ffffff',
          accent: '#2563eb',
          accentHover: '#1d4ed8',
          border: '#e2e8f0',
          muted: '#64748b',
          success: '#10b981',
          danger: '#ef4444',
          warning: '#f59e0b'
        }
      }
    },
  },
  plugins: [],
}
