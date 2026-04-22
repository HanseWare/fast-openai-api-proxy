import { useAuthStore } from './stores/auth'

const API_BASE = '/api/admin' // Setup Vite proxy in vite.config.js later

export async function fetchApi(endpoint, options = {}) {
  const authStore = useAuthStore()
  
  const headers = {
    'Content-Type': 'application/json',
    ...authStore.authHeaders(),
    ...(options.headers || {})
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers
  })

  if (response.status === 401) {
    authStore.logout()
    // Let the component handle redirection or error
    throw new Error('Unauthorized')
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || 'API Error')
  }

  return response.json()
}
