import { useAuthStore } from './stores/auth'

const ADMIN_API_BASE = '/api/admin'
const SELF_SERVICE_API_BASE = '/api'

async function requestApi(basePath, endpoint, options = {}) {
  const authStore = useAuthStore()

  const headers = {
    'Content-Type': 'application/json',
    ...authStore.authHeaders(),
    ...(options.headers || {})
  }

  const response = await fetch(`${basePath}${endpoint}`, {
    ...options,
    headers
  })

  if (response.status === 401) {
    authStore.logout()
    throw new Error('Unauthorized')
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || 'API Error')
  }

  return response.json()
}

export async function fetchApi(endpoint, options = {}) {
  return requestApi(ADMIN_API_BASE, endpoint, options)
}

export async function fetchSelfServiceApi(endpoint, options = {}) {
  return requestApi(SELF_SERVICE_API_BASE, endpoint, options)
}

