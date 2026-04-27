import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('foap_token') || '')
  
  function hasCookieSession() {
    return document.cookie.includes('foap_logged_in=true') || document.cookie.includes('foap_admin_logged_in=true')
  }

  const isAuthenticated = computed(() => !!token.value || hasCookieSession())

  function setToken(newToken) {
    token.value = newToken
    localStorage.setItem('foap_token', newToken)
  }
  
  function logout() {
    token.value = ''
    localStorage.removeItem('foap_token')
    document.cookie = 'foap_logged_in=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;'
    document.cookie = 'foap_admin_logged_in=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;'
    document.cookie = 'foap_session=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;'
  }

  // Helper for fetch headers
  function authHeaders() {
    if (!token.value) return {}
    return {
      'Authorization': `Bearer ${token.value}`
    }
  }

  return { token, isAuthenticated, setToken, logout, authHeaders }
})
