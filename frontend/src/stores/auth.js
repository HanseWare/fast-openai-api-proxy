import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('foap_token') || '')
  
  const isAuthenticated = computed(() => !!token.value)
  
  function setToken(newToken) {
    token.value = newToken
    localStorage.setItem('foap_token', newToken)
  }
  
  function logout() {
    token.value = ''
    localStorage.removeItem('foap_token')
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
