<template>
  <div class="login-container">
    <div class="glass-panel login-box">
      <div class="logo-area">
        <div class="logo-circle"></div>
        <h2>FOAP Admin</h2>
        <p>Hanseatic Tradition. Modern Tech.</p>
        <span class="mode-chip" :class="`mode-chip--${authModeClass}`" style="margin-top: 1rem;">{{ authModeLabel }}</span>
      </div>

      <form @submit.prevent="handleLogin" class="login-form">
        <div class="input-group">
          <label for="token">{{ loginFieldLabel }}</label>
          <input 
            type="password" 
            id="token" 
            v-model="token" 
            :placeholder="loginPlaceholder"
            required
            autocomplete="current-password"
          />
        </div>
        
        <p class="auth-guidance" v-if="authModeHint">{{ authModeHint }}</p>

        <button type="submit" class="btn-primary" :disabled="loading || authMode === 'loading'">
          {{ loading ? 'Authenticating...' : 'Sign In' }}
        </button>

        <p v-if="error" class="error-msg">{{ error }}</p>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { fetchApi } from '../api'

const router = useRouter()
const authStore = useAuthStore()

const token = ref('')
const loading = ref(false)
const error = ref('')
const authConfig = ref(null)

const authMode = computed(() => authConfig.value?.admin?.mode || 'loading')

const authModeLabel = computed(() => {
  switch (authMode.value) {
    case 'oidc-only': return 'OIDC Admin Only'
    case 'hybrid': return 'OIDC + Token Admin'
    case 'token-hash-only': return 'Static Token Admin'
    default: return 'Loading Auth Mode...'
  }
})

const authModeClass = computed(() => {
  switch (authMode.value) {
    case 'oidc-only': return 'oidc'
    case 'hybrid': return 'hybrid'
    case 'token-hash-only': return 'token'
    default: return 'loading'
  }
})

const authModeHint = computed(() => {
  switch (authMode.value) {
    case 'oidc-only': return 'Admin portal requires an OIDC-issued access token with admin roles.'
    case 'hybrid': return 'You can use an OIDC access token or a static FOAP admin token.'
    case 'token-hash-only': return 'Use your static FOAP admin token to access the dashboard.'
    default: return 'Loading...'
  }
})

const loginFieldLabel = computed(() => {
  if (authMode.value === 'oidc-only') return 'OIDC Access Token'
  if (authMode.value === 'hybrid') return 'OIDC or Admin Token'
  return 'Admin Token'
})

const loginPlaceholder = computed(() => {
  if (authMode.value === 'oidc-only') return 'Paste your OIDC access token'
  if (authMode.value === 'hybrid') return 'Paste your OIDC or FOAP Admin token'
  return 'Enter your FOAP Admin Token'
})

async function loadAuthConfig() {
  try {
    authConfig.value = await fetchApi('/auth-config')
  } catch (err) {
    authConfig.value = { admin: { mode: 'loading' } }
  }
}

async function handleLogin() {
  loading.value = true
  error.value = ''
  
  // Temporarily set token to verify it
  authStore.setToken(token.value.trim())
  
  try {
    await fetchApi('/health')
    // If health passes, token is valid
    router.push({ name: 'dashboard' })
  } catch (err) {
    authStore.logout()
    error.value = 'Invalid or expired token. Or missing admin claims.'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadAuthConfig()
})
</script>

<style scoped>
.login-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 1rem;
}

.login-box {
  width: 100%;
  max-width: 420px;
  padding: 3rem 2rem;
  display: flex;
  flex-direction: column;
  gap: 2rem;
  animation: slideUp 0.5s ease-out forwards;
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.logo-area {
  text-align: center;
}

.logo-circle {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--color-berry-magenta), var(--color-teal-cyan));
  margin: 0 auto 1rem;
  box-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
}

.logo-area h2 {
  margin-bottom: 0.5rem;
  background: -webkit-linear-gradient(0deg, #fff, var(--color-text-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.logo-area p {
  margin: 0;
  color: var(--color-text-muted);
  font-size: 0.9rem;
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.input-group label {
  font-size: 0.9rem;
  color: var(--color-text-secondary);
  font-weight: 500;
}

.error-msg {
  color: var(--color-danger);
  font-size: 0.85rem;
  text-align: center;
  margin: 0;
}

.mode-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
  white-space: nowrap;
}

.mode-chip--oidc,
.mode-chip--hybrid {
  color: var(--color-teal-cyan);
  background: rgba(0, 229, 255, 0.08);
  border-color: rgba(0, 229, 255, 0.2);
}

.mode-chip--token,
.mode-chip--loading {
  color: var(--color-text-secondary);
  background: rgba(255, 255, 255, 0.04);
  border-color: rgba(255, 255, 255, 0.08);
}

.auth-guidance {
  margin: 0;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  text-align: center;
}
</style>
