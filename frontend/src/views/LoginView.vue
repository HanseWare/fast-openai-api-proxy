<template>
  <div class="login-container">
    <div class="glass-panel login-box">
      <div class="logo-area">
        <!-- Placeholder for Logo -->
        <div class="logo-circle"></div>
        <h2>FOAP Admin</h2>
        <p>Hanseatic Tradition. Modern Tech.</p>
      </div>

      <form @submit.prevent="handleLogin" class="login-form">
        <div class="input-group">
          <label for="token">Admin Token</label>
          <input 
            type="password" 
            id="token" 
            v-model="token" 
            placeholder="Enter your FOAP Admin Token"
            required
            autocomplete="current-password"
          />
        </div>

        <button type="submit" class="btn-primary" :disabled="loading">
          {{ loading ? 'Authenticating...' : 'Sign In' }}
        </button>

        <p v-if="error" class="error-msg">{{ error }}</p>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { fetchApi } from '../api'

const router = useRouter()
const authStore = useAuthStore()

const token = ref('')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  loading.value = true
  error.value = ''
  
  // Temporarily set token to verify it
  authStore.setToken(token.value)
  
  try {
    await fetchApi('/health')
    // If health passes, token is valid
    router.push({ name: 'dashboard' })
  } catch (err) {
    authStore.logout()
    error.value = 'Invalid or expired token.'
  } finally {
    loading.value = false
  }
}
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
</style>
