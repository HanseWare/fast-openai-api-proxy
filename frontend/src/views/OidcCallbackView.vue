<template>
  <div class="callback-container">
    <div class="glass-panel callback-box">
      <div class="logo-circle"></div>
      <h2 v-if="!error">Completing sign-in…</h2>
      <h2 v-else class="error-title">Sign-in failed</h2>
      <p v-if="error" class="error-msg">{{ error }}</p>
      <p v-if="error" class="muted">
        <a href="#" @click.prevent="goBack">Return to login</a>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { fetchApi, fetchSelfServiceApi } from '../api'

const router = useRouter()
const authStore = useAuthStore()
const error = ref('')

function goBack() {
  const target = sessionStorage.getItem('foap_oidc_target') || 'admin'
  sessionStorage.removeItem('foap_oidc_target')
  if (target === 'account') {
    router.replace({ path: '/account' })
  } else {
    router.replace({ name: 'admin-login' })
  }
}

onMounted(async () => {
  const target = sessionStorage.getItem('foap_oidc_target') || 'admin'

  // In the BFF pattern, the backend handles the callback and redirects here.
  // We should already have a foap_session cookie if authentication succeeded.
  // Try to verify we have a valid session.

  try {
    if (target === 'account') {
      await fetchSelfServiceApi('/health')
      sessionStorage.removeItem('foap_oidc_target')
      router.replace({ path: '/account' })
    } else {
      await fetchApi('/health')
      sessionStorage.removeItem('foap_oidc_target')
      router.replace({ name: 'dashboard' })
    }
  } catch (err) {
    // The backend callback likely failed - show error
    error.value = 'Authentication failed. You may not have the required roles or permissions.'
    console.error('Session validation failed after OIDC callback:', err)
  }
})
</script>

<style scoped>
.callback-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 1rem;
}

.callback-box {
  width: 100%;
  max-width: 420px;
  padding: 3rem 2rem;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
  animation: slideUp 0.5s ease-out forwards;
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.logo-circle {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--color-berry-magenta), var(--color-teal-cyan));
  box-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { box-shadow: 0 0 20px rgba(0, 229, 255, 0.4); }
  50% { box-shadow: 0 0 40px rgba(0, 229, 255, 0.7); }
}

.callback-box h2 {
  background: -webkit-linear-gradient(0deg, #fff, var(--color-text-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.error-title {
  color: var(--color-danger);
}

.error-msg {
  color: var(--color-danger);
  font-size: 0.9rem;
}

.muted a {
  color: var(--color-teal-cyan);
  text-decoration: none;
}

.muted a:hover {
  text-decoration: underline;
}
</style>
