<template>
  <div class="callback-container">
    <div class="glass-panel callback-box">
      <div class="logo-circle"></div>
      <h2 v-if="!error">Completing sign-in…</h2>
      <h2 v-else>Sign-in failed</h2>
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
import { completeOidcLogin } from '../services/oidc'
import { fetchApi, fetchSelfServiceApi } from '../api'

const router = useRouter()
const authStore = useAuthStore()
const error = ref('')

function goBack() {
  const target = sessionStorage.getItem('foap_oidc_target') || 'admin'
  sessionStorage.removeItem('foap_oidc_target')
  if (target === 'account') {
    router.replace({ path: '/' })
  } else {
    router.replace({ name: 'admin-login' })
  }
}

onMounted(async () => {
  const target = sessionStorage.getItem('foap_oidc_target') || 'admin'

  // Fetch the OIDC client config so we can complete the flow
  let oidcClient = null
  try {
    const config = target === 'account'
      ? await fetchSelfServiceApi('/auth-config')
      : await fetchApi('/auth-config')
    oidcClient = config?.oidc_client
  } catch (err) {
    // Fallback: try the other endpoint
    try {
      const config = await fetchApi('/auth-config')
      oidcClient = config?.oidc_client
    } catch (err2) {
      error.value = 'Unable to load auth configuration.'
      return
    }
  }

  if (!oidcClient) {
    error.value = 'OIDC is not configured on this server.'
    return
  }

  const accessToken = await completeOidcLogin(oidcClient)
  if (!accessToken) {
    error.value = 'Failed to extract access token from identity provider response.'
    return
  }

  // Save token and verify it works
  authStore.setToken(accessToken)

  try {
    if (target === 'account') {
      await fetchSelfServiceApi('/health')
      sessionStorage.removeItem('foap_oidc_target')
      router.replace({ path: '/' })
    } else {
      await fetchApi('/health')
      sessionStorage.removeItem('foap_oidc_target')
      router.replace({ name: 'dashboard' })
    }
  } catch (err) {
    authStore.logout()
    error.value = 'Token is valid but you do not have the required roles/groups for this portal.'
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
