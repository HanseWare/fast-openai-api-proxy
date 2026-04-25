<template>
  <div class="account-page">
    <section class="hero glass-panel">
      <div>
        <p class="eyebrow">/account</p>
        <h1>Self-Service Portal</h1>
        <p class="muted">
          Generate, review, and revoke your own FOAP API keys. The same bearer token is used for both
          static-token and OIDC-backed self-service modes.
        </p>
      </div>
      <div class="hero-stats" v-if="isAuthenticated">
        <div class="stat-card">
          <span class="stat-value">{{ keys.length }}</span>
          <span class="stat-label">Active Keys</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{{ quotaSummary.configured }}</span>
          <span class="stat-label">With Quota</span>
        </div>
      </div>
    </section>

    <section v-if="!isAuthenticated" class="glass-panel login-card">
      <h2>Sign in to your account</h2>
      <p class="muted">
        Use your self-service bearer token or OIDC access token to access this portal.
      </p>
      <form class="login-form" @submit.prevent="handleLogin">
        <div class="input-group">
          <label for="account-token">Bearer Token</label>
          <input
            id="account-token"
            v-model="loginToken"
            type="password"
            placeholder="Paste your token"
            autocomplete="current-password"
            required
          />
        </div>
        <button class="btn-primary" type="submit" :disabled="loadingAuth">
          {{ loadingAuth ? 'Verifying…' : 'Open Account Portal' }}
        </button>
        <p v-if="loginError" class="error-msg">{{ loginError }}</p>
      </form>
    </section>

    <template v-else>
      <section class="toolbar glass-panel">
        <div>
          <h2>Identity</h2>
          <p class="muted">Authenticated as <code>{{ authStore.token ? 'Bearer token present' : 'unknown' }}</code>.</p>
        </div>
        <div class="toolbar-actions">
          <button class="btn-secondary" type="button" @click="refreshAll" :disabled="loading">
            Refresh
          </button>
          <button class="btn-secondary danger" type="button" @click="handleLogout">
            Logout
          </button>
        </div>
      </section>

      <section class="grid-two">
        <div class="glass-panel panel">
          <h2>Create API key</h2>
          <form class="stacked-form" @submit.prevent="createKey">
            <div class="input-group">
              <label for="key-name">Key Name</label>
              <input id="key-name" v-model="newKeyName" type="text" placeholder="e.g. laptop" required />
            </div>
            <button class="btn-primary" type="submit" :disabled="loading">
              {{ loading ? 'Working…' : 'Create Key' }}
            </button>
          </form>
        </div>

        <div class="glass-panel panel">
          <h2>Quota dashboard</h2>
          <p class="muted" v-if="!keys.length">Create a key to see minute quota and usage details here.</p>
          <div v-else class="quota-list">
            <article v-for="item in keyDetails" :key="item.key.id" class="quota-card">
              <div class="quota-card-head">
                <div>
                  <h3>{{ item.key.name }}</h3>
                  <p class="muted">{{ item.key.id }}</p>
                </div>
                <button class="btn-secondary danger small" type="button" @click="deleteKey(item.key.id)">
                  Revoke
                </button>
              </div>

              <div v-if="item.quota" class="quota-metrics">
                <div>
                  <span class="metric-label">Requests/minute</span>
                  <span class="metric-value">{{ item.quota.requests_per_minute }}</span>
                </div>
                <div>
                  <span class="metric-label">Used</span>
                  <span class="metric-value">{{ item.usage?.used ?? '—' }}</span>
                </div>
                <div>
                  <span class="metric-label">Remaining</span>
                  <span class="metric-value">{{ item.usage?.remaining ?? '—' }}</span>
                </div>
                <div>
                  <span class="metric-label">Reset In</span>
                  <span class="metric-value">{{ formatSeconds(item.usage?.reset_in_seconds) }}</span>
                </div>
              </div>
              <p v-else class="muted">No quota configured yet.</p>
            </article>
          </div>
        </div>
      </section>

      <section class="glass-panel panel">
        <h2>Your API keys</h2>
        <p v-if="!keys.length && !loading" class="muted">No keys yet.</p>
        <div class="key-list" v-else>
          <article v-for="key in keys" :key="key.id" class="key-row">
            <div>
              <h3>{{ key.name }}</h3>
              <p class="muted">{{ key.id }}</p>
            </div>
            <button class="btn-secondary danger small" type="button" @click="deleteKey(key.id)">
              Delete
            </button>
          </article>
        </div>
      </section>

      <p v-if="error" class="error-banner glass-panel">{{ error }}</p>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { fetchSelfServiceApi } from '../api'

const router = useRouter()
const authStore = useAuthStore()

const loginToken = ref(authStore.token || '')
const loginError = ref('')
const loadingAuth = ref(false)
const loading = ref(false)
const error = ref('')
const keys = ref([])
const keyDetails = ref([])
const newKeyName = ref('')

const isAuthenticated = computed(() => authStore.isAuthenticated)

const quotaSummary = computed(() => ({
  configured: keyDetails.value.filter((item) => item.quota).length,
}))

function handleLogout() {
  authStore.logout()
  keys.value = []
  keyDetails.value = []
  router.replace({ name: 'login' })
}

async function handleLogin() {
  loadingAuth.value = true
  loginError.value = ''
  authStore.setToken(loginToken.value.trim())

  try {
    await fetchSelfServiceApi('/health')
    await refreshAll()
  } catch (err) {
    loginError.value = err instanceof Error ? err.message : 'Unable to open account portal.'
    authStore.logout()
  } finally {
    loadingAuth.value = false
  }
}

async function refreshAll() {
  if (!authStore.isAuthenticated) {
    return
  }

  loading.value = true
  error.value = ''
  try {
    const response = await fetchSelfServiceApi('/keys')
    keys.value = response
    keyDetails.value = await Promise.all(
      response.map(async (key) => {
        const detail = { key, quota: null, usage: null }
        try {
          detail.quota = await fetchSelfServiceApi(`/keys/${key.id}/quota`)
        } catch (err) {
          detail.quota = null
        }
        try {
          detail.usage = await fetchSelfServiceApi(`/keys/${key.id}/usage`)
        } catch (err) {
          detail.usage = null
        }
        return detail
      })
    )
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to refresh account data.'
  } finally {
    loading.value = false
  }
}

async function createKey() {
  if (!newKeyName.value.trim()) return
  loading.value = true
  error.value = ''
  try {
    await fetchSelfServiceApi('/keys', {
      method: 'POST',
      body: JSON.stringify({ name: newKeyName.value.trim() }),
    })
    newKeyName.value = ''
    await refreshAll()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to create key.'
  } finally {
    loading.value = false
  }
}

async function deleteKey(keyId) {
  loading.value = true
  error.value = ''
  try {
    await fetchSelfServiceApi(`/keys/${keyId}`, {
      method: 'DELETE',
    })
    await refreshAll()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to delete key.'
  } finally {
    loading.value = false
  }
}

function formatSeconds(value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  value = Number(value)
  if (value < 60) return `${value}s`
  const minutes = Math.floor(value / 60)
  const seconds = value % 60
  return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`
}

onMounted(() => {
  if (authStore.isAuthenticated) {
    refreshAll()
  }
})
</script>

<style scoped>
.account-page {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.hero,
.panel,
.toolbar,
.login-card {
  padding: 1.5rem;
}

.hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.5rem;
}

.eyebrow {
  margin: 0 0 0.5rem;
  color: var(--color-teal-cyan);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 0.75rem;
}

.muted {
  color: var(--color-text-muted);
}

.hero-stats {
  display: flex;
  gap: 1rem;
}

.stat-card {
  min-width: 130px;
  padding: 1rem;
  border-radius: 12px;
  background: rgba(11, 16, 33, 0.5);
  border: 1px solid var(--glass-border);
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.stat-value {
  font-size: 1.6rem;
  font-weight: 700;
}

.stat-label {
  color: var(--color-text-muted);
  font-size: 0.85rem;
}

.login-card {
  max-width: 520px;
}

.login-form,
.stacked-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.input-group label {
  font-size: 0.9rem;
  color: var(--color-text-secondary);
}

.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.toolbar-actions {
  display: flex;
  gap: 0.75rem;
}

.grid-two {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1.5rem;
}

.quota-list,
.key-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.quota-card,
.key-row {
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 1rem;
  background: rgba(11, 16, 33, 0.45);
}

.quota-card-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.quota-card h3,
.key-row h3 {
  margin-bottom: 0.25rem;
}

.quota-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
}

.metric-label {
  display: block;
  color: var(--color-text-muted);
  font-size: 0.75rem;
  margin-bottom: 0.25rem;
}

.metric-value {
  font-size: 1rem;
  font-weight: 700;
}

.key-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.error-msg,
.error-banner {
  color: var(--color-danger);
}

.error-banner {
  padding: 1rem 1.25rem;
}

.small {
  padding: 0.55rem 0.9rem;
  font-size: 0.85rem;
}

.danger {
  color: #ffd6db;
  border-color: rgba(239, 68, 68, 0.35);
}

code {
  color: var(--color-teal-cyan);
}

@media (max-width: 900px) {
  .hero,
  .toolbar,
  .quota-card-head,
  .key-row {
    flex-direction: column;
  }

  .grid-two,
  .quota-metrics {
    grid-template-columns: 1fr;
  }

  .hero-stats {
    width: 100%;
    flex-direction: column;
  }
}
</style>


