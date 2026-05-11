<template>
  <div class="account-page">
    <template v-if="!isAuthenticated">
      <div class="login-container">
        <div class="glass-panel login-box">
          <div class="logo-area">
            <div class="logo-circle"></div>
            <h2>Self-Service Portal</h2>
            <p>Generate, review, and revoke your API keys.</p>
            <span class="mode-chip" :class="`mode-chip--${authModeClass}`" style="margin-top: 1rem;">{{ authModeLabel }}</span>
          </div>

          <!-- SSO Button -->
          <button
            v-if="oidcClient"
            class="btn-sso"
            type="button"
            :disabled="loadingAuth"
            @click="handleSsoLogin"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/></svg>
            {{ ssoButtonLabel }}
          </button>

          <!-- Divider -->
          <div v-if="oidcClient && authMode !== 'oidc-only'" class="divider">
            <span>or use a token</span>
          </div>

          <!-- Token form -->
          <form v-if="!oidcClient || authMode !== 'oidc-only'" class="login-form" @submit.prevent="handleLogin">
            <div class="input-group">
              <label for="account-token">{{ loginFieldLabel }}</label>
              <input
                id="account-token"
                v-model="loginToken"
                type="password"
                :placeholder="loginPlaceholder"
                autocomplete="current-password"
                required
              />
            </div>
            <p class="auth-guidance" v-if="authModeHint">{{ authModeHint }}</p>
            <button class="btn-primary" type="submit" :disabled="loadingAuth">
              {{ loadingAuth ? 'Verifying…' : 'Open Account Portal' }}
            </button>
            <p v-if="loginError" class="error-msg">{{ loginError }}</p>
          </form>
        </div>
      </div>
    </template>

    <template v-else>
      <section class="hero glass-panel">
        <div>
          <p class="eyebrow">/account</p>
          <h1>Self-Service Portal</h1>
          <p class="muted">
            Generate, review, and revoke your own FOAP API keys.
            <span v-if="authModeHint"> {{ authModeHint }}</span>
          </p>
        </div>
        <div class="hero-stats">
          <div class="stat-card">
            <span class="stat-value">{{ keys.length }}</span>
            <span class="stat-label">Active Keys</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{{ budgets.length }}</span>
            <span class="stat-label">Active Budgets</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{{ totalCost.toFixed(2) }}</span>
            <span class="stat-label">Total Credits Used</span>
          </div>
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
          <div v-if="createdSecret" class="secret-alert" style="margin-top:1rem">
            <strong>Key Created:</strong> <code>{{ createdSecret }}</code>
            <p class="muted">Please copy this now. You won't be able to see it again!</p>
          </div>
        </div>

        <div class="glass-panel panel">
          <h2>Your Budgets</h2>
          <p class="muted" v-if="!decoratedBudgets.length">No budgets allocated for your account.</p>
          <div v-else class="quota-list">
             <article v-for="item in decoratedBudgets" :key="item.budget.id" class="quota-card">
               <div class="quota-card-head">
                 <div>
                   <h3>{{ item.budget.window.charAt(0).toUpperCase() + item.budget.window.slice(1) }} Budget</h3>
                   <p class="muted">Scope: {{ item.budget.scope || 'Global (All Models)' }}</p>
                 </div>
               </div>

              <div class="quota-card-meta">
                <span class="status-pill" :class="`status-pill--${item.statusTone}`">{{ item.statusLabel }}</span>
              </div>

              <div class="quota-progress">
                <div class="quota-progress-track" aria-hidden="true">
                  <div class="quota-progress-fill" :style="{ width: `${item.usagePercent}%` }"></div>
                </div>
                <div class="quota-progress-labels">
                  <span>{{ item.cost.toFixed(2) }} / {{ item.budget.budget_amount.toFixed(2) }} credits</span>
                  <span>{{ item.usagePercent }}%</span>
                </div>
              </div>

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
import { startOidcLogin } from '../services/oidc'

const router = useRouter()
const authStore = useAuthStore()

const authConfig = ref(null)
const sessionInfo = ref(null)
const loginToken = ref(authStore.token || '')
const loginError = ref('')
const loadingAuth = ref(false)
const loading = ref(false)
const error = ref('')

const keys = ref([])
const newKeyName = ref('')
const createdSecret = ref('')

const budgets = ref([])
const budgetUsage = ref([])

const isAuthenticated = computed(() => authStore.isAuthenticated)

const authMode = computed(() => authConfig.value?.mode || 'loading')
const oidcClient = computed(() => authConfig.value?.oidc_client || null)
const ssoButtonLabel = computed(() => {
  const providerName = oidcClient.value?.display_name?.trim()
  return providerName ? `${providerName}` : 'Sign in with SSO'
})
const authModeLabel = computed(() => {
  switch (authMode.value) {
    case 'oidc-only': return 'OIDC only'
    case 'oidc-or-token-hash': return 'OIDC + Token'
    case 'token-hash-only': return 'Static token'
    default: return 'Loading auth mode…'
  }
})
const authModeClass = computed(() => {
  switch (authMode.value) {
    case 'oidc-only': return 'oidc'
    case 'oidc-or-token-hash': return 'hybrid'
    case 'token-hash-only': return 'token'
    default: return 'loading'
  }
})
const authModeHint = computed(() => authConfig.value?.login_hint || 'Loading self-service auth mode…')

const loginFieldLabel = computed(() => {
  if (authMode.value === 'oidc-only') return 'OIDC Access Token'
  if (authMode.value === 'oidc-or-token-hash') return 'OIDC or FOAP Token'
  return 'Bearer Token'
})
const loginPlaceholder = computed(() => {
  if (authMode.value === 'oidc-only') return 'Paste your OIDC access token'
  if (authMode.value === 'oidc-or-token-hash') return 'Paste your OIDC or FOAP token'
  return 'Paste your FOAP bearer token'
})

const totalCost = computed(() => {
  return budgetUsage.value.reduce((sum, usage) => sum + usage.cost, 0)
})

const decoratedBudgets = computed(() => {
  return budgets.value.map(budget => {
    // Find matching usage
    const usage = budgetUsage.value.find(u => u.budget_id === budget.id)
    const cost = usage ? usage.cost : 0.0
    const limit = budget.budget_amount

    const usagePercent = limit > 0 ? Math.min(100, Math.round((cost / limit) * 100)) : 0

    let statusTone = 'success'
    let statusLabel = 'Healthy'

    if (cost >= limit) {
      statusTone = 'danger'
      statusLabel = 'Exhausted'
    } else if (usagePercent >= 80) {
      statusTone = 'warning'
      statusLabel = 'Near limit'
    }

    return {
      budget,
      cost,
      usagePercent,
      statusTone,
      statusLabel
    }
  })
})

function handleLogout() {
  authStore.logout()
  fetchSelfServiceApi('/logout', { method: 'POST' }).catch(() => {})
  try { sessionStorage.removeItem('foap_oidc_target') } catch (e) {}
  keys.value = []
  budgets.value = []
  budgetUsage.value = []
  router.replace({ path: '/' })
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

async function handleSsoLogin() {
  loadingAuth.value = true
  loginError.value = ''
  try {
    if (!oidcClient.value) {
      loginError.value = 'OIDC is not configured on this server.'
      loadingAuth.value = false
      return
    }
    await startOidcLogin('/api', 'account')
  } catch (err) {
    loginError.value = err.message || 'Failed to start SSO login.'
    loadingAuth.value = false
  }
}

async function loadAuthConfig() {
  try {
    authConfig.value = await fetchSelfServiceApi('/auth-config')
  } catch (err) {
    authConfig.value = { mode: 'loading', login_hint: 'Unable to load self-service auth mode right now.' }
  }
}

async function refreshAll() {
  if (!authStore.isAuthenticated) return
  loading.value = true
  error.value = ''
  try {
    keys.value = await fetchSelfServiceApi('/keys')
    budgets.value = await fetchSelfServiceApi('/budgets')
    budgetUsage.value = await fetchSelfServiceApi('/budgets/usage')
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
  createdSecret.value = ''
  try {
    const res = await fetchSelfServiceApi('/keys', {
      method: 'POST',
      body: JSON.stringify({ name: newKeyName.value.trim() }),
    })
    createdSecret.value = res.api_key || ''
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
    await fetchSelfServiceApi(`/keys/${keyId}`, { method: 'DELETE' })
    await refreshAll()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to delete key.'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadAuthConfig()
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

.login-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 80vh;
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

.logo-area { text-align: center; }
.logo-circle {
  width: 64px; height: 64px; border-radius: 50%;
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
.logo-area p { margin: 0; color: var(--color-text-muted); font-size: 0.9rem; }

.hero, .panel, .login-card { padding: 1.5rem; }
.hero { display: flex; align-items: flex-start; justify-content: space-between; gap: 1.5rem; }
.eyebrow { margin: 0 0 0.5rem; color: var(--color-teal-cyan); text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.75rem; }
.muted { color: var(--color-text-muted); }
.hero-stats { display: flex; gap: 1rem; }

.mode-chip {
  display: inline-flex; align-items: center; padding: 0.35rem 0.75rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.02em; border: 1px solid transparent; white-space: nowrap;
}
.mode-chip--oidc, .mode-chip--hybrid { color: var(--color-teal-cyan); background: rgba(0, 229, 255, 0.08); border-color: rgba(0, 229, 255, 0.2); }
.mode-chip--token, .mode-chip--loading { color: var(--color-text-secondary); background: rgba(255, 255, 255, 0.04); border-color: rgba(255, 255, 255, 0.08); }

.auth-guidance { margin: 0; color: var(--color-text-secondary); font-size: 0.9rem; }

.stat-card {
  min-width: 130px; padding: 1rem; border-radius: 12px; background: rgba(11, 16, 33, 0.5);
  border: 1px solid var(--glass-border); display: flex; flex-direction: column; gap: 0.25rem;
}
.stat-value { font-size: 1.6rem; font-weight: 700; }
.stat-label { color: var(--color-text-muted); font-size: 0.85rem; }

.login-form, .stacked-form { display: flex; flex-direction: column; gap: 1rem; }
.input-group { display: flex; flex-direction: column; gap: 0.5rem; }
.input-group label { font-size: 0.9rem; color: var(--color-text-secondary); }

.toolbar-actions { display: flex; gap: 0.75rem; }
.grid-two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem; }

.quota-list, .key-list { display: flex; flex-direction: column; gap: 0.75rem; }
.quota-card, .key-row { border: 1px solid var(--glass-border); border-radius: 12px; padding: 1rem; background: rgba(11, 16, 33, 0.45); }
.quota-card-meta { display: flex; justify-content: space-between; gap: 1rem; align-items: center; margin-bottom: 0.75rem; }

.status-pill {
  display: inline-flex; align-items: center; padding: 0.28rem 0.65rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.02em; border: 1px solid transparent;
}
.status-pill--success { color: var(--color-success); background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.2); }
.status-pill--warning { color: var(--color-warning); background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.2); }
.status-pill--danger { color: var(--color-danger); background: rgba(239, 68, 68, 0.08); border-color: rgba(239, 68, 68, 0.2); }
.status-pill--neutral { color: var(--color-text-secondary); background: rgba(255, 255, 255, 0.04); border-color: rgba(255, 255, 255, 0.08); }

.quota-progress { display: flex; flex-direction: column; gap: 0.35rem; margin-bottom: 0.9rem; }
.quota-progress-track { height: 8px; border-radius: 999px; background: rgba(255, 255, 255, 0.06); overflow: hidden; }
.quota-progress-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--color-teal-cyan), var(--color-berry-magenta)); }
.quota-progress-labels { display: flex; justify-content: space-between; gap: 1rem; font-size: 0.85rem; color: var(--color-text-secondary); }
.quota-card-head { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; margin-bottom: 1rem; }
.quota-card h3, .key-row h3 { margin-bottom: 0.25rem; }

.key-row { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.error-msg, .error-banner { color: var(--color-danger); }
.error-banner { padding: 1rem 1.25rem; }
.small { padding: 0.55rem 0.9rem; font-size: 0.85rem; }
.danger { color: #ffd6db; border-color: rgba(239, 68, 68, 0.35); }
code { color: var(--color-teal-cyan); }

@media (max-width: 900px) {
  .hero, .quota-card-head, .key-row { flex-direction: column; }
  .grid-two { grid-template-columns: 1fr; }
  .hero-stats { width: 100%; flex-direction: column; }
}

.btn-sso {
  display: flex; align-items: center; justify-content: center; gap: 0.6rem; width: 100%; padding: 0.85rem 1.5rem;
  border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 8px;
  background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(162, 89, 255, 0.08));
  color: var(--color-teal-cyan); font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.25s ease; margin-top: 1rem;
}
.btn-sso:hover:not(:disabled) {
  background: linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(162, 89, 255, 0.15));
  border-color: rgba(0, 229, 255, 0.5); box-shadow: 0 0 20px rgba(0, 229, 255, 0.15); transform: translateY(-1px);
}
.btn-sso:disabled { opacity: 0.5; cursor: not-allowed; }
.divider { display: flex; align-items: center; gap: 1rem; color: var(--color-text-muted); font-size: 0.8rem; margin: 0.5rem 0; }
.divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: rgba(255, 255, 255, 0.08); }
.secret-alert { padding: 0.9rem 1rem; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; }
</style>
