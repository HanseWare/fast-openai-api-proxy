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
            <span class="stat-value">{{ quotaSummary.configured }}</span>
            <span class="stat-label">With Quota</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{{ quotaSummary.totalUsed }}</span>
            <span class="stat-label">Used / Window</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{{ quotaSummary.totalRemaining }}</span>
            <span class="stat-label">Remaining / Window</span>
          </div>
        </div>
      </section>

      <section class="toolbar glass-panel">
        <div>
          <h2>Identity</h2>
          <p class="muted">
            Authenticated as <code>{{ identityLabel }}</code>
            <span v-if="authModeLabel"> · Mode: {{ authModeLabel }}</span>
          </p>
          <p class="muted" v-if="sessionInfo">Owner: <code>{{ sessionInfo.owner_id }}</code> · Source: {{ sessionInfo.auth_source }}</p>
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

              <div class="quota-card-meta">
                <span class="status-pill" :class="`status-pill--${item.statusTone}`">{{ item.statusLabel }}</span>
                <span class="muted">Resets in {{ item.resetLabel }}</span>
              </div>

              <div v-if="item.quotaLimit !== null" class="quota-progress">
                <div class="quota-progress-track" aria-hidden="true">
                  <div class="quota-progress-fill" :style="{ width: `${item.usagePercent}%` }"></div>
                </div>
                <div class="quota-progress-labels">
                  <span>{{ item.usedLabel }} / {{ item.quotaLimit }}</span>
                  <span>{{ item.usagePercent }}%</span>
                </div>
              </div>

              <div v-if="item.quota" class="quota-metrics">
                <div>
                  <span class="metric-label">Requests/minute</span>
                  <span class="metric-value">{{ item.quotaLimit }}</span>
                </div>
                <div>
                  <span class="metric-label">Used</span>
                  <span class="metric-value">{{ item.usedLabel }}</span>
                </div>
                <div>
                  <span class="metric-label">Remaining</span>
                  <span class="metric-value">{{ item.remainingLabel }}</span>
                </div>
                <div>
                  <span class="metric-label">Reset In</span>
                  <span class="metric-value">{{ item.resetLabel }}</span>
                </div>
              </div>
              <p v-else class="muted">No quota configured yet.</p>
            </article>
          </div>
        </div>
      </section>

      <section class="glass-panel panel">
        <h2>Usage insights</h2>
        <form class="usage-filter-form" @submit.prevent="applyUsageFilters">
          <div class="input-group">
            <label for="usage-model">Model filter</label>
            <input id="usage-model" v-model="usageFilterModel" type="text" placeholder="e.g. gpt-4o" />
          </div>
          <div class="input-group">
            <label for="usage-path">API path filter</label>
            <input id="usage-path" v-model="usageFilterApiPath" type="text" placeholder="e.g. /v1/chat/completions" />
          </div>
          <div class="input-group">
            <label for="usage-window">Trend points</label>
            <select id="usage-window" v-model.number="usageWindowSize">
              <option :value="4">4</option>
              <option :value="6">6</option>
              <option :value="12">12</option>
              <option :value="24">24</option>
            </select>
          </div>
          <div class="usage-filter-actions">
            <button class="btn-secondary small" type="submit">Apply</button>
            <button class="btn-secondary small" type="button" @click="resetUsageFilters">Reset</button>
          </div>
        </form>
        <p class="muted" v-if="!usageSummary">No usage data yet.</p>
        <div v-else class="usage-grid">
          <article v-for="window in usageWindows" :key="window.name" class="usage-card">
            <div class="usage-card-head">
              <h3>{{ window.label }}</h3>
              <span class="status-pill status-pill--neutral">{{ window.total }} requests</span>
            </div>
            <p class="muted">Reset in {{ formatSeconds(window.resetIn) }}</p>
            <div class="trend-row" v-if="window.trend.length">
              <div class="trend-bar" v-for="point in window.trend" :key="`${window.name}-${point.window_bucket}`">
                <span class="trend-bar-fill" :style="{ height: `${point.percent}%` }"></span>
              </div>
            </div>
            <p class="muted" v-if="!window.rows.length">No requests in this window.</p>
            <div v-else class="usage-rows">
              <div class="usage-row" v-for="row in window.rows" :key="`${window.name}-${row.model}-${row.api_path}`">
                <div>
                  <strong>{{ row.model }}</strong>
                  <p class="muted">{{ row.api_path }}</p>
                </div>
                <span class="metric-value">{{ row.request_count }}</span>
              </div>
            </div>
          </article>
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
const usageSummary = ref(null)
const loginToken = ref(authStore.token || '')
const loginError = ref('')
const loadingAuth = ref(false)
const loading = ref(false)
const error = ref('')
const keys = ref([])
const keyDetails = ref([])
const newKeyName = ref('')
const usageFilterModel = ref('')
const usageFilterApiPath = ref('')
const usageWindowSize = ref(6)
const createdSecret = ref('')

const isAuthenticated = computed(() => authStore.isAuthenticated)

const authMode = computed(() => authConfig.value?.mode || 'loading')
const oidcClient = computed(() => authConfig.value?.oidc_client || null)
const ssoButtonLabel = computed(() => {
  const providerName = oidcClient.value?.display_name?.trim()
  return providerName ? `${providerName}` : 'Sign in with SSO'
})
const authModeLabel = computed(() => {
  switch (authMode.value) {
    case 'oidc-only':
      return 'OIDC only'
    case 'oidc-or-token-hash':
      return 'OIDC + Token'
    case 'token-hash-only':
      return 'Static token'
    default:
      return 'Loading auth mode…'
  }
})
const authModeClass = computed(() => {
  switch (authMode.value) {
    case 'oidc-only':
      return 'oidc'
    case 'oidc-or-token-hash':
      return 'hybrid'
    case 'token-hash-only':
      return 'token'
    default:
      return 'loading'
  }
})
const authModeHint = computed(() => authConfig.value?.login_hint || 'Loading self-service auth mode…')
const loginTitle = computed(() => {
  switch (authMode.value) {
    case 'oidc-only':
      return 'Continue with OIDC'
    case 'oidc-or-token-hash':
      return 'Sign in with OIDC or token'
    case 'token-hash-only':
      return 'Sign in with your FOAP token'
    default:
      return 'Sign in to your account'
  }
})
const loginDescription = computed(() => {
  switch (authMode.value) {
    case 'oidc-only':
      return 'Your portal expects an OIDC-issued access token. Tokens from your identity provider are accepted here.'
    case 'oidc-or-token-hash':
      return 'You can use either an OIDC access token or a static FOAP bearer token for self-service access.'
    case 'token-hash-only':
      return 'Use your static FOAP bearer token to manage personal API keys.'
    default:
      return 'Use your self-service bearer token or OIDC access token to access this portal.'
  }
})
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

const quotaSummary = computed(() => ({
  configured: keyDetails.value.filter((item) => item.quotaLimit !== null).length,
  totalUsed: keyDetails.value.reduce((sum, item) => sum + (item.used ?? 0), 0),
  totalRemaining: keyDetails.value.reduce((sum, item) => sum + (item.remaining ?? 0), 0),
  exhausted: keyDetails.value.filter((item) => item.statusTone === 'danger').length,
}))

const usageWindows = computed(() => {
  if (!usageSummary.value) {
    return []
  }

  const windows = usageSummary.value.windows || {}
  const totals = usageSummary.value.totals || {}

  const buildTrend = (trend = []) => {
    const maxValue = trend.reduce((max, row) => Math.max(max, row.request_count || 0), 0) || 1
    return trend.map((row) => ({
      ...row,
      percent: Math.max(8, Math.round(((row.request_count || 0) / maxValue) * 100)),
    }))
  }

  return [
    {
      name: 'minute',
      label: 'Current minute',
      total: totals.minute || 0,
      rows: (windows.minute || []).slice(0, 5),
      resetIn: windows.minute_meta?.reset_in_seconds,
      trend: buildTrend(windows.minute_trend || []),
    },
    {
      name: 'hour',
      label: 'Current hour',
      total: totals.hour || 0,
      rows: (windows.hour || []).slice(0, 5),
      resetIn: windows.hour_meta?.reset_in_seconds,
      trend: buildTrend(windows.hour_trend || []),
    },
    {
      name: 'day',
      label: 'Current day',
      total: totals.day || 0,
      rows: (windows.day || []).slice(0, 5),
      resetIn: windows.day_meta?.reset_in_seconds,
      trend: buildTrend(windows.day_trend || []),
    },
  ]
})

function _usageSummaryQuery() {
  const params = new URLSearchParams()
  if (usageFilterModel.value.trim()) {
    params.set('model', usageFilterModel.value.trim())
  }
  if (usageFilterApiPath.value.trim()) {
    params.set('api_path', usageFilterApiPath.value.trim())
  }
  params.set('window_size', String(usageWindowSize.value || 6))
  return params.toString()
}

function decorateKeyDetail(key, quota, usage) {
  const quotaLimit = quota?.requests_per_minute ?? null
  const used = usage?.used ?? null
  const remaining = usage?.remaining ?? null
  const usagePercent = quotaLimit && used !== null ? Math.min(100, Math.round((used / quotaLimit) * 100)) : 0

  let statusTone = 'neutral'
  let statusLabel = 'No quota'

  if (quotaLimit !== null) {
    if (remaining === 0) {
      statusTone = 'danger'
      statusLabel = 'Exhausted'
    } else if (usagePercent >= 80) {
      statusTone = 'warning'
      statusLabel = 'Near limit'
    } else {
      statusTone = 'success'
      statusLabel = 'Healthy'
    }
  }

  return {
    key,
    quota,
    usage,
    quotaLimit,
    used,
    remaining,
    usedLabel: used ?? '—',
    remainingLabel: remaining ?? '—',
    usagePercent,
    resetLabel: formatSeconds(usage?.reset_in_seconds),
    statusTone,
    statusLabel,
  }
}

const identityLabel = computed(() => {
  if (authStore.token) return 'Bearer token present'
  // If no token but cookie-SSO active and we have sessionInfo, show owner id short form
  if (sessionInfo.value?.owner_id) {
    const oid = sessionInfo.value.owner_id
    // Display short label, e.g., oidc:subhash → last 8 chars
    const short = oid.length > 12 ? `${oid.slice(0, 8)}…${oid.slice(-4)}` : oid
    return `cookie session (${short})`
  }
  return 'unknown'
})

function handleLogout() {
  authStore.logout()
  // Call backend to invalidate server-side sessions (best-effort)
  fetchSelfServiceApi('/logout', { method: 'POST' }).catch(() => {})
  // Clear any OIDC target hint
  try { sessionStorage.removeItem('foap_oidc_target') } catch (e) {}
  keys.value = []
  keyDetails.value = []
  usageSummary.value = null
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
    // Redirect to backend BFF for self-service OIDC login
    // This will NOT return; it redirects the browser
    await startOidcLogin('/api', 'account')
  } catch (err) {
    loginError.value = err.message || 'Failed to start SSO login. Check OIDC configuration.'
    loadingAuth.value = false
  }
}

async function loadAuthConfig() {
  try {
    authConfig.value = await fetchSelfServiceApi('/auth-config')
  } catch (err) {
    authConfig.value = {
      mode: 'loading',
      login_hint: 'Unable to load self-service auth mode right now.',
    }
  }
}

async function loadSessionInfo() {
  try {
    sessionInfo.value = await fetchSelfServiceApi('/session')
  } catch (err) {
    sessionInfo.value = null
  }
}

async function loadUsageSummary() {
  const qs = _usageSummaryQuery()
  usageSummary.value = await fetchSelfServiceApi(`/usage/summary${qs ? `?${qs}` : ''}`)
}

async function applyUsageFilters() {
  if (!authStore.isAuthenticated) {
    return
  }
  loading.value = true
  error.value = ''
  try {
    await loadUsageSummary()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load usage summary.'
  } finally {
    loading.value = false
  }
}

function resetUsageFilters() {
  usageFilterModel.value = ''
  usageFilterApiPath.value = ''
  usageWindowSize.value = 6
  applyUsageFilters()
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
    await loadSessionInfo()
    await loadUsageSummary()
    keyDetails.value = await Promise.all(
      response.map(async (key) => {
        let quota = null
        let usage = null
        try {
          quota = await fetchSelfServiceApi(`/keys/${key.id}/quota`)
        } catch (err) {
          quota = null
        }
        try {
          usage = await fetchSelfServiceApi(`/keys/${key.id}/usage`)
        } catch (err) {
          usage = null
        }
        return decorateKeyDetail(key, quota, usage)
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

.login-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
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
  font-size: 0.9rem;
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

.usage-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}

.usage-filter-form {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.usage-filter-actions {
  display: flex;
  align-items: end;
  gap: 0.5rem;
}

.usage-card {
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 1rem;
  background: rgba(11, 16, 33, 0.45);
}

.usage-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
}

.usage-card-head h3 {
  margin: 0;
}

.usage-rows {
  margin-top: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.trend-row {
  margin: 0.6rem 0;
  height: 48px;
  display: flex;
  align-items: end;
  gap: 0.35rem;
}

.trend-bar {
  flex: 1;
  height: 100%;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: end;
  overflow: hidden;
}

.trend-bar-fill {
  display: block;
  width: 100%;
  background: linear-gradient(180deg, rgba(0, 229, 255, 0.95), rgba(217, 28, 92, 0.75));
}

.usage-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
}

.usage-row p {
  margin: 0;
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

.quota-card-meta {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  margin-bottom: 0.75rem;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
}

.status-pill--success {
  color: var(--color-success);
  background: rgba(16, 185, 129, 0.08);
  border-color: rgba(16, 185, 129, 0.2);
}

.status-pill--warning {
  color: var(--color-warning);
  background: rgba(245, 158, 11, 0.08);
  border-color: rgba(245, 158, 11, 0.2);
}

.status-pill--danger {
  color: var(--color-danger);
  background: rgba(239, 68, 68, 0.08);
  border-color: rgba(239, 68, 68, 0.2);
}

.status-pill--neutral {
  color: var(--color-text-secondary);
  background: rgba(255, 255, 255, 0.04);
  border-color: rgba(255, 255, 255, 0.08);
}

.quota-progress {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-bottom: 0.9rem;
}

.quota-progress-track {
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  overflow: hidden;
}

.quota-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--color-teal-cyan), var(--color-berry-magenta));
}

.quota-progress-labels {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--color-text-secondary);
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

.metric-value--danger {
  color: var(--color-danger);
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

.warning {
  color: #ffe7b3;
}

.success {
  color: #bff7df;
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
  .quota-metrics,
  .usage-grid {
    grid-template-columns: 1fr;
  }

  .usage-filter-form {
    grid-template-columns: 1fr;
  }

  .hero-stats {
    width: 100%;
    flex-direction: column;
  }
}

.btn-sso {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  width: 100%;
  padding: 0.85rem 1.5rem;
  border: 1px solid rgba(0, 229, 255, 0.3);
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(162, 89, 255, 0.08));
  color: var(--color-teal-cyan);
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.25s ease;
  margin-top: 1rem;
}

.btn-sso:hover:not(:disabled) {
  background: linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(162, 89, 255, 0.15));
  border-color: rgba(0, 229, 255, 0.5);
  box-shadow: 0 0 20px rgba(0, 229, 255, 0.15);
  transform: translateY(-1px);
}

.btn-sso:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.divider {
  display: flex;
  align-items: center;
  gap: 1rem;
  color: var(--color-text-muted);
  font-size: 0.8rem;
  margin: 0.5rem 0;
}

.divider::before,
.divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(255, 255, 255, 0.08);
}

.secret-alert {
  padding: 0.9rem 1rem;
  background: rgba(16, 185, 129, 0.08);
  border: 1px solid rgba(16, 185, 129, 0.35);
  border-radius: 8px;
}
</style>


