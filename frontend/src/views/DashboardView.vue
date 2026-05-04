<template>
  <div class="dashboard-overview">
    <div class="glass-panel stat-card" v-if="health">
      <h3>System Status</h3>
      <div class="status-indicator">
        <div class="dot" :class="health.status === 'ok' ? 'ok' : 'error'"></div>
        <span class="status-text">{{ health.status === 'ok' ? 'Online' : 'Degraded' }}</span>
      </div>
      <p class="meta">Scope: {{ health.scope }}</p>
    </div>

    <div class="glass-panel stat-card" v-if="authConfig">
      <h3>Auth Mode Configuration</h3>
      <div class="auth-modes">
        <div class="mode-col">
          <h4>Admin API</h4>
          <p>Enabled: <strong :class="authConfig.admin.enabled ? 'text-success' : 'text-muted'">{{ authConfig.admin.enabled }}</strong></p>
          <p>Mode: <strong>{{ authConfig.admin.mode }}</strong></p>
          <p>OIDC Enabled: <span :class="authConfig.admin.oidc_enabled ? 'text-success' : 'text-muted'">{{ authConfig.admin.oidc_enabled }}</span></p>
          <p>OIDC Only: <span :class="authConfig.admin.oidc_only ? 'text-warning' : 'text-muted'">{{ authConfig.admin.oidc_only }}</span></p>
        </div>
        <div class="mode-col">
          <h4>Self-Service API</h4>
          <p>Enabled: <strong :class="authConfig.self_service.enabled ? 'text-success' : 'text-muted'">{{ authConfig.self_service.enabled }}</strong></p>
          <p>Mode: <strong>{{ authConfig.self_service.mode }}</strong></p>
          <p>OIDC Enabled: <span :class="authConfig.self_service.oidc_enabled ? 'text-success' : 'text-muted'">{{ authConfig.self_service.oidc_enabled }}</span></p>
          <p>OIDC Only: <span :class="authConfig.self_service.oidc_only ? 'text-warning' : 'text-muted'">{{ authConfig.self_service.oidc_only }}</span></p>
        </div>
      </div>
    </div>

    <div class="glass-panel stat-card" v-if="providers.length">
      <h3>Provider Ratelimits</h3>
      <div class="providers-grid">
        <div v-for="p in providers" :key="p.id" class="provider-card">
          <div class="provider-head">
            <strong>{{ p.name }}</strong>
            <span class="pill" :class="ratelimits[p.id] ? 'pill-ok' : 'pill-muted'">
              {{ ratelimits[p.id] ? 'Synced' : '—' }}
            </span>
          </div>
          <div v-if="ratelimits[p.id]" class="rate-rows">
            <div class="rate-row"><span>Minute</span><span>{{ fmt(ratelimits[p.id].remaining_minute) }} / {{ fmt(ratelimits[p.id].limit_minute) }}</span></div>
            <div class="rate-row"><span>Hour</span><span>{{ fmt(ratelimits[p.id].remaining_hour) }} / {{ fmt(ratelimits[p.id].limit_hour) }}</span></div>
            <div class="rate-row"><span>Day</span><span>{{ fmt(ratelimits[p.id].remaining_day) }} / {{ fmt(ratelimits[p.id].limit_day) }}</span></div>
            <div class="rate-row"><span>Day</span><span>{{ fmt(ratelimits[p.id].remaining_month) }} / {{ fmt(ratelimits[p.id].limit_month) }}</span></div>
            <div class="rate-row meta"><span>Updated</span><span>{{ ratelimits[p.id].updated_at }}</span></div>
          </div>
          <div v-else class="muted">No snapshot</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const health = ref(null)
const authConfig = ref(null)
const providers = ref([])
const ratelimits = ref({})

function fmt(v) { return (v === null || v === undefined) ? '—' : v }

onMounted(async () => {
  try {
    health.value = await fetchApi('/health')
    authConfig.value = await fetchApi('/auth-config')
    providers.value = await fetchApi('/config/providers')
    // fetch ratelimits per provider
    const entries = await Promise.all(providers.value.map(async (p) => {
      try {
        const snap = await fetchApi(`/config/providers/${p.id}/ratelimits`)
        return [p.id, snap]
      } catch { return [p.id, null] }
    }))
    ratelimits.value = Object.fromEntries(entries)
  } catch (e) {
    console.error('Failed to fetch dashboard data', e)
  }
})
</script>

<style scoped>
.dashboard-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.stat-card { padding: 1.5rem; }
.stat-card h3 { margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem; color: var(--color-text-secondary); }

.status-indicator { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
.dot { width: 12px; height: 12px; border-radius: 50%; }
.dot.ok { background-color: var(--color-success); box-shadow: 0 0 10px var(--color-success); }
.dot.error { background-color: var(--color-danger); box-shadow: 0 0 10px var(--color-danger); }
.status-text { font-size: 1.25rem; font-weight: 600; }
.meta { color: var(--color-text-muted); font-size: 0.85rem; margin: 0; }

.auth-modes { display: flex; gap: 2rem; }
.mode-col { flex: 1; }
.mode-col h4 { font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--color-teal-cyan); }
.mode-col p { margin: 0.25rem 0; font-size: 0.9rem; }

.providers-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
.provider-card { border: 1px solid var(--glass-border); border-radius: 12px; padding: 1rem; background: rgba(11,16,33,0.45); }
.provider-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: .5rem; }
.pill { padding: .1rem .5rem; border-radius: 999px; font-size: .75rem; border: 1px solid transparent; }
.pill-ok { color: var(--color-success); background: rgba(16,185,129,.08); border-color: rgba(16,185,129,.2); }
.pill-muted { color: var(--color-text-secondary); background: rgba(255,255,255,.04); border-color: rgba(255,255,255,.08); }
.rate-rows { display: grid; gap: .25rem; }
.rate-row { display: flex; justify-content: space-between; }
.rate-row.meta { color: var(--color-text-muted); font-size: .85rem; }

.text-success { color: var(--color-success); }
.text-warning { color: var(--color-warning); }
.text-muted { color: var(--color-text-muted); }
</style>
