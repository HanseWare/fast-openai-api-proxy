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
          <p>Mode: <strong>{{ authConfig.admin.mode }}</strong></p>
          <p>OIDC Enabled: <span :class="authConfig.admin.oidc_enabled ? 'text-success' : 'text-muted'">{{ authConfig.admin.oidc_enabled }}</span></p>
          <p>OIDC Only: <span :class="authConfig.admin.oidc_only ? 'text-warning' : 'text-muted'">{{ authConfig.admin.oidc_only }}</span></p>
        </div>
        <div class="mode-col">
          <h4>Self-Service API</h4>
          <p>Mode: <strong>{{ authConfig.self_service.mode }}</strong></p>
          <p>OIDC Enabled: <span :class="authConfig.self_service.oidc_enabled ? 'text-success' : 'text-muted'">{{ authConfig.self_service.oidc_enabled }}</span></p>
          <p>OIDC Only: <span :class="authConfig.self_service.oidc_only ? 'text-warning' : 'text-muted'">{{ authConfig.self_service.oidc_only }}</span></p>
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

onMounted(async () => {
  try {
    health.value = await fetchApi('/health')
    authConfig.value = await fetchApi('/auth-config')
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

.stat-card {
  padding: 1.5rem;
}

.stat-card h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  color: var(--color-text-secondary);
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.dot.ok {
  background-color: var(--color-success);
  box-shadow: 0 0 10px var(--color-success);
}

.dot.error {
  background-color: var(--color-danger);
  box-shadow: 0 0 10px var(--color-danger);
}

.status-text {
  font-size: 1.25rem;
  font-weight: 600;
}

.meta {
  color: var(--color-text-muted);
  font-size: 0.85rem;
  margin: 0;
}

.auth-modes {
  display: flex;
  gap: 2rem;
}

.mode-col {
  flex: 1;
}

.mode-col h4 {
  font-size: 0.95rem;
  margin-bottom: 0.5rem;
  color: var(--color-teal-cyan);
}

.mode-col p {
  margin: 0.25rem 0;
  font-size: 0.9rem;
}

.text-success { color: var(--color-success); }
.text-warning { color: var(--color-warning); }
.text-muted { color: var(--color-text-muted); }
</style>
