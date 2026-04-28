<template>
  <div class="dashboard-layout">
    <!-- Sidebar -->
    <aside class="sidebar glass-panel">
      <div class="sidebar-header">
        <div class="logo-small"></div>
        <h2>FOAP</h2>
      </div>

      <nav class="nav-links">
        <router-link v-if="adminEnabled" to="/admin" class="nav-link" exact-active-class="active">
          <span class="icon">📊</span>
          Overview
        </router-link>
        <router-link v-if="adminEnabled" to="/admin/keys" class="nav-link" active-class="active">
          <span class="icon">🔑</span>
          API Keys
        </router-link>
        <router-link v-if="adminEnabled" to="/admin/endpoints" class="nav-link" active-class="active">
          <span class="icon">🛡️</span>
          Protected Endpoints
        </router-link>
        <router-link v-if="adminEnabled" to="/admin/quotas" class="nav-link" active-class="active">
          <span class="icon">⚖️</span>
          Quotas
        </router-link>
        <router-link v-if="adminEnabled" to="/admin/providers" class="nav-link" active-class="active">
          <span class="icon">🌐</span>
          Providers & Routing
        </router-link>
        <router-link v-if="adminEnabled" to="/admin/aliases" class="nav-link" active-class="active">
          <span class="icon">🎭</span>
          Virtual Models
        </router-link>
        <router-link v-if="selfServiceEnabled" to="/account" class="nav-link" active-class="active">
          <span class="icon">👤</span>
          Self-Service Portal
        </router-link>
      </nav>

      <div class="sidebar-footer">
        <button @click="handleLogout" class="btn-logout">
          <span class="icon">🚪</span> Logout
        </button>
      </div>
    </aside>

    <!-- Main Content Area -->
    <main class="main-content">
      <header class="top-header">
        <div class="header-content">
          <h1>{{ currentRouteName }}</h1>
          <div class="user-profile">
            Admin
          </div>
        </div>
      </header>
      
      <div class="content-wrapper">
        <router-view v-slot="{ Component }">
          <transition name="slide-up" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { fetchApi } from '../api'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const adminEnabled = ref(true)
const selfServiceEnabled = ref(true)

const currentRouteName = computed(() => {
  const map = {
    'dashboard': 'Overview',
    'keys': 'API Keys Management',
    'endpoints': 'Protected Endpoints',
    'quotas': 'Quota Policies & Overrides',
    'providers': 'Provider Routing Configuration',
    'aliases': 'Virtual Models Mapping',
    'import': 'JSON Config Import Studio'
  }
  return map[route.name] || 'Admin Dashboard'
})

onMounted(async () => {
  try {
    const cfg = await fetchApi('/auth-config')
    adminEnabled.value = !!cfg?.admin?.enabled
    selfServiceEnabled.value = !!cfg?.self_service?.enabled
  } catch (e) {
    // keep defaults (true) on error
  }
})

function handleLogout() {
  authStore.logout()
  // Best-effort: inform backend to invalidate cookie session
  fetchApi('/logout', { method: 'POST' }).catch(() => {})
  router.push({ name: 'admin-login' })
}
</script>

<style scoped>
.dashboard-layout {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 260px;
  border-radius: 0;
  border-left: none;
  border-top: none;
  border-bottom: none;
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  bottom: 0;
  left: 0;
  z-index: 10;
}

.sidebar-header {
  padding: 2rem 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.logo-small {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--color-berry-magenta), var(--color-teal-cyan));
}

.sidebar-header h2 {
  margin: 0;
  font-size: 1.5rem;
  color: var(--color-text-primary);
}

.nav-links {
  flex: 1;
  padding: 0 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  color: var(--color-text-secondary);
  font-weight: 500;
  transition: all var(--transition-fast);
}

.nav-link:hover {
  background: rgba(255, 255, 255, 0.05);
  color: var(--color-text-primary);
}

.nav-link.active {
  background: linear-gradient(90deg, rgba(217, 28, 92, 0.2), transparent);
  color: var(--color-text-primary);
  border-left: 3px solid var(--color-berry-magenta);
}

.sidebar-footer {
  padding: 1.5rem;
  border-top: 1px solid var(--glass-border);
}

.btn-logout {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  background: transparent;
  color: var(--color-text-secondary);
  border: 1px solid transparent;
  padding: 0.75rem;
  border-radius: 8px;
  transition: all var(--transition-fast);
}

.btn-logout:hover {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
  border-color: rgba(239, 68, 68, 0.3);
}

.main-content {
  flex: 1;
  margin-left: 260px;
  display: flex;
  flex-direction: column;
}

.top-header {
  height: 80px;
  border-bottom: 1px solid var(--glass-border);
  background: rgba(11, 16, 33, 0.5);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  position: sticky;
  top: 0;
  z-index: 5;
}

.header-content {
  height: 100%;
  padding: 0 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-content h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 500;
  color: var(--color-text-primary);
}

.user-profile {
  background: var(--color-navy-light);
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
  border: 1px solid var(--glass-border);
}

.content-wrapper {
  padding: 2rem;
  flex: 1;
}
</style>
