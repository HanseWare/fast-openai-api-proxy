<template>
  <div class="management-view">
    <div class="view-header">
      <h2>Protected Endpoints</h2>
      <button @click="showCreate = !showCreate" class="btn-primary">
        {{ showCreate ? 'Cancel' : '+ Protect Endpoint' }}
      </button>
    </div>

    <!-- Create Form -->
    <transition name="slide-up">
      <div v-if="showCreate" class="glass-panel form-panel">
        <h3>Add Protected Endpoint Rule</h3>
        <form @submit.prevent="createEndpoint" class="inline-form">
          <div class="input-group">
            <label>Path Pattern</label>
            <input v-model="newRule.path" required placeholder="/v1/models" />
          </div>
          <div class="input-group">
            <label>HTTP Method</label>
            <select v-model="newRule.method" required>
              <option value="GET">GET</option>
              <option value="POST">POST</option>
              <option value="PUT">PUT</option>
              <option value="DELETE">DELETE</option>
              <option value="*">* (All)</option>
            </select>
          </div>
          <button type="submit" class="btn-primary" :disabled="creating">Add Rule</button>
        </form>
      </div>
    </transition>

    <!-- Data Table -->
    <div class="glass-panel table-container">
      <table v-if="endpoints.length > 0">
        <thead>
          <tr>
            <th>Path Pattern</th>
            <th>Method</th>
            <th>Rule ID</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="ep in endpoints" :key="ep.id">
            <td class="mono">{{ ep.path }}</td>
            <td>
              <span class="badge" :class="ep.method.toLowerCase()">{{ ep.method }}</span>
            </td>
            <td class="mono text-muted">{{ ep.id }}</td>
            <td>
              <button @click="deleteEndpoint(ep.id)" class="btn-icon delete" title="Delete Rule">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">
        <p>No protected endpoints configured. All routes open.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const endpoints = ref([])
const showCreate = ref(false)
const creating = ref(false)

const newRule = ref({
  path: '',
  method: 'POST'
})

async function loadEndpoints() {
  try {
    endpoints.value = await fetchApi('/protected-endpoints')
  } catch (e) {
    alert('Failed to load endpoints: ' + e.message)
  }
}

async function createEndpoint() {
  creating.value = true
  try {
    await fetchApi('/protected-endpoints', {
      method: 'POST',
      body: JSON.stringify({
        path: newRule.value.path,
        method: newRule.value.method
      })
    })
    newRule.value.path = ''
    await loadEndpoints()
    showCreate.value = false
  } catch (e) {
    alert('Failed to add rule: ' + e.message)
  } finally {
    creating.value = false
  }
}

async function deleteEndpoint(id) {
  if (!confirm('Are you sure you want to remove this protection rule?')) return
  try {
    await fetchApi(`/protected-endpoints/${id}`, { method: 'DELETE' })
    await loadEndpoints()
  } catch (e) {
    alert('Failed to delete rule: ' + e.message)
  }
}

onMounted(() => {
  loadEndpoints()
})
</script>

<style scoped>
/* Inherits from KeysView structure conceptually, but redefined for scoped */
.view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}
.view-header h2 { margin: 0; color: var(--color-text-primary); }

.form-panel { padding: 1.5rem; margin-bottom: 2rem; }
.form-panel h3 { margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem; }
.inline-form { display: flex; gap: 1rem; align-items: flex-end; }
.input-group { display: flex; flex-direction: column; gap: 0.5rem; flex: 1; }
.input-group label { font-size: 0.85rem; color: var(--color-text-secondary); }

.table-container { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 1rem; text-align: left; border-bottom: 1px solid var(--glass-border); }
th { color: var(--color-text-secondary); font-weight: 500; font-size: 0.9rem; }
tbody tr:hover { background: rgba(255, 255, 255, 0.02); }

.mono { font-family: monospace; color: var(--color-teal-cyan); }
.text-muted { color: var(--color-text-muted); font-size: 0.85rem; }

.badge {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.1);
}
.badge.post { background: rgba(16, 185, 129, 0.2); color: var(--color-success); }
.badge.get { background: rgba(0, 229, 255, 0.2); color: var(--color-teal-cyan); }
.badge.delete { background: rgba(239, 68, 68, 0.2); color: var(--color-danger); }

.btn-icon { background: transparent; border: none; cursor: pointer; padding: 0.5rem; border-radius: 4px; }
.btn-icon.delete:hover { background: rgba(239, 68, 68, 0.2); }
.empty-state { padding: 3rem; text-align: center; color: var(--color-text-muted); }
</style>
