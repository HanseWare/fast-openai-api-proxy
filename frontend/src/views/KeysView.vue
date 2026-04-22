<template>
  <div class="management-view">
    <div class="view-header">
      <h2>API Keys Management</h2>
      <button @click="showCreate = !showCreate" class="btn-primary">
        {{ showCreate ? 'Cancel' : '+ Create Key' }}
      </button>
    </div>

    <!-- Create Form -->
    <transition name="slide-up">
      <div v-if="showCreate" class="glass-panel form-panel">
        <h3>Create New API Key</h3>
        <form @submit.prevent="createKey" class="inline-form">
          <div class="input-group">
            <label>Name</label>
            <input v-model="newKey.name" required placeholder="e.g. Production App" />
          </div>
          <div class="input-group">
            <label>Owner ID (Optional)</label>
            <input v-model="newKey.owner_id" placeholder="e.g. user_123" />
          </div>
          <button type="submit" class="btn-primary" :disabled="creating">Create</button>
        </form>

        <div v-if="createdSecret" class="secret-alert">
          <strong>Key Created:</strong> <code>{{ createdSecret }}</code>
          <p>Please copy this now. You won't be able to see it again!</p>
        </div>
      </div>
    </transition>

    <!-- Data Table -->
    <div class="glass-panel table-container">
      <table v-if="keys.length > 0">
        <thead>
          <tr>
            <th>Name</th>
            <th>Owner ID</th>
            <th>Key ID</th>
            <th>Masked Key</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="k in keys" :key="k.id">
            <td>{{ k.name }}</td>
            <td>{{ k.owner_id || '-' }}</td>
            <td class="mono">{{ k.id }}</td>
            <td class="mono">{{ k.masked_key }}</td>
            <td>
              <button @click="deleteKey(k.id)" class="btn-icon delete" title="Revoke Key">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">
        <p>No API keys found.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const keys = ref([])
const showCreate = ref(false)
const creating = ref(false)
const createdSecret = ref('')

const newKey = ref({
  name: '',
  owner_id: ''
})

async function loadKeys() {
  try {
    keys.value = await fetchApi('/keys')
  } catch (e) {
    alert('Failed to load keys: ' + e.message)
  }
}

async function createKey() {
  creating.value = true
  createdSecret.value = ''
  try {
    const payload = { name: newKey.value.name }
    if (newKey.value.owner_id) {
      payload.owner_id = newKey.value.owner_id
    }
    const res = await fetchApi('/keys', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
    createdSecret.value = res.api_key
    newKey.value.name = ''
    newKey.value.owner_id = ''
    await loadKeys()
  } catch (e) {
    alert('Failed to create key: ' + e.message)
  } finally {
    creating.value = false
  }
}

async function deleteKey(id) {
  if (!confirm('Are you sure you want to revoke this key?')) return
  try {
    await fetchApi(`/keys/${id}`, { method: 'DELETE' })
    await loadKeys()
  } catch (e) {
    alert('Failed to delete key: ' + e.message)
  }
}

onMounted(() => {
  loadKeys()
})
</script>

<style scoped>
/* Scoped layout styles for management views */
.view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.view-header h2 {
  margin: 0;
  color: var(--color-text-primary);
}

.form-panel {
  padding: 1.5rem;
  margin-bottom: 2rem;
}

.form-panel h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.1rem;
}

.inline-form {
  display: flex;
  gap: 1rem;
  align-items: flex-end;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  flex: 1;
}

.input-group label {
  font-size: 0.85rem;
  color: var(--color-text-secondary);
}

.secret-alert {
  margin-top: 1.5rem;
  padding: 1rem;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 8px;
  color: var(--color-success);
}

.secret-alert p {
  margin: 0.5rem 0 0 0;
  font-size: 0.9rem;
}

.table-container {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid var(--glass-border);
}

th {
  color: var(--color-text-secondary);
  font-weight: 500;
  font-size: 0.9rem;
}

tbody tr {
  transition: background var(--transition-fast);
}

tbody tr:hover {
  background: rgba(255, 255, 255, 0.02);
}

.mono {
  font-family: monospace;
  color: var(--color-teal-cyan);
}

.btn-icon {
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 4px;
  transition: background var(--transition-fast);
}

.btn-icon.delete:hover {
  background: rgba(239, 68, 68, 0.2);
}

.empty-state {
  padding: 3rem;
  text-align: center;
  color: var(--color-text-muted);
}
</style>
