<template>
  <div class="management-view">
    <div class="view-header">
      <h2>Quota Policies</h2>
      <button @click="showCreatePolicy = !showCreatePolicy" class="btn-primary">
        {{ showCreatePolicy ? 'Cancel' : '+ New Policy' }}
      </button>
    </div>

    <!-- Create Policy Form -->
    <transition name="slide-up">
      <div v-if="showCreatePolicy" class="glass-panel form-panel">
        <h3>Create Quota Policy</h3>
        <p class="text-muted" style="margin-bottom: 1rem;">Base policies apply to specific endpoints/models. Enable "Per User?" to track consumption per unique API Key.</p>
        <form @submit.prevent="createPolicy" class="inline-form">
          <div class="input-group">
            <label>API Path</label>
            <input list="quota-api-paths" v-model="newPolicy.api_path" required placeholder="/v1/chat/completions" />
            <datalist id="quota-api-paths">
              <option value="v1/chat/completions"></option>
              <option value="v1/completions"></option>
              <option value="v1/embeddings"></option>
              <option value="v1/images/generations"></option>
              <option value="*"></option>
            </datalist>
          </div>
          <div class="input-group">
            <label>Model (or *)</label>
            <input list="quota-models" v-model="newPolicy.model" required placeholder="gpt-4" />
            <datalist id="quota-models">
              <option value="*"></option>
              <option v-for="m in availableModels" :key="m" :value="m"></option>
            </datalist>
          </div>
          <div class="input-group">
            <label>Window</label>
            <select v-model="newPolicy.window_type" required>
              <option value="minute">Minute</option>
              <option value="hour">Hour</option>
              <option value="day">Day</option>
            </select>
          </div>
          <div class="input-group">
            <label>Limit</label>
            <input type="number" v-model="newPolicy.request_limit" required min="1" />
          </div>
          <div class="input-group" style="flex-direction: row; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <input type="checkbox" v-model="newPolicy.enforce_per_user" id="enforce_per_user" />
            <label for="enforce_per_user" style="margin: 0;">Per User?</label>
          </div>
          <button type="submit" class="btn-primary" :disabled="creating">Save</button>
        </form>
      </div>
    </transition>

    <!-- Policies Table -->
    <div class="glass-panel table-container">
      <table v-if="policies.length > 0">
        <thead>
          <tr>
            <th>API Path</th>
            <th>Model</th>
            <th>Limit</th>
            <th>Per User?</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="p in policies" :key="p.id">
            <td class="mono">{{ p.api_path }}</td>
            <td class="mono">{{ p.model }}</td>
            <td><strong>{{ p.request_limit }}</strong> / {{ p.window_type }}</td>
            <td>{{ p.enforce_per_user ? 'Yes' : 'No (Global)' }}</td>
            <td>
              <button @click="deletePolicy(p.id)" class="btn-icon delete">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">No quota policies set.</div>
    </div>

    <!-- Overrides Section -->
    <div class="view-header" style="margin-top: 3rem;">
      <h2>Quota Overrides</h2>
      <button @click="showCreateOverride = !showCreateOverride" class="btn-secondary">
        {{ showCreateOverride ? 'Cancel' : '+ Add Override' }}
      </button>
    </div>

    <transition name="slide-up">
      <div v-if="showCreateOverride" class="glass-panel form-panel">
        <h3>Create Quota Override</h3>
        <p class="text-muted" style="margin-bottom: 1rem;">Overrides take precedence over base policies. Use this to grant specific users higher limits or exempt them entirely.</p>
        <form @submit.prevent="createOverride" class="inline-form">
          <div class="input-group">
            <label>API Path</label>
            <input list="quota-api-paths" v-model="newOverride.api_path" required placeholder="/v1/chat/completions" />
          </div>
          <div class="input-group">
            <label>Model</label>
            <input list="quota-models" v-model="newOverride.model" required placeholder="*" />
          </div>
          <div class="input-group">
            <label>Owner ID</label>
            <input v-model="newOverride.owner_id" required placeholder="user_123" />
          </div>
          <div class="input-group">
            <label>Window</label>
            <select v-model="newOverride.window_type" required>
              <option value="minute">Minute</option>
              <option value="hour">Hour</option>
              <option value="day">Day</option>
            </select>
          </div>
          <div class="input-group">
            <label>Limit</label>
            <input type="number" v-model="newOverride.request_limit" required min="1" />
          </div>
          <button type="submit" class="btn-secondary" :disabled="creating">Save Override</button>
        </form>
      </div>
    </transition>

    <div class="glass-panel table-container">
      <table v-if="overrides.length > 0">
        <thead>
          <tr>
            <th>API Path</th>
            <th>Model</th>
            <th>Owner ID</th>
            <th>Limit</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="o in overrides" :key="o.id">
            <td class="mono">{{ o.api_path }}</td>
            <td class="mono">{{ o.model }}</td>
            <td class="mono" style="color: var(--color-berry-light)">{{ o.owner_id }}</td>
            <td><strong>{{ o.request_limit }}</strong> / {{ o.window_type }}</td>
            <td>
              <span class="badge" :class="o.active_now ? 'active' : 'inactive'">
                {{ o.window_state }}
              </span>
            </td>
            <td>
              <button @click="deleteOverride(o.id)" class="btn-icon delete">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">No quota overrides set.</div>
    </div>

  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const policies = ref([])
const overrides = ref([])
const showCreatePolicy = ref(false)
const showCreateOverride = ref(false)
const creating = ref(false)
const availableModels = ref([])

const newPolicy = ref({ api_path: 'v1/chat/completions', model: '*', window_type: 'minute', request_limit: 10, enforce_per_user: true })
const newOverride = ref({ api_path: 'v1/chat/completions', model: '*', owner_id: '', window_type: 'minute', request_limit: 100 })

async function loadData() {
  try {
    policies.value = await fetchApi('/quota-policies')
    overrides.value = await fetchApi('/quota-overrides')
    
    // Fetch unique model names from config API for datalist
    const provs = await fetchApi('/config/providers')
    const modelsSet = new Set()
    for (let p of provs) {
      const models = await fetchApi(`/config/providers/${p.id}/models`)
      for (let m of models) modelsSet.add(m.name)
    }
    availableModels.value = Array.from(modelsSet)
  } catch (e) {
    alert('Failed to load quotas: ' + e.message)
  }
}

async function createPolicy() {
  creating.value = true
  try {
    await fetchApi('/quota-policies', {
      method: 'POST',
      body: JSON.stringify(newPolicy.value)
    })
    showCreatePolicy.value = false
    await loadData()
  } catch(e) { alert('Failed: ' + e.message) }
  finally { creating.value = false }
}

async function createOverride() {
  creating.value = true
  try {
    await fetchApi('/quota-overrides', {
      method: 'POST',
      body: JSON.stringify({...newOverride.value, exempt: false})
    })
    showCreateOverride.value = false
    await loadData()
  } catch(e) { alert('Failed: ' + e.message) }
  finally { creating.value = false }
}

async function deletePolicy(id) {
  if (!confirm('Delete this policy?')) return
  await fetchApi(`/quota-policies/${id}`, { method: 'DELETE' })
  await loadData()
}

async function deleteOverride(id) {
  if (!confirm('Delete this override?')) return
  await fetchApi(`/quota-overrides/${id}`, { method: 'DELETE' })
  await loadData()
}

onMounted(() => loadData())
</script>

<style scoped>
.view-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.view-header h2 { margin: 0; color: var(--color-text-primary); }
.form-panel { padding: 1.5rem; margin-bottom: 2rem; }
.form-panel h3 { margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem; }
.inline-form { display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap; }
.input-group { display: flex; flex-direction: column; gap: 0.5rem; flex: 1; min-width: 150px;}
.input-group label { font-size: 0.85rem; color: var(--color-text-secondary); }

.table-container { overflow-x: auto; margin-bottom: 2rem; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 1rem; text-align: left; border-bottom: 1px solid var(--glass-border); }
th { color: var(--color-text-secondary); font-weight: 500; font-size: 0.9rem; }
tbody tr:hover { background: rgba(255, 255, 255, 0.02); }
.mono { font-family: monospace; color: var(--color-teal-cyan); }
.btn-icon { background: transparent; border: none; cursor: pointer; padding: 0.5rem; border-radius: 4px; }
.btn-icon.delete:hover { background: rgba(239, 68, 68, 0.2); }
.empty-state { padding: 3rem; text-align: center; color: var(--color-text-muted); }

.badge {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: capitalize;
}
.badge.active { background: rgba(16, 185, 129, 0.2); color: var(--color-success); }
.badge.inactive { background: rgba(100, 116, 139, 0.2); color: var(--color-text-muted); }
</style>
