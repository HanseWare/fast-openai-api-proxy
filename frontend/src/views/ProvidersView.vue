<template>
  <div class="management-view">
    <div class="view-header">
      <h2>Routing Providers</h2>
      <div style="display: flex; gap: 1rem;">
        <router-link to="/import" class="btn-secondary">JSON Import Studio</router-link>
        <button @click="showCreateProvider = !showCreateProvider" class="btn-primary">
          {{ showCreateProvider ? 'Cancel' : '+ New Provider' }}
        </button>
      </div>
    </div>

    <!-- Create Provider Form -->
    <transition name="slide-up">
      <div v-if="showCreateProvider" class="glass-panel form-panel">
        <h3>Create Provider</h3>
        <form @submit.prevent="createProvider" class="inline-form">
          <div class="input-group">
            <label>Name</label>
            <input v-model="newProvider.name" required placeholder="openAI" />
          </div>
          <div class="input-group">
            <label>API Key Variable</label>
            <input v-model="newProvider.api_key_variable" required placeholder="OPENAI_API_TOKEN" />
          </div>
          <div class="input-group">
            <label>Prefix (Optional)</label>
            <input v-model="newProvider.prefix" placeholder="Azure/" />
          </div>
          <div class="input-group">
            <label>Default Base URL</label>
            <input v-model="newProvider.default_base_url" placeholder="https://api.openai.com" />
          </div>
          <button type="submit" class="btn-primary" :disabled="creating">Save</button>
        </form>
      </div>
    </transition>

    <!-- Providers List -->
    <div v-for="p in providers" :key="p.id" class="glass-panel provider-card">
      <div class="provider-header">
        <div>
          <h3>{{ p.name }}</h3>
          <p class="meta">Var: <code>{{ p.api_key_variable }}</code> | Default URL: <code>{{ p.default_base_url || 'N/A' }}</code></p>
        </div>
        <button @click="deleteProvider(p.id)" class="btn-icon delete">🗑️</button>
      </div>

      <!-- Models for this provider -->
      <div class="models-section">
        <div class="models-header">
          <h4>Models</h4>
          <button @click="addModel(p.id)" class="btn-text">+ Add Model</button>
        </div>
        
        <table v-if="p.models && p.models.length > 0">
          <thead>
            <tr>
              <th>Model Name</th>
              <th>Endpoints</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in p.models" :key="m.id">
              <td class="mono">{{ m.name }}</td>
              <td>
                <span class="badge" v-for="ep in m.endpoints" :key="ep.id" :title="ep.target_base_url || p.default_base_url">
                  {{ ep.path }} -> {{ ep.target_model_name }}
                </span>
                <button @click="addEndpoint(m.id)" class="btn-text" style="font-size: 0.8rem; margin-left: 0.5rem">+ Add Endpoint</button>
              </td>
              <td>
                <button @click="deleteModel(m.id)" class="btn-icon delete">🗑️</button>
              </td>
            </tr>
          </tbody>
        </table>
        <p v-else class="text-muted" style="margin-left: 1rem; font-size: 0.9rem;">No models configured.</p>
      </div>
    </div>
    
    <div v-if="providers.length === 0" class="empty-state">
      No providers configured in the database. The proxy is currently running in JSON fallback mode.
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const providers = ref([])
const showCreateProvider = ref(false)
const creating = ref(false)

const newProvider = ref({ name: '', api_key_variable: '', prefix: '', default_base_url: '' })

async function loadData() {
  try {
    const provs = await fetchApi('/config/providers')
    for (let p of provs) {
      p.models = await fetchApi(`/config/providers/${p.id}/models`)
      for (let m of p.models) {
        m.endpoints = await fetchApi(`/config/models/${m.id}/endpoints`)
      }
    }
    providers.value = provs
  } catch (e) {
    alert('Failed to load routing config: ' + e.message)
  }
}

async function createProvider() {
  creating.value = true
  try {
    await fetchApi('/config/providers', {
      method: 'POST',
      body: JSON.stringify(newProvider.value)
    })
    showCreateProvider.value = false
    await loadData()
  } catch(e) { alert('Failed: ' + e.message) }
  finally { creating.value = false }
}

async function deleteProvider(id) {
  if (!confirm('Delete provider and ALL its models?')) return
  await fetchApi(`/config/providers/${id}`, { method: 'DELETE' })
  await loadData()
}

async function addModel(providerId) {
  const name = prompt("Enter new model name (e.g. gpt-4o):")
  if (!name) return
  await fetchApi('/config/models', {
    method: 'POST',
    body: JSON.stringify({ provider_id: providerId, name })
  })
  await loadData()
}

async function deleteModel(id) {
  if (!confirm('Delete model and ALL its endpoints?')) return
  await fetchApi(`/config/models/${id}`, { method: 'DELETE' })
  await loadData()
}

async function addEndpoint(modelId) {
  const path = prompt("Enter endpoint path (e.g. v1/chat/completions):")
  if (!path) return
  const target = prompt("Enter upstream target_model_name:")
  if (!target) return
  await fetchApi('/config/endpoints', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId, path, target_model_name: target })
  })
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

.provider-card {
  margin-bottom: 2rem;
  overflow: hidden;
}

.provider-header {
  padding: 1.5rem;
  background: rgba(0, 0, 0, 0.2);
  border-bottom: 1px solid var(--glass-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.provider-header h3 { margin: 0; color: var(--color-teal-cyan); }
.meta { margin: 0.5rem 0 0 0; color: var(--color-text-muted); font-size: 0.85rem; }

.models-section {
  padding: 1rem;
}

.models-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 0 1rem 1rem 1rem;
}
.models-header h4 { margin: 0; color: var(--color-text-secondary); }

table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--glass-border); }
th { color: var(--color-text-secondary); font-weight: 500; font-size: 0.85rem; }
tbody tr:hover { background: rgba(255, 255, 255, 0.02); }
.mono { font-family: monospace; color: var(--color-text-primary); }
.btn-icon { background: transparent; border: none; cursor: pointer; padding: 0.5rem; border-radius: 4px; }
.btn-icon.delete:hover { background: rgba(239, 68, 68, 0.2); }
.empty-state { padding: 3rem; text-align: center; color: var(--color-text-muted); }

.badge {
  display: inline-block;
  padding: 0.2rem 0.4rem;
  margin-right: 0.5rem;
  margin-bottom: 0.2rem;
  border-radius: 4px;
  font-size: 0.75rem;
  background: rgba(0, 229, 255, 0.1);
  color: var(--color-teal-cyan);
  border: 1px solid rgba(0, 229, 255, 0.3);
}

.btn-text {
  background: transparent;
  border: none;
  color: var(--color-berry-magenta);
  cursor: pointer;
  font-weight: 600;
}
.btn-text:hover { color: var(--color-berry-light); }
</style>
