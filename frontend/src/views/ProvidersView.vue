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
          <div class="input-group">
            <label>Max Upstream Retry (sec)</label>
            <input type="number" v-model="newProvider.max_upstream_retry_seconds" placeholder="0" />
          </div>
          <div class="input-group" style="flex-direction: row; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <input type="checkbox" v-model="newProvider.sync_provider_ratelimits" id="sync_limits" />
            <label for="sync_limits" style="margin: 0;">Sync Provider Ratelimits</label>
          </div>
          <div class="input-group" style="width: 100%;">
            <label>Route Fallbacks (JSON format: {"/v1/chat": "gpt-4o-mini"})</label>
            <input v-model="newProvider.route_fallbacks_str" placeholder="{}" />
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
          <p class="meta">Retry: {{ p.max_upstream_retry_seconds || 0 }}s | Sync Limits: {{ p.sync_provider_ratelimits ? 'Yes' : 'No' }}</p>
        </div>
        <div style="display: flex; gap: 0.5rem;">
          <button @click="openEditProvider(p)" class="btn-icon" title="Edit Provider">✏️</button>
          <button @click="deleteProvider(p.id)" class="btn-icon delete" title="Delete Provider">🗑️</button>
        </div>
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
              <td class="mono">
                {{ m.name }}
                <button @click="editModelName(m)" class="btn-icon" style="font-size: 0.8rem; margin-left: 0.5rem" title="Edit Model Name">✏️</button>
              </td>
              <td>
                <div v-for="ep in m.endpoints" :key="ep.id" class="endpoint-row" @click="openEditEndpoint(ep, m.id)" title="Click to edit">
                  <span class="ep-path">{{ ep.path }}</span>
                  <span class="ep-arrow">&rarr;</span>
                  <span class="ep-target">{{ ep.target_model_name }}</span>
                  <span class="ep-meta" v-if="ep.fallback_model_name"> | fallback: {{ep.fallback_model_name}}</span>
                  <span class="ep-meta" v-if="ep.target_base_url"> | url: {{ep.target_base_url}}</span>
                </div>
                <button @click="addEndpoint(m.id)" class="btn-text" style="font-size: 0.8rem; margin-top: 0.5rem">+ Add Endpoint</button>
              </td>
              <td>
                <button @click="deleteModel(m.id)" class="btn-icon delete" title="Delete Model">🗑️</button>
              </td>
            </tr>
          </tbody>
        </table>
        <p v-else class="text-muted" style="margin-left: 1rem; font-size: 0.9rem;">No models configured.</p>
      </div>
    </div>

    <!-- Edit Provider Modal -->
    <div v-if="editProviderForm.show" class="modal-overlay">
      <div class="glass-panel modal-content">
        <h3>Edit Provider</h3>
        <form @submit.prevent="submitEditProvider">
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Name</label>
            <input v-model="editProviderForm.name" required />
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>API Key Variable</label>
            <input v-model="editProviderForm.api_key_variable" />
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Prefix (Optional)</label>
            <input v-model="editProviderForm.prefix" />
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Default Base URL</label>
            <input v-model="editProviderForm.default_base_url" />
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Max Upstream Retry (sec)</label>
            <input type="number" v-model="editProviderForm.max_upstream_retry_seconds" />
          </div>
          <div class="input-group" style="flex-direction: row; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;">
            <input type="checkbox" v-model="editProviderForm.sync_provider_ratelimits" id="edit_sync_limits" />
            <label for="edit_sync_limits" style="margin: 0;">Sync Provider Ratelimits</label>
          </div>
          <div class="input-group" style="margin-bottom: 1.5rem;">
            <label>Route Fallbacks (JSON format)</label>
            <input v-model="editProviderForm.route_fallbacks_str" placeholder="{}" />
          </div>
          <div style="display: flex; gap: 1rem; justify-content: flex-end;">
            <button type="button" class="btn-secondary" @click="editProviderForm.show = false">Cancel</button>
            <button type="submit" class="btn-primary" :disabled="creating">Update</button>
          </div>
        </form>
      </div>
    </div>
    
    <div v-if="providers.length === 0" class="empty-state">
      No providers configured in the database. The proxy is currently running in JSON fallback mode.
    </div>

    <!-- Add/Edit Endpoint Modal (Overlay) -->
    <div v-if="endpointForm.show" class="modal-overlay">
      <div class="glass-panel modal-content">
        <h3>{{ endpointForm.editMode ? 'Edit Endpoint' : 'Add Endpoint' }}</h3>
        <form @submit.prevent="submitEndpoint">
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Endpoint Path</label>
            <input list="common-paths" v-model="endpointForm.path" required placeholder="v1/chat/completions" />
            <datalist id="common-paths">
              <option value="v1/chat/completions"></option>
              <option value="v1/completions"></option>
              <option value="v1/embeddings"></option>
              <option value="v1/models"></option>
            </datalist>
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Target Model Name</label>
            <input v-model="endpointForm.target_model_name" required placeholder="gpt-4o" />
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Target Base URL (Overrides Provider)</label>
            <input v-model="endpointForm.target_base_url" placeholder="https://api.openai.com" />
          </div>
          <div class="inline-form" style="margin-bottom: 1.5rem;">
            <div class="input-group">
              <label>Req Timeout (s)</label>
              <input type="number" v-model="endpointForm.request_timeout" placeholder="60" />
            </div>
            <div class="input-group">
              <label>Health Timeout (s)</label>
              <input type="number" v-model="endpointForm.health_timeout" placeholder="60" />
            </div>
          </div>
          <div class="input-group" style="margin-bottom: 1.5rem;">
            <label>Fallback Model Name (Optional)</label>
            <input v-model="endpointForm.fallback_model_name" placeholder="gpt-4o-mini" />
          </div>
          <div style="display: flex; gap: 1rem; justify-content: flex-end;">
            <button v-if="endpointForm.editMode" type="button" class="btn-icon delete" style="margin-right: auto;" @click="deleteEndpoint(endpointForm.id)">🗑️ Delete</button>
            <button type="button" class="btn-secondary" @click="endpointForm.show = false">Cancel</button>
            <button type="submit" class="btn-primary" :disabled="creating">{{ endpointForm.editMode ? 'Update' : 'Add' }}</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const providers = ref([])
const showCreateProvider = ref(false)
const creating = ref(false)

const newProvider = ref({ name: '', api_key_variable: '', prefix: '', default_base_url: '', max_upstream_retry_seconds: 0, sync_provider_ratelimits: false, route_fallbacks_str: '{}' })
const endpointForm = ref({ show: false, editMode: false, id: null, modelId: null, path: '', target_model_name: '', fallback_model_name: '', target_base_url: '', request_timeout: null, health_timeout: null })
const editProviderForm = ref({ show: false, id: null, name: '', api_key_variable: '', prefix: '', default_base_url: '', max_upstream_retry_seconds: 0, sync_provider_ratelimits: false, route_fallbacks_str: '{}' })

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
      body: JSON.stringify({
        ...newProvider.value,
        route_fallbacks: JSON.parse(newProvider.value.route_fallbacks_str || '{}')
      })
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

function openEditProvider(p) {
  editProviderForm.value = {
    show: true,
    id: p.id,
    name: p.name,
    api_key_variable: p.api_key_variable,
    prefix: p.prefix,
    default_base_url: p.default_base_url,
    max_upstream_retry_seconds: p.max_upstream_retry_seconds || 0,
    sync_provider_ratelimits: p.sync_provider_ratelimits || false,
    route_fallbacks_str: p.route_fallbacks ? JSON.stringify(p.route_fallbacks) : '{}'
  }
}

async function submitEditProvider() {
  creating.value = true
  try {
    await fetchApi(`/config/providers/${editProviderForm.value.id}`, {
      method: 'PUT',
      body: JSON.stringify({
        name: editProviderForm.value.name,
        api_key_variable: editProviderForm.value.api_key_variable,
        prefix: editProviderForm.value.prefix,
        default_base_url: editProviderForm.value.default_base_url,
        max_upstream_retry_seconds: editProviderForm.value.max_upstream_retry_seconds,
        sync_provider_ratelimits: editProviderForm.value.sync_provider_ratelimits,
        route_fallbacks: JSON.parse(editProviderForm.value.route_fallbacks_str || '{}')
      })
    })
    editProviderForm.value.show = false
    await loadData()
  } catch (e) {
    alert('Failed to update provider: ' + e.message)
  } finally {
    creating.value = false
  }
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

async function editModelName(model) {
  const name = prompt("Edit model name:", model.name)
  if (!name || name === model.name) return
  try {
    await fetchApi(`/config/models/${model.id}`, {
      method: 'PUT',
      body: JSON.stringify({ name })
    })
    await loadData()
  } catch (e) {
    alert('Failed to edit model: ' + e.message)
  }
}

async function deleteModel(id) {
  if (!confirm('Delete model and ALL its endpoints?')) return
  await fetchApi(`/config/models/${id}`, { method: 'DELETE' })
  await loadData()
}

async function addEndpoint(modelId) {
  endpointForm.value = { show: true, editMode: false, id: null, modelId, path: 'v1/chat/completions', target_model_name: '', fallback_model_name: '' }
}

async function openEditEndpoint(ep, modelId) {
  endpointForm.value = {
    show: true,
    editMode: true,
    id: ep.id,
    modelId,
    path: ep.path,
    target_model_name: ep.target_model_name,
    fallback_model_name: ep.fallback_model_name || '',
    target_base_url: ep.target_base_url || '',
    request_timeout: ep.request_timeout || null,
    health_timeout: ep.health_timeout || null
  }
}

async function deleteEndpoint(id) {
  if (!confirm('Delete endpoint?')) return
  await fetchApi(`/config/endpoints/${id}`, { method: 'DELETE' })
  endpointForm.value.show = false
  await loadData()
}

async function submitEndpoint() {
  creating.value = true
  try {
    if (endpointForm.value.editMode) {
      await fetchApi(`/config/endpoints/${endpointForm.value.id}`, {
        method: 'PUT',
        body: JSON.stringify({
          path: endpointForm.value.path,
          target_model_name: endpointForm.value.target_model_name,
          fallback_model_name: endpointForm.value.fallback_model_name || null,
          target_base_url: endpointForm.value.target_base_url || null,
          request_timeout: endpointForm.value.request_timeout || null,
          health_timeout: endpointForm.value.health_timeout || null
        })
      })
    } else {
      await fetchApi('/config/endpoints', {
        method: 'POST',
        body: JSON.stringify({
          model_id: endpointForm.value.modelId,
          path: endpointForm.value.path,
          target_model_name: endpointForm.value.target_model_name,
          fallback_model_name: endpointForm.value.fallback_model_name || null,
          target_base_url: endpointForm.value.target_base_url || null,
          request_timeout: endpointForm.value.request_timeout || null,
          health_timeout: endpointForm.value.health_timeout || null
        })
      })
    }
    endpointForm.value.show = false
    await loadData()
  } catch (e) {
    alert('Failed to save endpoint: ' + e.message)
  } finally {
    creating.value = false
  }
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

.endpoint-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  padding: 0.4rem 0.6rem;
  margin-bottom: 0.3rem;
  background: rgba(0, 0, 0, 0.15);
  border: 1px solid var(--glass-border);
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
  font-size: 0.85rem;
}
.endpoint-row:hover {
  background: rgba(0, 229, 255, 0.05);
  border-color: rgba(0, 229, 255, 0.3);
}
.ep-path {
  font-family: monospace;
  color: var(--color-teal-cyan);
  font-weight: bold;
}
.ep-arrow {
  margin: 0 0.5rem;
  color: var(--color-text-muted);
}
.ep-target {
  font-weight: 500;
  color: var(--color-text-primary);
}
.ep-meta {
  color: var(--color-text-secondary);
  font-size: 0.8rem;
  margin-left: 0.4rem;
}

.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-content {
  background: var(--color-bg-base);
  padding: 2rem;
  width: 400px;
  max-width: 90vw;
  border-radius: 12px;
}
.modal-content h3 { margin-top: 0; color: var(--color-text-primary); }
</style>
