<template>
  <div class="management-view">
    <div class="view-header">
      <h2>Routing Providers</h2>
      <div style="display: flex; gap: 1rem;">
        <router-link to="/admin/import" class="btn-secondary">JSON Import Studio</router-link>
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
            <input v-model="newProvider.base_url" placeholder="https://api.openai.com" />
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
          <p class="meta">Var: <code>{{ p.api_key_variable }}</code> | Default URL: <code>{{ p.base_url || 'N/A' }}</code></p>
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
          <button @click="openAddModel(p.id)" class="btn-text">+ Add Model</button>
        </div>
        
        <table v-if="p.models && p.models.length > 0">
          <thead>
            <tr>
              <th>Model Name</th>
              <th>Type</th>
              <th>Target Model / URL</th>
              <th>Endpoints</th>
              <th>Economics</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in p.models" :key="m.id">
              <td class="mono">
                {{ m.name }}
                <div v-if="m.hide_on_models_endpoint" style="font-size:0.75rem; color:var(--color-danger);">Hidden</div>
              </td>
              <td><span class="status-pill status-pill--neutral">{{ m.type }}</span></td>
              <td>
                <div>{{ m.target_model_name }}</div>
                <div v-if="m.target_base_url" class="text-muted" style="font-size:0.8rem">{{ m.target_base_url }}</div>
              </td>
              <td>
                <div v-if="m.supported_endpoints && m.supported_endpoints.length">
                  <span v-for="ep in m.supported_endpoints" :key="ep" class="badge">{{ ep }}</span>
                </div>
                <span v-else class="text-muted">None</span>
              </td>
              <td style="font-size: 0.85rem">
                <div>Price/Unit: {{ m.price_per_unit || 0 }}</div>
                <div>Min Credits: {{ m.min_credits_per_request || 0 }}</div>
              </td>
              <td>
                <button @click="openEditModel(m, p.id)" class="btn-icon" title="Edit Model">✏️</button>
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
            <input v-model="editProviderForm.base_url" />
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

    <!-- Add/Edit Model Modal -->
    <div v-if="modelForm.show" class="modal-overlay">
      <div class="glass-panel modal-content" style="max-height: 90vh; overflow-y: auto;">
        <h3>{{ modelForm.id ? 'Edit Model' : 'Add Model' }}</h3>
        <form @submit.prevent="submitModelForm">
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Name</label>
            <input v-model="modelForm.name" required placeholder="gpt-4o" />
          </div>
          
          <div class="inline-form" style="margin-bottom: 1rem;">
            <div class="input-group">
              <label>Target Model Name</label>
              <input v-model="modelForm.target_model_name" required placeholder="gpt-4o" />
            </div>
            <div class="input-group">
              <label>Model Type</label>
              <select v-model="modelForm.type" @change="onModelTypeChange" required>
                <option value="llm">LLM</option>
                <option value="embedding">Embedding</option>
                <option value="image">Image</option>
                <option value="audio_transcription">Audio Transcription</option>
                <option value="audio_speech">Audio Speech</option>
              </select>
            </div>
          </div>

          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Supported Endpoints (Hold Ctrl/Cmd to select multiple)</label>
            <select v-model="modelForm.supported_endpoints" multiple size="4" required class="multiselect">
              <option v-for="ep in availableEndpoints" :key="ep" :value="ep">{{ ep }}</option>
            </select>
          </div>

          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Target Base URL (Override Provider URL)</label>
            <input v-model="modelForm.target_base_url" placeholder="https://api.openai.com" />
          </div>

          <div class="inline-form" style="margin-bottom: 1rem;">
            <div class="input-group">
              <label>Price Per Unit</label>
              <input type="number" step="0.0000001" v-model.number="modelForm.price_per_unit" />
            </div>
            <div class="input-group">
              <label>Min Credits / Request</label>
              <input type="number" step="0.0001" v-model.number="modelForm.min_credits_per_request" />
            </div>
          </div>

          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Owned By</label>
            <input v-model="modelForm.owned_by" placeholder="FOAP" />
          </div>

          <div class="input-group" style="flex-direction: row; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;">
            <input type="checkbox" v-model="modelForm.hide_on_models_endpoint" id="edit_model_hide" />
            <label for="edit_model_hide" style="margin: 0;">Hide on /v1/models</label>
          </div>

          <div style="display: flex; gap: 1rem; justify-content: flex-end;">
            <button type="button" class="btn-secondary" @click="modelForm.show = false">Cancel</button>
            <button type="submit" class="btn-primary" :disabled="creating">{{ modelForm.id ? 'Update' : 'Add' }}</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { fetchApi } from '../api'

const providers = ref([])
const showCreateProvider = ref(false)
const creating = ref(false)

const newProvider = ref({ name: '', api_key_variable: '', prefix: '', base_url: '', max_upstream_retry_seconds: 0, sync_provider_ratelimits: false, route_fallbacks_str: '{}' })
const editProviderForm = ref({ show: false, id: null, name: '', api_key_variable: '', prefix: '', base_url: '', max_upstream_retry_seconds: 0, sync_provider_ratelimits: false, route_fallbacks_str: '{}' })

const endpointMap = {
  llm: ['/v1/chat/completions', '/v1/completions', '/v1/responses', "/v1/moderations"],
  embedding: ['/v1/embeddings'],
  image: ['/v1/images/generations', '/v1/images/edits', '/v1/images/variations', '/v1/images/data'],
  audio_transcription: ['/v1/audio/transcriptions', '/v1/audio/translations'],
  audio_speech: ['/v1/audio/speech']
}

const modelForm = ref({ 
  show: false, 
  id: null, 
  provider_id: null,
  name: '', 
  type: 'llm',
  target_model_name: '',
  target_base_url: '',
  supported_endpoints: [],
  price_per_unit: 0.0,
  min_credits_per_request: 0.0,
  owned_by: 'FOAP', 
  hide_on_models_endpoint: false 
})

const availableEndpoints = computed(() => {
  return endpointMap[modelForm.value.type] || []
})

function onModelTypeChange() {
  modelForm.value.supported_endpoints = []
}

async function loadData() {
  try {
    const provs = await fetchApi('/config/providers')
    for (let p of provs) {
      p.models = await fetchApi(`/config/providers/${p.id}/models`)
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
        base_url: newProvider.value.base_url || null,
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
    base_url: p.base_url,
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
        base_url: editProviderForm.value.base_url || null,
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

function openAddModel(providerId) {
  modelForm.value = {
    show: true,
    id: null,
    provider_id: providerId,
    name: '', 
    type: 'llm',
    target_model_name: '',
    target_base_url: '',
    supported_endpoints: ['/v1/chat/completions'],
    price_per_unit: 0.0,
    min_credits_per_request: 0.0,
    owned_by: 'FOAP',
    hide_on_models_endpoint: false
  }
}

function openEditModel(model, providerId) {
  modelForm.value = {
    show: true,
    id: model.id,
    provider_id: providerId,
    name: model.name,
    type: model.type || 'llm',
    target_model_name: model.target_model_name,
    target_base_url: model.target_base_url || '',
    supported_endpoints: model.supported_endpoints || [],
    price_per_unit: model.price_per_unit || 0.0,
    min_credits_per_request: model.min_credits_per_request || 0.0,
    owned_by: model.owned_by || 'FOAP',
    hide_on_models_endpoint: !!model.hide_on_models_endpoint
  }
}

async function submitModelForm() {
  creating.value = true
  try {
    const payload = {
      name: modelForm.value.name,
      type: modelForm.value.type,
      target_model_name: modelForm.value.target_model_name,
      target_base_url: modelForm.value.target_base_url || null,
      supported_endpoints: modelForm.value.supported_endpoints,
      price_per_unit: modelForm.value.price_per_unit,
      min_credits_per_request: modelForm.value.min_credits_per_request,
      owned_by: modelForm.value.owned_by,
      hide_on_models_endpoint: modelForm.value.hide_on_models_endpoint
    }

    if (modelForm.value.id) {
      await fetchApi(`/config/models/${modelForm.value.id}`, {
        method: 'PUT',
        body: JSON.stringify(payload)
      })
    } else {
      payload.provider_id = modelForm.value.provider_id
      await fetchApi('/config/models', {
        method: 'POST',
        body: JSON.stringify(payload)
      })
    }
    modelForm.value.show = false
    await loadData()
  } catch (e) {
    alert('Failed to save model: ' + e.message)
  } finally {
    creating.value = false
  }
}

async function deleteModel(id) {
  if (!confirm('Delete this model?')) return
  await fetchApi(`/config/models/${id}`, { method: 'DELETE' })
  await loadData()
}

onMounted(() => loadData())
</script>

<style scoped>
.view-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.view-header h2 { margin: 0; color: var(--color-text-primary); }
.glass-panel {
  background: rgba(12,14,16,0.96);
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: 0 6px 18px rgba(0,0,0,0.5);
}

.form-panel { padding: 1.5rem; margin-bottom: 2rem; }
.form-panel h3 { margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem; }
.inline-form { display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap; }
.input-group { display: flex; flex-direction: column; gap: 0.5rem; flex: 1; min-width: 150px;}
.input-group label { font-size: 0.85rem; color: var(--color-text-secondary); }

.multiselect {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid var(--glass-border);
  border-radius: 6px;
  color: var(--color-text-primary);
  padding: 0.5rem;
}
.multiselect option {
  padding: 0.5rem;
  border-radius: 4px;
}
.multiselect option:checked {
  background: rgba(0, 229, 255, 0.2);
  color: var(--color-teal-cyan);
}

.provider-card {
  margin-bottom: 2rem;
  overflow: hidden;
}

.provider-header {
  padding: 1.5rem;
  background: rgba(0, 0, 0, 0.72);
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

.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-content {
  background: rgba(12,14,16,0.98);
  padding: 2rem;
  width: 500px;
  max-width: 90vw;
  border-radius: 12px;
}
.modal-content h3 { margin-top: 0; color: var(--color-text-primary); }
</style>
