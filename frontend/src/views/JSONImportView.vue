<template>
  <div class="management-view">
    <div class="view-header">
      <h2>JSON Config Import Studio</h2>
      <button @click="$router.push('/providers')" class="btn-secondary">Back to Providers</button>
    </div>

    <div class="glass-panel" style="padding: 1.5rem;">
      <p class="text-muted" style="margin-top: 0;">
        Paste your existing <code>openAI-example.json</code> or similar JSON routing payload here to bootstrap the SQLite DB.
      </p>

      <div class="split-pane">
        <div class="pane">
          <label>JSON Configuration Payload</label>
          <textarea 
            v-model="jsonPayload" 
            class="json-editor"
            placeholder='{
  "my-provider": {
    "api_key_variable": "MY_KEY",
    "models": { ... }
  }
}'
          ></textarea>
        </div>
      </div>

      <div class="action-bar" style="margin-top: 1.5rem; display: flex; justify-content: flex-end; gap: 1rem; align-items: center;">
        <span v-if="error" style="color: var(--color-danger)">{{ error }}</span>
        <span v-if="success" style="color: var(--color-success)">{{ success }}</span>
        <button @click="applyImport" class="btn-primary" :disabled="importing || !jsonPayload">
          {{ importing ? 'Importing...' : 'Apply Import' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { fetchApi } from '../api'

const jsonPayload = ref('')
const importing = ref(false)
const error = ref('')
const success = ref('')

async function applyImport() {
  error.value = ''
  success.value = ''
  let parsed = null
  
  try {
    parsed = JSON.parse(jsonPayload.value)
  } catch (e) {
    error.value = "Invalid JSON format: " + e.message
    return
  }

  importing.value = true
  try {
    const res = await fetchApi('/config/import', {
      method: 'POST',
      body: JSON.stringify({ config_json: parsed })
    })
    success.value = `Import successful! Added ${res.imported.providers} Providers, ${res.imported.models} Models, and ${res.imported.endpoints} Endpoints.`
    jsonPayload.value = ''
  } catch (e) {
    error.value = 'Import failed: ' + e.message
  } finally {
    importing.value = false
  }
}
</script>

<style scoped>
.view-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.view-header h2 { margin: 0; color: var(--color-text-primary); }

.split-pane {
  display: flex;
  gap: 1.5rem;
  height: 50vh;
}

.pane {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.pane label {
  font-size: 0.9rem;
  color: var(--color-teal-cyan);
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.json-editor {
  flex: 1;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--glass-border);
  border-radius: 8px;
  color: var(--color-text-primary);
  font-family: monospace;
  padding: 1rem;
  resize: none;
  font-size: 0.85rem;
  line-height: 1.4;
}

.json-editor:focus {
  outline: none;
  border-color: var(--color-teal-cyan);
}
</style>
