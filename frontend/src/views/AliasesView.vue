<template>
  <div class="management-view">
    <div class="view-header">
      <h2>Virtual Models (Aliases)</h2>
      <button @click="showCreate = !showCreate" class="btn-primary">
        {{ showCreate ? 'Cancel' : '+ New Virtual Model' }}
      </button>
    </div>

    <!-- Create Form -->
    <transition name="slide-up">
      <div v-if="showCreate" class="glass-panel form-panel">
        <h3>Create Virtual Model Alias</h3>
        <p class="text-muted" style="margin-bottom: 1rem;">Link a generic name to a specific upstream model (e.g. <code>chat-large</code> -> <code>gpt-4o</code>).</p>
        <form @submit.prevent="createAlias" class="inline-form" style="flex-wrap: wrap;">
          <div class="input-group">
            <label>Virtual Model Name</label>
            <input v-model="newAlias.alias_name" required placeholder="chat-large" />
          </div>
          <div class="input-group">
            <label>Target Upstream Model Name</label>
            <input v-model="newAlias.target_model_name" required placeholder="gpt-4o" />
          </div>
          <div class="input-group">
            <label>Owned By</label>
            <input v-model="newAlias.owned_by" placeholder="FOAP" />
          </div>
          <div class="input-group" style="flex-direction: row; align-items: center; gap: 0.5rem; min-width: auto; flex: 0;">
            <input type="checkbox" v-model="newAlias.hide_on_models_endpoint" id="new_alias_hide" />
            <label for="new_alias_hide" style="margin: 0; white-space: nowrap;">Hide on /v1/models</label>
          </div>
          <button type="submit" class="btn-primary" :disabled="creating">Save Alias</button>
        </form>
      </div>
    </transition>

    <!-- Data Table -->
    <div class="glass-panel table-container">
      <table v-if="aliases.length > 0">
        <thead>
          <tr>
            <th>Virtual Model (Alias)</th>
            <th>Maps To (Target)</th>
            <th>Owned By</th>
            <th>Hidden</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="a in aliases" :key="a.id">
            <td class="mono"><strong>{{ a.alias_name }}</strong></td>
            <td class="mono" style="color: var(--color-teal-cyan)">{{ a.target_model_name }}</td>
            <td>{{ a.owned_by || 'FOAP' }}</td>
            <td>{{ a.hide_on_models_endpoint ? '🚫' : '—' }}</td>
            <td>
              <button @click="openEditAlias(a)" class="btn-icon" title="Edit Alias">✏️</button>
              <button @click="deleteAlias(a.id)" class="btn-icon delete" title="Delete Alias">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">
        <p>No virtual models configured.</p>
      </div>
    </div>

    <!-- Edit Alias Modal -->
    <div v-if="editForm.show" class="modal-overlay">
      <div class="glass-panel modal-content">
        <h3>Edit Virtual Model</h3>
        <form @submit.prevent="submitEditAlias">
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Virtual Model Name</label>
            <input v-model="editForm.alias_name" required />
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Target Upstream Model</label>
            <input v-model="editForm.target_model_name" required />
          </div>
          <div class="input-group" style="margin-bottom: 1rem;">
            <label>Owned By</label>
            <input v-model="editForm.owned_by" placeholder="FOAP" />
          </div>
          <div class="input-group" style="flex-direction: row; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;">
            <input type="checkbox" v-model="editForm.hide_on_models_endpoint" id="edit_alias_hide" />
            <label for="edit_alias_hide" style="margin: 0;">Hide on /v1/models</label>
          </div>
          <div style="display: flex; gap: 1rem; justify-content: flex-end;">
            <button type="button" class="btn-secondary" @click="editForm.show = false">Cancel</button>
            <button type="submit" class="btn-primary" :disabled="creating">Update</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const aliases = ref([])
const showCreate = ref(false)
const creating = ref(false)

const newAlias = ref({
  alias_name: '',
  target_model_name: '',
  owned_by: 'FOAP',
  hide_on_models_endpoint: false
})

const editForm = ref({
  show: false,
  id: null,
  alias_name: '',
  target_model_name: '',
  owned_by: 'FOAP',
  hide_on_models_endpoint: false
})

async function loadAliases() {
  try {
    aliases.value = await fetchApi('/config/aliases')
  } catch (e) {
    alert('Failed to load virtual models: ' + e.message)
  }
}

async function createAlias() {
  creating.value = true
  try {
    await fetchApi('/config/aliases', {
      method: 'POST',
      body: JSON.stringify(newAlias.value)
    })
    newAlias.value = { alias_name: '', target_model_name: '', owned_by: 'FOAP', hide_on_models_endpoint: false }
    await loadAliases()
    showCreate.value = false
  } catch (e) {
    alert('Failed to add alias: ' + e.message)
  } finally {
    creating.value = false
  }
}

function openEditAlias(a) {
  editForm.value = {
    show: true,
    id: a.id,
    alias_name: a.alias_name,
    target_model_name: a.target_model_name,
    owned_by: a.owned_by || 'FOAP',
    hide_on_models_endpoint: !!a.hide_on_models_endpoint
  }
}

async function submitEditAlias() {
  creating.value = true
  try {
    await fetchApi(`/config/aliases/${editForm.value.id}`, {
      method: 'PUT',
      body: JSON.stringify({
        alias_name: editForm.value.alias_name,
        target_model_name: editForm.value.target_model_name,
        owned_by: editForm.value.owned_by,
        hide_on_models_endpoint: editForm.value.hide_on_models_endpoint
      })
    })
    editForm.value.show = false
    await loadAliases()
  } catch (e) {
    alert('Failed to update alias: ' + e.message)
  } finally {
    creating.value = false
  }
}

async function deleteAlias(id) {
  if (!confirm('Are you sure you want to remove this virtual model mapping?')) return
  try {
    await fetchApi(`/config/aliases/${id}`, { method: 'DELETE' })
    await loadAliases()
  } catch (e) {
    alert('Failed to delete alias: ' + e.message)
  }
}

onMounted(() => {
  loadAliases()
})
</script>

<style scoped>
.view-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.view-header h2 { margin: 0; color: var(--color-text-primary); }

.form-panel { padding: 1.5rem; margin-bottom: 2rem; }
.form-panel h3 { margin-top: 0; margin-bottom: 0.5rem; font-size: 1.1rem; }
.inline-form { display: flex; gap: 1rem; align-items: flex-end; }
.input-group { display: flex; flex-direction: column; gap: 0.5rem; flex: 1; }
.input-group label { font-size: 0.85rem; color: var(--color-text-secondary); }

.table-container { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 1rem; text-align: left; border-bottom: 1px solid var(--glass-border); }
th { color: var(--color-text-secondary); font-weight: 500; font-size: 0.9rem; }
tbody tr:hover { background: rgba(255, 255, 255, 0.02); }

.mono { font-family: monospace; }
.btn-icon { background: transparent; border: none; cursor: pointer; padding: 0.5rem; border-radius: 4px; }
.btn-icon.delete:hover { background: rgba(239, 68, 68, 0.2); }
.empty-state { padding: 3rem; text-align: center; color: var(--color-text-muted); }
.text-muted { color: var(--color-text-muted); font-size: 0.9rem; }

.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.modal-content {
  width: 100%;
  max-width: 480px;
  padding: 2rem;
}
</style>
