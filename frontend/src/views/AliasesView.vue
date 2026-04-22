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
        <form @submit.prevent="createAlias" class="inline-form">
          <div class="input-group">
            <label>Virtual Model Name</label>
            <input v-model="newAlias.alias_name" required placeholder="chat-large" />
          </div>
          <div class="input-group">
            <label>Target Upstream Model Name</label>
            <input v-model="newAlias.target_model_name" required placeholder="gpt-4o" />
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
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="a in aliases" :key="a.id">
            <td class="mono"><strong>{{ a.alias_name }}</strong></td>
            <td class="mono" style="color: var(--color-teal-cyan)">{{ a.target_model_name }}</td>
            <td>
              <button @click="deleteAlias(a.id)" class="btn-icon delete" title="Delete Alias">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">
        <p>No virtual models configured.</p>
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
  target_model_name: ''
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
    newAlias.value.alias_name = ''
    newAlias.value.target_model_name = ''
    await loadAliases()
    showCreate.value = false
  } catch (e) {
    alert('Failed to add alias: ' + e.message)
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
</style>
