<template>
  <div class="management-view">
    <div class="view-header">
      <h2>Budget Management</h2>
      <button @click="showCreate = !showCreate" class="btn-primary">
        {{ showCreate ? 'Cancel' : '+ Create Budget' }}
      </button>
    </div>

    <!-- Create Form -->
    <transition name="slide-up">
      <div v-if="showCreate" class="glass-panel form-panel">
        <h3>Create New Budget</h3>
        <form @submit.prevent="createBudget" class="inline-form">
          <div class="input-group">
            <label>Entity Type</label>
            <select v-model="newBudget.entity_type" required>
              <option value="user">User</option>
              <option value="group">Group</option>
            </select>
          </div>
          <div class="input-group">
            <label>Entity ID</label>
            <input v-model="newBudget.entity_id" required placeholder="e.g. user_123" />
          </div>
          <div class="input-group">
            <label>Model Type Scope</label>
            <select v-model="newBudget.model_type">
              <option value="">All Types</option>
              <option value="llm">LLM (Text)</option>
              <option value="embedding">Embedding</option>
              <option value="image">Image</option>
              <option value="audio_transcription">Audio Transcription</option>
              <option value="audio_speech">Audio Speech</option>
            </select>
          </div>
          <div class="input-group">
            <label>Window</label>
            <select v-model="newBudget.window" required>
              <option value="daily">Daily</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>
          <div class="input-group">
            <label>Budget Amount</label>
            <input type="number" step="0.0001" min="0" v-model.number="newBudget.budget_amount" required placeholder="e.g. 100.0" />
          </div>
          <button type="submit" class="btn-primary" :disabled="creating">Create</button>
        </form>
      </div>
    </transition>

    <!-- Data Table -->
    <div class="glass-panel table-container">
      <table v-if="budgets.length > 0">
        <thead>
          <tr>
            <th>Entity Type</th>
            <th>Entity ID</th>
            <th>Model Scope</th>
            <th>Window</th>
            <th>Budget Amount</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="b in budgets" :key="b.id">
            <td style="text-transform: capitalize;">{{ b.entity_type }}</td>
            <td class="mono">{{ b.entity_id }}</td>
            <td>
              <span v-if="b.model_type" class="status-pill status-pill--success">{{ b.model_type }}</span>
              <span v-else class="status-pill status-pill--neutral">All</span>
            </td>
            <td style="text-transform: capitalize;">{{ b.window }}</td>
            <td>
              <div v-if="editingId === b.id" class="edit-mode">
                <input type="number" step="0.0001" min="0" v-model.number="editAmount" class="inline-input" />
                <button @click="saveEdit(b.id)" class="btn-icon">✅</button>
                <button @click="cancelEdit" class="btn-icon">❌</button>
              </div>
              <div v-else class="view-mode">
                {{ b.budget_amount }}
                <button @click="startEdit(b)" class="btn-icon edit-btn" title="Edit Amount">✏️</button>
              </div>
            </td>
            <td>
              <button @click="deleteBudget(b.id)" class="btn-icon delete" title="Delete Budget">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">
        <p>No budgets found.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { fetchApi } from '../api'

const budgets = ref([])
const showCreate = ref(false)
const creating = ref(false)

const newBudget = ref({
  entity_type: 'user',
  entity_id: '',
  model_type: '',
  window: 'daily',
  budget_amount: null
})

const editingId = ref(null)
const editAmount = ref(0)

async function loadBudgets() {
  try {
    budgets.value = await fetchApi('/budgets')
  } catch (e) {
    alert('Failed to load budgets: ' + e.message)
  }
}

async function createBudget() {
  creating.value = true
  try {
    const payload = {
      entity_type: newBudget.value.entity_type,
      entity_id: newBudget.value.entity_id,
      window: newBudget.value.window,
      budget_amount: newBudget.value.budget_amount,
    }
    if (newBudget.value.model_type) {
      payload.model_type = newBudget.value.model_type
    } else {
      payload.model_type = null
    }

    await fetchApi('/budgets', {
      method: 'POST',
      body: JSON.stringify(payload)
    })

    newBudget.value.entity_id = ''
    newBudget.value.budget_amount = null
    showCreate.value = false
    await loadBudgets()
  } catch (e) {
    alert('Failed to create budget: ' + e.message)
  } finally {
    creating.value = false
  }
}

function startEdit(budget) {
  editingId.value = budget.id
  editAmount.value = budget.budget_amount
}

function cancelEdit() {
  editingId.value = null
  editAmount.value = 0
}

async function saveEdit(id) {
  try {
    await fetchApi(`/budgets/${id}`, {
      method: 'PUT',
      body: JSON.stringify({ budget_amount: editAmount.value })
    })
    cancelEdit()
    await loadBudgets()
  } catch (e) {
    alert('Failed to update budget: ' + e.message)
  }
}

async function deleteBudget(id) {
  if (!confirm('Are you sure you want to delete this budget?')) return
  try {
    await fetchApi(`/budgets/${id}`, { method: 'DELETE' })
    await loadBudgets()
  } catch (e) {
    alert('Failed to delete budget: ' + e.message)
  }
}

onMounted(() => {
  loadBudgets()
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
  flex-wrap: wrap;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  flex: 1;
  min-width: 150px;
}

.input-group label {
  font-size: 0.85rem;
  color: var(--color-text-secondary);
}

.input-group input, .input-group select {
  width: 100%;
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
  color: inherit;
}

.btn-icon:hover {
  background: rgba(255, 255, 255, 0.1);
}

.btn-icon.delete:hover {
  background: rgba(239, 68, 68, 0.2);
}

.edit-mode {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.view-mode {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.edit-btn {
  opacity: 0.3;
  padding: 0.25rem;
}

tr:hover .edit-btn {
  opacity: 1;
}

.inline-input {
  width: 100px;
  padding: 0.25rem 0.5rem;
}

.empty-state {
  padding: 3rem;
  text-align: center;
  color: var(--color-text-muted);
}

.status-pill {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-pill--success {
  background: rgba(16, 185, 129, 0.1);
  color: var(--color-success);
}

.status-pill--neutral {
  background: rgba(255, 255, 255, 0.1);
  color: var(--color-text-secondary);
}
</style>
