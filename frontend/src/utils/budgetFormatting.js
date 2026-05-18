const TOKEN_RATE_SCALE = 1_000_000

const TOKEN_PRICED_MODEL_TYPES = new Set(['llm', 'embedding'])
const AUDIO_MODEL_TYPES = new Set(['audio_transcription', 'audio_speech', 'stt', 'tts'])
const IMAGE_MODEL_TYPES = new Set(['image', 'image-gen'])

function normalizeValue(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function formatNumber(value, maximumFractionDigits = 2) {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  }).format(normalizeValue(value))
}

export function isTokenPricedModelType(modelType) {
  return TOKEN_PRICED_MODEL_TYPES.has((modelType || '').trim())
}

export function formatCredits(value, maximumFractionDigits = 2) {
  return formatNumber(value, maximumFractionDigits)
}

export function formatPricePerDisplayUnit(pricePerUnit, modelType) {
  const normalized = normalizeValue(pricePerUnit)
  if (isTokenPricedModelType(modelType)) {
    return `${formatNumber(normalized * TOKEN_RATE_SCALE, 4)} credits / 1M tokens`
  }
  return `${formatNumber(normalized, 4)} credits / unit`
}

export function toBackendPrice(priceDisplayValue, modelType) {
  const normalized = normalizeValue(priceDisplayValue)
  return isTokenPricedModelType(modelType) ? normalized / TOKEN_RATE_SCALE : normalized
}

export function fromBackendPrice(pricePerUnit, modelType) {
  const normalized = normalizeValue(pricePerUnit)
  return isTokenPricedModelType(modelType) ? normalized * TOKEN_RATE_SCALE : normalized
}

export function formatBudgetTypeLabel(type) {
  switch (type) {
    case 'tokens':
      return 'Tokens'
    case 'audio':
      return 'Audio'
    case 'images':
      return 'Images'
    default:
      return type ? String(type) : 'Budget'
  }
}

export function formatBudgetWindowLabel(window) {
  switch (window) {
    case 'daily':
      return 'Daily'
    case 'monthly':
      return 'Monthly'
    default:
      return window ? String(window) : 'Window'
  }
}

export function formatBudgetScopeLabel(scope) {
  const normalized = (scope || '').trim()
  if (!normalized) {
    return 'Global'
  }

  if (TOKEN_PRICED_MODEL_TYPES.has(normalized)) {
    return 'Tokens'
  }
  if (AUDIO_MODEL_TYPES.has(normalized)) {
    return 'Audio'
  }
  if (IMAGE_MODEL_TYPES.has(normalized)) {
    return 'Images'
  }
  return normalized
}

export function getCurrentBucket(window, timestamp = Date.now()) {
  const dt = new Date(timestamp)
  const year = dt.getFullYear()
  const month = String(dt.getMonth() + 1).padStart(2, '0')
  const day = String(dt.getDate()).padStart(2, '0')

  if (window === 'daily') {
    return `${year}-${month}-${day}`
  }

  return `${year}-${month}`
}

