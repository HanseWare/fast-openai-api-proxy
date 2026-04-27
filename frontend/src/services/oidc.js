/**
 * OIDC Backend-for-Frontend (BFF) service.
 *
 * This service orchestrates OIDC login via the backend BFF pattern:
 * 1. Frontend requests `/api/admin/oidc/login` or `/api/oidc/login`
 * 2. Backend returns authorization URI
 * 3. Frontend redirects user to identity provider
 * 4. User authenticates and is redirected back to backend callback endpoint
 * 5. Backend handles token exchange securely (with client_secret)
 * 6. Backend sets session cookie and redirects to frontend
 */

/**
 * Start the OIDC login flow via backend BFF.
 *
 * @param {string} apiBasePath - '/api/admin' for admin, '/api' for self-service
 * @param {string} target - 'admin' or 'account' (used for error handling)
 * @throws {Error} If backend returns error or network fails
 */
export async function startOidcLogin(apiBasePath = '/api/admin', target = 'admin') {
  // Store the target so we can handle errors appropriately
  sessionStorage.setItem('foap_oidc_target', target)

  try {
    // Request login initiation from backend BFF
    const response = await fetch(`${apiBasePath}/oidc/login`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      credentials: 'include', // Include cookies for session
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `OIDC login failed: ${response.status}`)
    }

    const data = await response.json()
    if (!data.authorization_uri) {
      throw new Error('Backend did not return authorization_uri')
    }

    // Redirect user to identity provider (handled by backend/PKCE)
    window.location.href = data.authorization_uri
  } catch (err) {
    console.error('Failed to start OIDC login:', err)
    throw err
  }
}

