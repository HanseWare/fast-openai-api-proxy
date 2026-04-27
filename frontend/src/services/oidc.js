import { UserManager, WebStorageStateStore } from 'oidc-client-ts'

let _userManager = null

/**
 * Initialise or return the cached UserManager instance.
 *
 * @param {Object} oidcClient - The oidc_client block from /auth-config
 * @param {string} oidcClient.authority - OIDC issuer URL
 * @param {string} oidcClient.client_id - OIDC client ID
 * @param {string} redirectPath - The path to redirect to after login (default: /oidc-callback)
 */
export function getOidcManager(oidcClient, redirectPath = '/oidc-callback') {
  if (_userManager) return _userManager

  _userManager = new UserManager({
    authority: oidcClient.authority,
    client_id: oidcClient.client_id,
    redirect_uri: `${window.location.origin}${redirectPath}`,
    response_type: 'code',
    scope: 'openid profile email',
    userStore: new WebStorageStateStore({ store: window.sessionStorage }),
    automaticSilentRenew: false,
  })

  return _userManager
}

/**
 * Start the OIDC login redirect.
 * Saves a `foap_oidc_target` key so the callback knows where to go afterwards.
 *
 * @param {Object} oidcClient
 * @param {string} target - 'admin' or 'account'
 */
export async function startOidcLogin(oidcClient, target = 'admin') {
  const mgr = getOidcManager(oidcClient)
  sessionStorage.setItem('foap_oidc_target', target)
  await mgr.signinRedirect()
}

/**
 * Complete the OIDC login by processing the callback URL.
 * Returns the access_token string on success, or null on failure.
 */
export async function completeOidcLogin(oidcClient) {
  const mgr = getOidcManager(oidcClient)
  try {
    const user = await mgr.signinCallback()
    return user?.access_token || null
  } catch (err) {
    console.error('OIDC callback error:', err)
    return null
  }
}

/**
 * Reset the cached UserManager (e.g. on logout).
 */
export function resetOidcManager() {
  _userManager = null
}
