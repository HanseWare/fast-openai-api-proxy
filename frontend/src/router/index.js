import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'account',
      component: () => import('../views/AccountView.vue')
    },
    {
      path: '/oidc-callback',
      name: 'oidc-callback',
      component: () => import('../views/OidcCallbackView.vue')
    },
    {
      path: '/admin/login',
      name: 'admin-login',
      component: () => import('../views/LoginView.vue')
    },
    {
      path: '/admin',
      component: () => import('../views/DashboardLayout.vue'),
      meta: { requiresAuth: true },
      children: [
        {
          path: '',
          name: 'dashboard',
          component: () => import('../views/DashboardView.vue')
        },
        {
          path: 'keys',
          name: 'keys',
          component: () => import('../views/KeysView.vue')
        },
        {
          path: 'endpoints',
          name: 'endpoints',
          component: () => import('../views/EndpointsView.vue')
        },
        {
          path: 'quotas',
          name: 'quotas',
          component: () => import('../views/QuotasView.vue')
        },
        {
          path: 'providers',
          name: 'providers',
          component: () => import('../views/ProvidersView.vue')
        },
        {
          path: 'aliases',
          name: 'aliases',
          component: () => import('../views/AliasesView.vue')
        },
        {
          path: 'import',
          name: 'import',
          component: () => import('../views/JSONImportView.vue')
        }
      ]
    }
  ]
})

router.beforeEach((to, from, next) => {
  const authStore = useAuthStore()
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    next({ name: 'admin-login' })
  } else if (to.name === 'admin-login' && authStore.isAuthenticated) {
    next({ name: 'dashboard' })
  } else {
    next()
  }
})

export default router
