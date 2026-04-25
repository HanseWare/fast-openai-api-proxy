import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/account',
      name: 'account',
      component: () => import('../views/AccountView.vue')
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/LoginView.vue')
    },
    {
      path: '/',
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
    next({ name: 'login' })
  } else if (to.name === 'login' && authStore.isAuthenticated) {
    next({ name: 'dashboard' })
  } else {
    next()
  }
})

export default router
