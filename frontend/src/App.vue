<template>
  <router-view v-slot="{ Component }">
    <transition name="fade" mode="out-in">
      <component :is="Component" />
    </transition>
  </router-view>
</template>

<script setup>
import { RouterView } from 'vue-router'
</script>

<style>
/* Global App Styles (Transitions handled in style.css) */
</style>
