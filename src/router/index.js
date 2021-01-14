import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '@/pages/Home.vue'
import TwoDimensionalPrediction from '@/pages/TwoDimensionalPrediction';
import HandwrittenDigits from '@/pages/HandwrittenDigits';

Vue.use(VueRouter)

const routes = [
  {
    path: '',
    redirect: { name: 'Home' },
  },
  {
    path: '/home',
    name: 'Home',
    component: Home,
  },
  {
    path: '/two-dimensional-prediction',
    name: 'TwoDimensionalPrediction',
    component: TwoDimensionalPrediction,
  },
  {
    path: '/handwritten-digits',
    name: 'HandwrittenDigits',
    component: HandwrittenDigits,
  },
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
})

export default router
