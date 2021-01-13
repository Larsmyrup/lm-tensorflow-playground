<template>
  <v-app>
    <v-app-bar
        app
        color="primary"
    >
      <v-btn
          text
          :to="{name: 'Home'}"
      >
        Home
      </v-btn>

      <v-btn
          text
          :to="{name: 'TwoDimensionalPrediction'}"
      >
        Two Dimensional Prediction
      </v-btn>
    </v-app-bar>
    <v-main>
      <router-view/>
    </v-main>
  </v-app>
</template>
