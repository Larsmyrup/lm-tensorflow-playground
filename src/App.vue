<template>
  <v-app>
    <v-app-bar
        class="app-bar"
        app
        color="white"
        elevate-on-scroll
    >
      <v-btn
          text
          :to="{name: 'Home'}"
      >
        Home
      </v-btn>

      <v-spacer/>

      <v-btn
          text
          :to="{name: 'TwoDimensionalPrediction'}"
      >
        Two Dimensional Prediction
      </v-btn>
      <v-btn
          text
          :to="{name: 'HandwrittenDigits'}"
      >
        Handwritten Digits
      </v-btn>
    </v-app-bar>
    <v-main>
      <router-view/>
    </v-main>
  </v-app>
</template>

<style lang="scss">
.app-bar {
  &::after {
    content: "";
    display: block;
    border-bottom: 1px solid rgba(0,0,0,0.12);
    width: 100%;
    position: relative;
  }
}
</style>
