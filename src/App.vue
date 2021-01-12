<script>
import TwoDimensionalPrediction from '@/two-dimensional-prediction/TwoDimensionalPrediction';

export default {

  components: {
    TwoDimensionalPrediction,
  },

  data: () => ({
    //
  }),
};
</script>

<template>
  <v-app>
    <v-app-bar
        app
        color="primary"
    >

    </v-app-bar>
    <v-main>
      <two-dimensional-prediction/>
    </v-main>
  </v-app>
</template>
