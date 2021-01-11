<script>
export default {
  methods: {
    /**
     * Get the car data reduced to just the variables we are interested
     * and cleaned of missing data.
     */
    async getData() {
      const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
      const carsData = await carsDataResponse.json();
      const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
      }))
          .filter(car => (car.mpg != null && car.horsepower != null));

      return cleaned;
    },

    async run() {
      // Load and plot the original input data that we are going to train on.
      const data = await this.getData();
      const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
      }));

      tfvis.render.scatterplot(
          { name: 'Horsepower v MPG' },
          { values },
          {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300,
          },
      );

      // More code will be added below
    },
  },
  mounted() {
    this.run();
  },

}
</script>
<template>


</template>
