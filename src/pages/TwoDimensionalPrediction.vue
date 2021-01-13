<script>
import * as tfvis from "@tensorflow/tfjs-vis";
import { sequential, layers, tidy, util, train, losses, tensor2d, linspace } from '@tensorflow/tfjs';
import { cloneDeep } from 'lodash';

const defaultDraftLayer = {
  name: null,
  activation: null,
  units: null,
};

export default {
  data() {
    return {
      rawData: null,
      epochs: 10,
      batchSize: 32,
      draftLayer: cloneDeep(defaultDraftLayer),
      activationOptions: [
        {
          value: 'elu',
          name: 'elu',
        },
        {
          value: 'hard_sigmoid',
          name: 'hard_sigmoid',
        },
        {
          value: 'linear',
          name: 'linear',
        },
        {
          value: 'relu',
          name: 'relu',
        },
        {
          value: 'relu6',
          name: 'relu6',
        },
        {
          value: 'selu',
          name: 'selu',
        },
        {
          value: 'sigmoid',
          name: 'sigmoid',
        },
        {
          value: 'softmax',
          name: 'softmax',
        },
        {
          value: 'softplus',
          name: 'softplus',
        },
        {
          value: 'softsign',
          name: 'softsign',
        },
        {
          value: 'tanh',
          name: 'tanh',
        },
      ],
      model: null,
      inputLayer: { inputShape: [1], units: 1, name: 'input_layer' },
      outputLayer: { units: 1, name: 'output_layer' },
      middleLayers: [],
    };
  }
  ,
  mounted() {
    this.getData();
  }
  ,
  methods: {
    /**
     * Get the car data reduced to just the variables we are interested
     * and cleaned of missing data.
     */
    async getData() {
      const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
      const carsData = await carsDataResponse.json();
      this.rawData = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
      })).filter(car => (car.mpg != null && car.horsepower != null));

      await this.plotData();
    },
    async plotData() {
      const values = this.rawData.map(d => ({
        x: d.horsepower,
        y: d.mpg,
      }));
      await tfvis.render.scatterplot(
          document.getElementById('data-container'),
          { values },
          {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300,
          },
      );
    },
    async run() {
      // Create the model
      this.model = this.createModel();
      // Convert the data to a form we can use for training.
      const tensorData = this.convertToTensor(this.rawData);
      const { inputs, labels } = tensorData;

      // Train the model
      await this.trainModel(this.model, inputs, labels);

      this.testModel(this.model, this.rawData, tensorData);
    },
    createModel() {
      this.model = null;
      const model = sequential();

      model.add(layers.dense(this.inputLayer));

      this.middleLayers.forEach((layer) => {
        model.add(layers.dense(layer));
      });

      model.add(layers.dense(this.outputLayer));


      tfvis.show.modelSummary(document.getElementById('model-summary-container'), model);

      return model;
    },
    /**
     * Convert the input data to tensors that we can use for machine
     * learning. We will also do the important best practices of _shuffling_
     * the data and _normalizing_ the data
     * MPG on the y-axis.
     */
    convertToTensor(data) {
      // Wrapping these calculations in a tidy will dispose any
      // intermediate tensors.

      return tidy(() => {
        // Step 1. Shuffle the data
        util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);

        const inputTensor = tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
          inputs: normalizedInputs,
          labels: normalizedLabels,
          // Return the min/max bounds so we can use them later.
          inputMax,
          inputMin,
          labelMax,
          labelMin,
        }
      });
    },
    testModel(model, inputData, normalizationData) {
      const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

      // Generate predictions for a uniform range of numbers between 0 and 1;
      // We un-normalize the data by doing the inverse of the min-max scaling
      // that we did earlier.
      const [xs, predictions] = tidy(() => {

        const xs = linspace(0, 1, 100);
        const predictions = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const unNormPredictions = predictions
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPredictions.dataSync()];
      });


      const predictedPoints = Array.from(xs).map((val, i) => {
        return { x: val, y: predictions[i] }
      });

      const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg,
      }));

      tfvis.render.scatterplot(
          document.getElementById('prediction-container'),
          { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
          {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300,
          },
      );
    },
    onAddDraftToList() {
      this.middleLayers.push(cloneDeep(this.draftLayer));
      this.draftLayer = defaultDraftLayer;
    },
    async trainModel(model, inputs, labels) {
      // Prepare the model for training.
      model.compile({
        optimizer: train.adam(),
        loss: losses.meanSquaredError,
        metrics: ['mse'],
      });

      const batchSize = this.batchSize;
      const epochs = this.epochs;
      return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            document.getElementById('training-performance-container'),
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] },
        ),
      });
    },
  },
}
</script>
<template>
  <v-container>
    <v-row>
      <v-col>
        <h1
            class="mt-8"
        >
          Training Specifications
        </h1>
      </v-col>
    </v-row>
    <v-row>
      <v-col>
        <v-text-field
            class="my-6"
            label="Batch Size"
            hide-details="auto"
            :rules="[
            value => !!value || 'Batch size is required.',
            value => value > 0 || 'Batch size must be a positive whole number.',
        ]"
            v-model="batchSize"
        />
      </v-col>
      <v-col>
        <v-text-field
            class="my-6"
            label="Epochs"
            hide-details="auto"
            :rules="[
            value => !!value || 'Epochs is required.',
            value => value > 0 || 'Epochs must be a positive whole number.',
        ]"
            v-model="epochs"
        />
      </v-col>
    </v-row>
    <v-row>
      <v-col>
        <h1
            class="mt-8"
        >
          Layers
        </h1>
      </v-col>
    </v-row>
    <v-row>
      <v-col>
        <h2
            class="mt-8"
        >
          Input Layer
        </h2>
      </v-col>
    </v-row>

    <v-row>
      <v-col>
        <v-text-field
            label="Name"
            :value="inputLayer.name"
            readonly
            dense
        />
      </v-col>
      <v-col>
        <v-text-field
            label="Units"
            :value="inputLayer.units"
            readonly
            dense
        />
      </v-col>
    </v-row>

    <v-divider
        class="my-10"
    />

    <v-row>
      <v-col>
        <h2
        >
          Middle Layers
        </h2>
      </v-col>
    </v-row>

    <v-row
        v-for="layer in middleLayers"
        :key="layer.name"
    >
      <v-col>
        <v-text-field
            label="Name"
            :value="layer.name"
            dense
            readonly
        />
      </v-col>
      <v-col>
        <v-text-field
            label="Units"
            :value="layer.units"
            type="number"
            dense
            readonly
        />
      </v-col>
      <v-col>
        <v-select
            label="Activation Function"
            :items="activationOptions"
            v-model="layer.activation"
            dense
            item-text="name"
            item-value="value"
            readonly
        />
      </v-col>
      <v-col
          cols="1"
      >
        <v-btn
            icon
        >
          <v-icon>mdi-close-circle-outline</v-icon>
        </v-btn>
      </v-col>
    </v-row>
    <v-row
    >
      <v-col>
        <v-text-field
            label="Name"
            v-model="draftLayer.name"
            dense
            :rules="[
            value => !!value || 'Name is required.',
            value => !middleLayers.indexOf(value) > -1 || 'Name must be unique',
            ]"
        />
      </v-col>
      <v-col>
        <v-text-field
            label="Units"
            v-model.number="draftLayer.units"
            type="number"
            :rules="[
            value => !!value || 'Units are required.',
            value => value > 0 || 'Units must be a positive whole number.'
            ]"
            dense
        />
      </v-col>
      <v-col>
        <v-select
            label="Activation Function"
            :items="activationOptions"
            dense
            item-text="name"
            item-value="value"
            v-model="draftLayer.activation"
        />
      </v-col>
      <v-col
          cols="1"
          @click="onAddDraftToList"
      >
        <v-btn
            icon
        >
          <v-icon>mdi-check-circle-outline</v-icon>
        </v-btn>
      </v-col>
    </v-row>
    <v-divider
        class="my-10"
    />
    <v-row>
      <v-col>
        <h2
        >
          Output Layer
        </h2>
      </v-col>
    </v-row>
    <v-row>
      <v-col>
        <v-text-field
            label="Name"
            :value="outputLayer.name"
            readonly
            dense
        />
      </v-col>
      <v-col>
        <v-text-field
            label="Units"
            :value="outputLayer.units"
            readonly
            dense
        />
      </v-col>
    </v-row>


    <v-btn
        class="my-6"
        :disabled="!rawData"
        color="primary"
        elevation="2"
        large
        block
        @click="run"
    >
      Train

    </v-btn>
    <h1
        class="mx-4 my-6"
    >
      Data
    </h1>
    <div
        id="data-container"
    />
    <h1
        class="mx-4 my-6"
    >
      Model Summary
    </h1>
    <div
        id="model-summary-container"
    />
    <h1
        class="mx-4 my-6"
    >
      Training Performance
    </h1>
    <div
        id="training-performance-container"
    />
    <h1
        class="mx-4 my-6"
    >
      Prediction Data
    </h1>
    <div
        id="prediction-container"
    />
  </v-container>
</template>
