const { log } = require("console");
const data = require("./data.json");
const fs = require("fs");

let d = {
  X: [],
  y: [],
};

function numToBinary(number) {
  let binary = [];
  for (let i = 0; i < 10; i++) {
    binary.push(number === i ? 1 : 0);
  }
  return binary;
}
function binaryToNumber(binaryArray) {
  let decimalNumber = 0;

  // Iterate over each bit in the binary array
  for (let i = 0; i < binaryArray.length; i++) {
    // Multiply the current bit by the corresponding power of 2 and add it to the decimal number
    decimalNumber += binaryArray[i] * Math.pow(2, binaryArray.length - 2 - i);
  }

  return decimalNumber;
}

const Sigmoid = () => {
  return (x) => {
    return 1 / (1 + Math.exp(-x));
  };
};

class Neuron {
  constructor(options) {
    this.id = options.id;
    this.inputs = [];
    this.outputs = [];
    this.raw = null;
    this.output = null;
    this.is_bias = options && options.bias ? true : false;

    this.activationFunc = function (x) {
      return 1 / (1 + Math.exp(-x));
    };
  }
  activate() {
    this.raw = 0;

    if (this.is_bias) {
      this.output = -1;
    } else {
      for (var i = 0; i < this.inputs.length; i++) {
        var input = this.inputs[i];
        this.raw += input.from.output * input.weight;
      }

      this.output = sigmoid(this.raw);
    }
  }

  add_input(input) {
    this.inputs.push(input);
  }

  add_output(output) {
    this.outputs.push(output);
  }
}
class Connection {
  constructor(from, to, weight) {
    this.weight = weight || this.random_weight();
    this.from = from;
    this.to = to;
  }
  random_weight() {
    return Math.random() * 2 - 1;
    //  * 0.0001;
  }
}
class NeuralNetwork {
  constructor(size, data, options) {
    if (!options) {
      options = {};
    }

    this.size = size;
    this.data = data;

    this.neuronId = 0;

    this.layers = [];
    this.iterations = options.iterations || 1000;
    this.alpha = options.alpha || 0.1;

    this.weights = [];
    this.biases = [];

    this.init();
  }
  init() {
    for (var i = 0; i < this.size.length; i++) {
      var layer = [];

      if (i !== this.size.length - 1) {
        layer.push(new Neuron({ id: this.neuronId++, bias: true }));

        this.biases.push(layer[0]);
      }

      for (var j = 0; j < this.size[i]; j++) {
        layer.push(new Neuron({ id: this.neuronId++ }));
      }

      this.layers.push(layer);
    }

    // For each layer from the second-last to the second layer...
    for (var i = 1; i < this.size.length; i++) {
      var from_layer = this.layers[i - 1];
      var to_layer = this.layers[i];

      // For each neuron in the from layer...
      for (var j = 0; j < from_layer.length; j++) {
        // Determine the starting point for the to layer.
        // If the to layer is the last layer, there is no need to start at the bias unit.
        // Otherwise, start at the bias unit.
        var start = i === this.size.length - 1 ? 0 : 1;

        // For each neuron in the to layer...
        for (var k = start; k < to_layer.length; k++) {
          // Get the "from" neuron.
          var from = from_layer[j];

          // Get the "to" neuron.
          var to = to_layer[k];

          // Set the weight for the connection.
          var weight = j === 0 ? 1 : null; // non-random weights for bias units

          // Create the connection.
          var connection = new Connection(from, to, weight);

          // Add the connection to the "from" neuron's outputs.
          from.add_output(connection);

          // Add the connection to the "to" neuron's inputs.
          to.add_input(connection);
        }
      }
    }
  }
  forwardPropagate() {
    for (var i = 1; i < this.layers.length; i++) {
      var layer = this.layers[i];

      for (var j = 0; j < layer.length; j++) {
        var neuron = layer[j];
        neuron.activate();
      }
    }
  }

  backPropagate() {
    // delta[j] := g_prime(input[j]) * (y - a)

    var deltas = [];

    var output_layer = this.layers[this.layers.length - 1];

    for (var i = 0; i < output_layer.length; i++) {
      var neuron = output_layer[i];
      var g_prime = neuron.output * (1 - neuron.output);
      var error = this.expected_output[i] - neuron.output;
      deltas[neuron.id] = g_prime * error;
      this.cost += Math.abs(error);
    }

    // For each output neuron, calculate its delta value.
    for (var i = this.layers.length - 2; i >= 1; i--) {
      var layer = this.layers[i];

      for (var j = 0; j < layer.length; j++) {
        var neuron = layer[j];

        // delta[i] := g_prime(output[i]) * sum(weights[i][j] * delta[j])
        var g_prime = neuron.output * (1 - neuron.output);
        var sum = 0;

        for (var k = 0; k < neuron.outputs.length; k++) {
          var out_connection = neuron.outputs[k];
          var out_neuron = out_connection.to;
          var out_delta = deltas[out_neuron.id];

          sum += out_connection.weight * out_delta;
        }

        deltas[neuron.id] = g_prime * sum;
      }
    }

    // Adjust each weight based on its input and the neuron's delta.
    for (var i = 1; i < this.layers.length; i++) {
      var layer = this.layers[i];

      for (var j = 0; j < layer.length; j++) {
        var neuron = layer[j];

        for (var k = 0; k < neuron.inputs.length; k++) {
          var input = neuron.inputs[k];
          input.weight =
            input.weight + this.alpha * input.from.output * deltas[neuron.id];
        }
      }
    }
  }

  runOnce() {
    var input_layer = this.layers[0];

    for (var i = 0; i < this.data.X.length; i++) {
      this.training_example = this.data.X[i];
      this.expected_output = this.data.y[i];

      for (var j = 1; j < input_layer.length; j++) {
        input_layer[j].output = this.training_example[j - 1];
      }

      this.forwardPropagate();
      this.backPropagate();
    }
  }

  predict(input) {
    var input_layer = this.layers[0];

    for (var i = 1; i < input_layer.length; i++) {
      input_layer[i].output = input[i - 1];
    }

    this.forwardPropagate();

    var output = this.layers[this.layers.length - 1];
    var vals = [];

    for (var i = 0; i < output.length; i++) {
      vals[i] = output[i].output;
    }

    return vals;
  }

  train(threshold) {
    for (var i = 0; i < this.iterations; i++) {
      if (this.cost <= threshold) break;
      this.cost = 0;

      this.runOnce();

      console.log("Iteration:", i + 1, " Loss:", this.cost);
    }

    console.log("Training complete");
  }
  getWeights() {
    const weights = {};

    nn.layers.forEach((layer, layerIndex) => {
      // if(layerIndex === 0) return;
      layer.forEach((neuron, j) => {
        const neuronId = neuron.id;

        // Make sure the Neuron ID exists in our map
        if (!weights[neuronId]) {
          weights[neuronId] = {
            id: neuronId,
            layer: layerIndex,
            weights: [],
          };
        }

        // Set Input Connections
        neuron.inputs.forEach((input, k) => {
          weights[neuronId].weights.push({
            id: input.from.id,
            isInput: true,
            weight: input.weight,
          });
        });

        // Set Outputs Connections
        neuron.outputs.forEach((output, k) => {
          weights[neuronId].weights.push({
            id: output.to.id,
            isInput: false,
            weight: output.weight,
          });
        });
      });
    });
    return weights;
  }
}

for (let i = 0; i < 10; i++) {
  d.X.push(data["data"][i]);
  d.y.push(numToBinary(data["labels"][i]));
}

var nn = new NeuralNetwork([4, 5, 5, 10], d, {
  iterations: 100000,
});
// log("Training...");
let threshold = 0.5;
// nn.train(threshold);

// use cycle.js to save the nn
const cycle = require("./cycle.js");
const sigmoid = require("./nn/ActivationFunctions");

// console.log(
//   nn.predict([
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
//     0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0, 0, 0, 0, 0,
//   ])
// );


console.log("Saving...");

console.log(nn.layers[0]);

// fs.writeFileSync("network.json", JSON.stringify(cycle.decycle(nn)));

// // load the network
// const nnData = cycle.retrocycle(JSON.parse(fs.readFileSync("network.json")));
// const nn2 = new NeuralNetwork(nnData.size, nnData.data);
// nn2.fromJSON(nnData);
// nn2.train(threshold);

// nn.train(threshold);
const weights = nn.getWeights();
// console.log(nn.getWeights());

function setWeights(weights) {
  const layers = {};
  Object.values(weights).forEach(weightedNueron => {
    let nWeights = weightedNueron.weights;
    let nLayer   = weightedNueron.layer;
    let nId      = weightedNueron.id;

    if (!layers[weightedNueron.layer]) {
      layers[weightedNueron.layer] = [];
    }
    
    // Create new Neuron
    const neuron = new Neuron({ id: nId });
    neuron.outputs = [];

    // Add weights to the Neuron
    nWeights.forEach((weight, i) => {
      let from = new Neuron({ id: nId });
      let to = new Neuron({ id: weight.id });
      let con = new Connection(from, to);
      con.weight = weight.weight;
      neuron.outputs.push(con);
    });

    // Add this Neuron to the layer
    const currentLayer = layers[weightedNueron.layer];
    currentLayer.push(neuron);
  })
  return layers;;
}
// setWeights(weights);

const test = setWeights(weights);
console.log(test);
nn.layers = test;

nn.train()