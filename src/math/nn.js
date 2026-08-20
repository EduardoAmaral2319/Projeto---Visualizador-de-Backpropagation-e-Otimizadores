// src/math/nn.js
import { zeros, dot, transpose, add, hadamard, scale, toColumn } from './matrix.js';
import { sigmoid, sigmoidDerivative, softmax, softmaxJacobian } from './activations.js';

// Rede feedforward simples: N camadas totalmente conectadas.
// Última camada usa softmax (classificação de ação: pulo/esquerda/direita).
// Camadas ocultas usam sigmoid.
export class NeuralNetwork {
  constructor(layerSizes) {
    // layerSizes ex: [3, 5, 3] -> 3 entradas, 5 neurônios ocultos, 3 saídas
    this.layerSizes = layerSizes;
    this.weights = []; // W[l]: (layerSizes[l+1] x layerSizes[l])
    this.biases = [];  // b[l]: (layerSizes[l+1] x 1)

    for (let l = 0; l < layerSizes.length - 1; l++) {
      const rows = layerSizes[l + 1], cols = layerSizes[l];
      // Inicialização pequena aleatória, evita saturação inicial da sigmoid
      this.weights.push(
        Array.from({ length: rows }, () =>
          Array.from({ length: cols }, () => (Math.random() * 2 - 1) * 0.5)
        )
      );
      this.biases.push(zeros(rows, 1));
    }
  }

  // Forward pass: guarda z (pré-ativação) e a (pós-ativação) de cada camada,
  // porque o backward precisa deles para a regra da cadeia.
  // Complexidade: soma de dot() em cada camada = O(Σ layerSizes[l]·layerSizes[l+1])
  forward(xVec) {
    const activations = [toColumn(xVec)]; // a^(0) = entrada
    const zs = []; // z^(l) de cada camada

    for (let l = 0; l < this.weights.length; l++) {
      const z = add(dot(this.weights[l], activations[l]), this.biases[l]);
      zs.push(z);

      const isLastLayer = l === this.weights.length - 1;
      const a = isLastLayer ? softmax(z) : sigmoid(z);
      activations.push(a);
    }

    return { activations, zs };
  }

  // Backpropagation: propaga o erro de trás para frente usando a regra da cadeia.
  // yTarget: vetor one-hot da ação do especialista humano.
  // Retorna gradientes ∂L/∂W e ∂L/∂b para cada camada.
  // Complexidade: O(Σ layerSizes[l]·layerSizes[l+1]) — mesma ordem do forward
  backward(xVec, yTarget) {
    const { activations, zs } = this.forward(xVec);
    const L = this.weights.length - 1; // índice da última camada
    const yCol = toColumn(yTarget);

    const gradW = new Array(this.weights.length);
    const gradB = new Array(this.weights.length);
    const deltas = new Array(this.weights.length);

    // δ^(L) = ∇ₐL ⊙ σ'(z^(L))
    // Com softmax + entropia cruzada, o gradiente combinado simplifica para (ŷ - y)
    // (a "mágica" clássica de cross-entropy + softmax — evita calcular a Jacobiana cheia aqui)
    deltas[L] = activations[L + 1].map((row, i) => [row[0] - yCol[i][0]]);

    gradW[L] = dot(deltas[L], transpose(activations[L]));
    gradB[L] = deltas[L];

    // δ^(l) = (W^(l+1))ᵀ δ^(l+1) ⊙ σ'(z^(l))  — propaga camada a camada, de trás pra frente
    for (let l = L - 1; l >= 0; l--) {
      const wT = transpose(this.weights[l + 1]);
      const propagated = dot(wT, deltas[l + 1]);
      const localDeriv = sigmoidDerivative(zs[l]);
      deltas[l] = hadamard(propagated, localDeriv);

      gradW[l] = dot(deltas[l], transpose(activations[l]));
      gradB[l] = deltas[l];
    }

    return { gradW, gradB };
  }
}