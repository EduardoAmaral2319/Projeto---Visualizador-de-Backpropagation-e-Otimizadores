// src/main.js — versão completa, com fix de resolução do canvas (devicePixelRatio)
import { GameEngine } from './game/engine.js';
import { Telemetry, ACTIONS } from './game/telemetry.js';
import { NeuralNetwork } from './math/nn.js';
import { SGD, Adam } from './math/optimizers.js';
import { drawNetworkDiagram, drawMatrixPanel } from './ui/visualizer.js';
import { LossChart } from './ui/charts.js';

// --- DOM refs ---
const canvasGame = document.getElementById('canvas-game');
const ctxGame = canvasGame.getContext('2d');
const canvasNetwork = document.getElementById('canvas-network');
const ctxNetwork = canvasNetwork.getContext('2d');
const canvasMatrix = document.getElementById('canvas-matrix');
const ctxMatrix = canvasMatrix.getContext('2d');
const canvasChart = document.getElementById('canvas-chart');
const ctxChart = canvasChart.getContext('2d');

const btnRecord = document.getElementById('btn-record');
const btnTrain = document.getElementById('btn-train');
const btnOptimizer = document.getElementById('btn-optimizer');
const optimizerLabel = document.getElementById('optimizer-label');
const btnReset = document.getElementById('btn-reset');
const phaseLabel = document.getElementById('phase-label');
const scoreValue = document.getElementById('score-value');
const datasetSize = document.getElementById('dataset-size');

// --- Fix de resolução: sincroniza o buffer interno do canvas com o tamanho
// exibido em CSS, escalando por devicePixelRatio pra texto/linhas nítidos.
// Sem isso, o navegador faz upscale do desenho já renderizado -> texto borrado
// (foi exatamente o bug visto no painel "Jacobiana + Gradiente").
// Complexidade: O(1) por canvas.
function fitCanvasToDisplaySize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = canvas.clientWidth;
  const displayHeight = canvas.clientHeight;

  canvas.width = displayWidth * dpr;
  canvas.height = displayHeight * dpr;

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // desenha em coordenadas "lógicas" (CSS px)
  return { width: displayWidth, height: displayHeight };
}

const allCanvases = [canvasGame, canvasNetwork, canvasMatrix, canvasChart];
let logicalSizes = {};

function resizeAllCanvases() {
  for (const canvas of allCanvases) {
    logicalSizes[canvas.id] = fitCanvasToDisplaySize(canvas);
  }
}

// Precisa rodar ANTES de criar o engine, senão GameEngine nasce com
// canvasWidth/canvasHeight = 0 (o CSS grid ainda não tinha aplicado tamanho).
resizeAllCanvases();
window.addEventListener('resize', () => {
  resizeAllCanvases();
  engine.canvasWidth = logicalSizes['canvas-game'].width;
  engine.canvasHeight = logicalSizes['canvas-game'].height;
});

// --- Núcleo (criado DEPOIS do resize) ---
const engine = new GameEngine(logicalSizes['canvas-game'].width, logicalSizes['canvas-game'].height);
const telemetry = new Telemetry();
const network = new NeuralNetwork([3, 6, 4]); // 3 entradas (estado), 4 saídas (ACTIONS)
const chart = new LossChart();

let optimizerName = 'sgd';
let optimizer = new SGD(0.15);
let phase = 'expert'; // 'expert' (grava humano) | 'training' (IA joga + treina ao vivo)
let lastDeltas = null; // gradB da última amostra treinada — alimenta o diagrama de rede

// --- Teclado (Fase Especialista) ---
const keysHeld = new Set();
window.addEventListener('keydown', (e) => keysHeld.add(e.code));
window.addEventListener('keyup', (e) => keysHeld.delete(e.code));

function currentHumanAction() {
  if (keysHeld.has('Space') || keysHeld.has('ArrowUp')) return 'jump';
  if (keysHeld.has('ArrowLeft')) return 'left';
  if (keysHeld.has('ArrowRight')) return 'right';
  return 'none';
}

function applyAction(action) {
  if (action === 'jump') engine.player.jump();
  else if (action === 'left') engine.player.moveLeft();
  else if (action === 'right') engine.player.moveRight();
  else engine.player.stopMoving();
}

function aiAction(state) {
  const { activations } = network.forward(state);
  const output = activations[activations.length - 1].map(row => row[0]);
  return ACTIONS[output.indexOf(Math.max(...output))]; // argmax do softmax
}

// --- Botões: cada um só troca estado, nunca desenha diretamente ---
btnRecord.addEventListener('click', () => {
  phase = 'expert';
  telemetry.start();
  btnRecord.classList.add('active');
  btnTrain.classList.remove('active');
  phaseLabel.textContent = 'Fase: Especialista (gravando)';
});

btnTrain.addEventListener('click', () => {
  if (telemetry.size() < 10) {
    alert('Grave pelo menos ~10 ações como especialista antes de treinar.');
    return;
  }
  phase = 'training';
  telemetry.stop();
  btnTrain.classList.add('active');
  btnRecord.classList.remove('active');
  phaseLabel.textContent = 'Fase: Aprendiz (IA treinando ao vivo)';
});

btnOptimizer.addEventListener('click', () => {
  optimizerName = optimizerName === 'sgd' ? 'adam' : 'sgd';
  optimizer = optimizerName === 'sgd' ? new SGD(0.15) : new Adam(network, 0.01);
  optimizerLabel.textContent = optimizerName.toUpperCase();
});

btnReset.addEventListener('click', () => {
  engine.reset();
  telemetry.clear();
  chart.reset();
  lastDeltas = null;
  phase = 'expert';
  phaseLabel.textContent = 'Fase: Especialista';
  btnRecord.classList.remove('active');
  btnTrain.classList.remove('active');
});

// --- Treino: um mini-batch por frame, gradientes SOMADOS (não a média) ---
const BATCH_SIZE = 8;

function trainStep() {
  const batch = telemetry.sampleBalancedBatch(BATCH_SIZE);
  if (batch.length === 0) return;

  let accGradW = null, accGradB = null, batchLoss = 0;

  for (const sample of batch) {
    const { gradW, gradB } = network.backward(sample.state, sample.action);
    const { activations } = network.forward(sample.state);
    const yPred = activations[activations.length - 1].map(row => row[0]);
    batchLoss += LossChart.crossEntropy(sample.action, yPred);

    if (!accGradW) {
      accGradW = gradW;
      accGradB = gradB;
    } else {
      for (let l = 0; l < gradW.length; l++) {
        for (let i = 0; i < gradW[l].length; i++) {
          for (let j = 0; j < gradW[l][i].length; j++) accGradW[l][i][j] += gradW[l][i][j];
          accGradB[l][i][0] += gradB[l][i][0];
        }
      }
    }
    lastDeltas = gradB; // guarda o δ da amostra mais recente pro diagrama
  }

  optimizer.step(network, accGradW, accGradB);
  chart.push(optimizerName, batchLoss / batch.length);
}

// --- Loop principal: acoplado ao onFrame do GameEngine (Passo 4) ---
engine.onFrame = (snapshot) => {
  const action = phase === 'expert' ? currentHumanAction() : aiAction(snapshot.state);
  applyAction(action);

  if (phase === 'expert') telemetry.record(snapshot.state, action);
  else trainStep();

  if (snapshot.gameOver) engine.reset();

  scoreValue.textContent = snapshot.score.toFixed(1);
  datasetSize.textContent = telemetry.size();

  drawGame(snapshot, logicalSizes['canvas-game']);

  drawNetworkDiagram(
    ctxNetwork, network, lastDeltas,
    logicalSizes['canvas-network'].width, logicalSizes['canvas-network'].height
  );

  ctxMatrix.clearRect(0, 0, logicalSizes['canvas-matrix'].width, logicalSizes['canvas-matrix'].height);
  drawMatrixPanel(ctxMatrix, network.forward(snapshot.state).activations.at(-1), 'Saída (softmax)', 20, 40);

  chart.draw(ctxChart, logicalSizes['canvas-chart'].width, logicalSizes['canvas-chart'].height);
};

function drawGame(snapshot, size) {
  ctxGame.clearRect(0, 0, size.width, size.height);
  ctxGame.fillStyle = '#63b3ed';
  const hit = snapshot.player.getHitbox();
  ctxGame.fillRect(hit.left, hit.top, hit.right - hit.left, hit.bottom - hit.top);

  ctxGame.fillStyle = '#f87171';
  for (const obs of snapshot.obstacles) ctxGame.fillRect(obs.x, obs.y, obs.width, obs.height);
}

engine.start();