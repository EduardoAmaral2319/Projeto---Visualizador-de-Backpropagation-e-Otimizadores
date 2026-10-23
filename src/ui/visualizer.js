// src/ui/visualizer.js
// Desenha o diagrama da rede neural e os painéis de matriz/vetor num <canvas>.
// Não modifica nada — só lê `network` (nn.js) e `deltas` (retornado por backward()).

const NEURON_RADIUS = 12;
const LAYER_GAP = 120;

// Calcula as coordenadas (x,y) de cada neurônio, camada por camada.
// Complexidade: O(N), N = total de neurônios na rede
function computeLayout(layerSizes, width, height) {
  const positions = []; // positions[l][i] = {x, y}
  const layerGapX = width / (layerSizes.length + 1);

  layerSizes.forEach((size, l) => {
    const x = layerGapX * (l + 1);
    const gapY = height / (size + 1);
    const layerPositions = [];
    for (let i = 0; i < size; i++) {
      layerPositions.push({ x, y: gapY * (i + 1) });
    }
    positions.push(layerPositions);
  });

  return positions;
}

// Desenha neurônios + conexões. deltas (opcional) colore/anima as conexões
// da camada l+1 -> l com base na magnitude de δ^(l+1) — "o erro fluindo".
// Complexidade: O(N + E), N = neurônios, E = conexões (pesos totais)
export function drawNetworkDiagram(ctx, network, deltas, width, height) {
  ctx.clearRect(0, 0, width, height);
  const positions = computeLayout(network.layerSizes, width, height);

  // Conexões primeiro (ficam atrás dos neurônios)
  for (let l = 0; l < network.weights.length; l++) {
    const W = network.weights[l];
    const fromLayer = positions[l];
    const toLayer = positions[l + 1];

    // Magnitude máxima de δ nesta camada de destino, pra normalizar o brilho (0 a 1)
    const maxDelta = deltas
      ? Math.max(...deltas[l].map(row => Math.abs(row[0])), 1e-6)
      : null;

    for (let i = 0; i < W.length; i++) {       // neurônio de destino
      for (let j = 0; j < W[i].length; j++) {   // neurônio de origem
        const from = fromLayer[j];
        const to = toLayer[i];
        const weight = W[i][j];

        let alpha = Math.min(1, Math.abs(weight) / 2); // espessura visual base
        let color = '99, 179, 237'; // azul default (RGB)

        if (deltas) {
          const intensity = Math.abs(deltas[l][i][0]) / maxDelta;
          alpha = 0.15 + 0.85 * intensity; // brilho proporcional ao erro propagado
          color = '244, 114, 182'; // rosa durante backprop, distinto do azul "parado"
        }

        ctx.strokeStyle = `rgba(${color}, ${alpha})`;
        ctx.lineWidth = weight >= 0 ? 1.5 : 1; // peso negativo = linha mais fina
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    }
  }

  // Neurônios por cima
  positions.forEach(layer => {
    layer.forEach(({ x, y }) => {
      ctx.beginPath();
      ctx.arc(x, y, NEURON_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = '#1e293b';
      ctx.fill();
      ctx.strokeStyle = '#94a3b8';
      ctx.stroke();
    });
  });
}

// Painel genérico: desenha uma matriz (ou vetor coluna) como grade de números.
// Reaproveitado tanto pra Jacobiana (n×n) quanto pro gradiente (n×1).
// Complexidade: O(rows·cols)
export function drawMatrixPanel(ctx, matrix, label, x, y, cellSize = 34) {
  ctx.font = '11px monospace';
  ctx.fillStyle = '#e2e8f0';
  ctx.fillText(label, x, y - 8);

  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      const cx = x + j * cellSize;
      const cy = y + i * cellSize;

      ctx.strokeStyle = '#334155';
      ctx.strokeRect(cx, cy, cellSize, cellSize);

      const value = matrix[i][j];
      // Cor por sinal: verde positivo, vermelho negativo — leitura visual rápida
      ctx.fillStyle = value >= 0 ? '#4ade80' : '#f87171';
      ctx.font = '10px monospace';
      ctx.fillText(value.toFixed(3), cx + 4, cy + cellSize / 2 + 3);
    }
  }
}