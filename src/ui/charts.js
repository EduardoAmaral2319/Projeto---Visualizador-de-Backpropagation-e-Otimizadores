// src/ui/charts.js
// Gráfico de convergência (Loss) — plota SGD vs Adam simultaneamente,
// pra comparação visual direta da trajetória de cada otimizador.

const MAX_POINTS = 200; // buffer circular — janela deslizante do gráfico

export class LossChart {
  constructor() {
    this.seriesSGD = [];
    this.seriesAdam = [];
  }

  // Cross-entropy: L = -Σ y_i·log(ŷ_i)
  // Complexidade: O(n), n = número de classes (aqui, 4 ações)
  static crossEntropy(yTarget, yPred) {
    const EPS = 1e-9; // evita log(0) -> -Infinity se a predição zerar uma classe
    let loss = 0;
    for (let i = 0; i < yTarget.length; i++) {
      loss -= yTarget[i] * Math.log(yPred[i] + EPS);
    }
    return loss;
  }

  // Complexidade: O(1) amortizado (shift só ocorre quando o buffer está cheio)
  push(optimizerName, lossValue) {
    const series = optimizerName === 'adam' ? this.seriesAdam : this.seriesSGD;
    series.push(lossValue);
    if (series.length > MAX_POINTS) series.shift();
  }

  reset() {
    this.seriesSGD = [];
    this.seriesAdam = [];
  }

  // Desenha as duas curvas sobrepostas num <canvas>, com eixo Y autoescalado.
  // Complexidade: O(MAX_POINTS) — constante, independe de quanto tempo o treino já rodou
  draw(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);

    const allValues = [...this.seriesSGD, ...this.seriesAdam];
    if (allValues.length === 0) return;

    const maxLoss = Math.max(...allValues, 0.01); // evita divisão por 0 se loss=0 sempre
    const toY = v => height - (v / maxLoss) * height;
    const toX = i => (i / (MAX_POINTS - 1)) * width;

    this._drawSeries(ctx, this.seriesSGD, toX, toY, '#f87171');  // vermelho = SGD
    this._drawSeries(ctx, this.seriesAdam, toX, toY, '#4ade80'); // verde = Adam

    this._drawLegend(ctx);
  }

  _drawSeries(ctx, series, toX, toY, color) {
    if (series.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(series[0]));
    for (let i = 1; i < series.length; i++) {
      ctx.lineTo(toX(i), toY(series[i]));
    }
    ctx.stroke();
  }

  _drawLegend(ctx) {
    ctx.font = '11px monospace';
    ctx.fillStyle = '#f87171';
    ctx.fillText('SGD', 8, 14);
    ctx.fillStyle = '#4ade80';
    ctx.fillText('Adam', 8, 28);
  }
}