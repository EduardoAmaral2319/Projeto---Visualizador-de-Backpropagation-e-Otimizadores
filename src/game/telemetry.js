// src/game/telemetry.js
// Grava, durante a Fase 1 (Especialista humano), pares [estado, ação]
// que servirão de dataset de treino pra Fase 2 (nn.js aprendendo por imitação).

const ACTIONS = ['jump', 'left', 'right', 'none']; // ordem fixa -> índice do one-hot

export class Telemetry {
  constructor() {
    this.dataset = []; // [{ state: [n1,n2,n3], action: [a1,a2,a3,a4] }]
    this.recording = false;
  }

  start() { this.recording = true; }
  stop()  { this.recording = false; }
  clear() { this.dataset = []; }

  // action: string, uma de ACTIONS. Ex.: 'jump'
  // state: vetor já normalizado, vindo de player.getState() via engine.getSnapshot()
  // Complexidade: O(k), k = ACTIONS.length (constante pequena, 4) -> O(1) na prática
  record(state, action) {
    if (!this.recording) return;

    const oneHot = ACTIONS.map(a => (a === action ? 1 : 0));
    this.dataset.push({ state: [...state], action: oneHot });
  }

  size() {
    return this.dataset.length;
  }

  // Amostragem ESTRATIFICADA: garante que cada uma das 4 classes apareça
  // aproximadamente igual no batch, mesmo que o dataset bruto seja desbalanceado
  // (ex: muito mais 'none' do que 'jump'/'left'/'right').
  // Complexidade: O(n) pra agrupar por classe (uma vez) + O(batchSize) pra sortear
  sampleBalancedBatch(batchSize) {
    const byClass = [[], [], [], []]; // um array por ação em ACTIONS
    for (const sample of this.dataset) {
      const classIdx = sample.action.indexOf(1);
      byClass[classIdx].push(sample);
    }

    const perClass = Math.ceil(batchSize / 4);
    const batch = [];

    for (const classSamples of byClass) {
      if (classSamples.length === 0) continue; // classe ainda não gravada, pula
      for (let i = 0; i < perClass; i++) {
        const idx = Math.floor(Math.random() * classSamples.length);
        batch.push(classSamples[idx]); // COM reposição (classes raras podem ter <perClass amostras)
      }
    }

    // Embaralha o batch final (Fisher-Yates) pra não treinar em blocos por classe
    for (let i = batch.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [batch[i], batch[j]] = [batch[j], batch[i]];
    }

    return batch;
  }

  // Serializa pra debug/inspeção externa (ex: baixar como JSON)
  toJSON() {
    return JSON.stringify(this.dataset);
  }
}

export { ACTIONS };