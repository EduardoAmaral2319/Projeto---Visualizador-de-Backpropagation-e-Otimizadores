// src/game/player.js
const GRAVITY = 1800;       // px/s²
const JUMP_FORCE = -650;    // px/s
const MOVE_SPEED = 300;     // px/s

export class Player {
  constructor(x = 80, groundY = 320) {
    this.x = x;
    this.groundY = groundY;
    this.y = groundY;
    this.vx = 0;
    this.vy = 0;
    this.width = 24;
    this.height = 32;
    this.onGround = true;
  }

  jump() {
    if (!this.onGround) return;
    this.vy = JUMP_FORCE;
    this.onGround = false;
  }

  moveLeft()  { this.vx = -MOVE_SPEED; }
  moveRight() { this.vx = MOVE_SPEED; }
  stopMoving() { this.vx = 0; }

  // Complexidade: O(1)
  update(dt) {
    this.vy += GRAVITY * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    if (this.y >= this.groundY) {
      this.y = this.groundY;
      this.vy = 0;
      this.onGround = true;
    }
  }

  getHitbox() {
    return { left: this.x, right: this.x + this.width, top: this.y - this.height, bottom: this.y };
  }

  getState(nearestObstacle, canvasWidth) {
    const normY = (this.y - this.groundY) / this.groundY;
    const normVy = Math.max(-1, Math.min(1, this.vy / 1000));
    const obstacleDist = nearestObstacle
      ? Math.max(0, Math.min(1, (nearestObstacle.x - this.x) / canvasWidth))
      : 1;

    return [normY, normVy, obstacleDist];
  }
}