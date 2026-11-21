// Sokoban solver — push-optimal per room, deterministic.
//
// Reads every level in /input/<id>/level.txt (XSB format) and writes the
// player's move string to /output/<id>/solution.txt.
//
// Method: a generated level is a one-square-wide hallway with rooms off it.
// A box pushed out of its room can never come back, so each room is an
// independent push-optimal subproblem and the level optimum is the sum of
// the room optima. Each room is solved by a breadth-first search over
// pushes (state = box cells + the player's reachable region), then the
// pushes are expanded into full player moves and the rooms are chained by
// walking the player along the hallway.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR || '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR || '/output';

// Direction order is fixed everywhere: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const DCH = ['u', 'd', 'l', 'r'];

// ---------------------------------------------------------------- parsing

function parseLevel(text) {
  const rows = text.replace(/\r/g, '').split('\n');
  while (rows.length && rows[rows.length - 1] === '') rows.pop();
  const H = rows.length;
  let W = 0;
  for (const r of rows) W = Math.max(W, r.length);
  const wall = new Uint8Array(W * H).fill(1);
  const goal = new Uint8Array(W * H);
  const boxes = [];
  let player = -1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < rows[y].length; x++) {
      const c = rows[y][x];
      const i = y * W + x;
      if (c === '#') continue;
      wall[i] = 0;
      if (c === '.' || c === '*' || c === '+') goal[i] = 1;
      if (c === '$' || c === '*') boxes.push(i);
      if (c === '@' || c === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { W, H, wall, goal, boxes, player };
}

// ------------------------------------------------------------ flood fill

// Cells the player can reach from `start` without moving boxes.
// Returns a Uint8Array mask over the board; boxesAt is a Set of box cells.
function flood(W, H, wall, boxesAt, start) {
  const seen = new Uint8Array(W * H);
  if (start < 0 || wall[start] || boxesAt.has(start)) return seen;
  const stack = [start];
  seen[start] = 1;
  while (stack.length) {
    const c = stack.pop();
    const cx = c % W, cy = (c / W) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d], ny = cy + DY[d];
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const n = ny * W + nx;
      if (seen[n] || wall[n] || boxesAt.has(n)) continue;
      seen[n] = 1;
      stack.push(n);
    }
  }
  return seen;
}

function regionRep(mask) {
  for (let i = 0; i < mask.length; i++) if (mask[i]) return i;
  return -1;
}

// Shortest player walk from `from` to `to` with boxes as obstacles.
// Returns a move string, or null when unreachable.
function walkPath(W, H, wall, boxesAt, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(W * H).fill(-1);
  const prevDir = new Int8Array(W * H).fill(-1);
  const queue = [from];
  prev[from] = from;
  let head = 0;
  while (head < queue.length) {
    const c = queue[head++];
    if (c === to) break;
    const cx = c % W, cy = (c / W) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d], ny = cy + DY[d];
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const n = ny * W + nx;
      if (prev[n] !== -1 || wall[n] || boxesAt.has(n)) continue;
      prev[n] = c;
      prevDir[n] = d;
      queue.push(n);
    }
  }
  if (prev[to] === -1) return null;
  const chars = [];
  let c = to;
  while (c !== from) {
    chars.push(DCH[prevDir[c]]);
    c = prev[c];
  }
  return chars.reverse().join('');
}

// --------------------------------------------------------- decomposition

// Split a parsed level into hallway + rooms. Returns null when the level
// does not have the generated shape; the caller falls back to one room.
function decompose(L) {
  const { W, H, wall, goal } = L;
  const hallY = (L.player / W) | 0;
  // The hallway is the wall-free strip in the player's row; the row's end
  // cells are walls, so take every non-wall cell of the row as hallway.
  const inHall = (i) => ((i / W) | 0) === hallY && !wall[i];
  let hallCells = 0;
  for (let x = 0; x < W; x++) if (!wall[hallY * W + x]) hallCells++;
  if (hallCells < 2) return null;
  const compOf = new Int32Array(W * H).fill(-1);
  const comps = [];
  for (let i = 0; i < W * H; i++) {
    if (wall[i] || inHall(i) || compOf[i] !== -1) continue;
    const cells = [];
    const stack = [i];
    compOf[i] = comps.length;
    while (stack.length) {
      const c = stack.pop();
      cells.push(c);
      const cx = c % W, cy = (c / W) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d], ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const n = ny * W + nx;
        if (wall[n] || inHall(n) || compOf[n] !== -1) continue;
        compOf[n] = comps.length;
        stack.push(n);
      }
    }
    comps.push(cells);
  }
  const rooms = [];
  for (const cells of comps) {
    const cellSet = new Set(cells);
    const boxes = L.boxes.filter((b) => cellSet.has(b));
    const goals = cells.filter((c) => goal[c]);
    if (boxes.length === 0 && goals.length === 0) continue; // stray pocket
    if (boxes.length !== goals.length) return null;
    // Hallway cells touching this component are its doors.
    const entries = new Set();
    for (const c of cells) {
      const up = c - W, down = c + W;
      if (up >= 0 && inHall(up)) entries.add(up);
      if (down < W * H && inHall(down)) entries.add(down);
    }
    if (entries.size === 0) return null;
    rooms.push({
      cells,
      cellSet,
      boxes,
      goals,
      entries: [...entries].sort((a, b) => a - b),
    });
  }
  if (rooms.length === 0) return null;
  // Every box must belong to exactly one room.
  if (rooms.reduce((s, r) => s + r.boxes.length, 0) !== L.boxes.length) return null;
  rooms.sort((a, b) => a.entries[0] - b.entries[0]);
  return rooms;
}

// ------------------------------------------------------------ room solver

// Solve one room push-optimally by breadth-first search over pushes.
// The room grid is the room's cells plus its hallway entry cells; the
// player starts on an entry cell. When needExit is set, the final player
// region must contain an entry cell so the player can leave again.
// Returns { pushes: [{from, dir}], finalPlayer } in global cell indices,
// or null when unsolved within the expansion budget.
function solveRoom(L, room, needExit, budget) {
  const { W, H, wall, goal } = L;
  void wall; void goal;
  const open = new Uint8Array(W * H);
  for (const c of room.cells) open[c] = 1;
  for (const e of room.entries) open[e] = 1;
  const localWall = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) localWall[i] = open[i] ? 0 : 1;
  const goalSet = new Set(room.goals);
  const entrySet = new Set(room.entries);

  const startBoxes = [...room.boxes].sort((a, b) => a - b);
  const startRegion = flood(W, H, localWall, new Set(startBoxes), room.entries[0]);
  // The player may walk the hallway between this room's own entry cells.
  for (const e of room.entries) {
    const r = flood(W, H, localWall, new Set(startBoxes), e);
    for (let i = 0; i < r.length; i++) if (r[i]) startRegion[i] = 1;
  }
  const startRep = regionRep(startRegion);

  const keyOf = (boxes, rep) => boxes.join(',') + '|' + rep;
  const nodes = [{ boxes: startBoxes, rep: startRep, parent: -1, from: -1, dir: -1 }];
  const seen = new Map([[keyOf(startBoxes, startRep), 0]]);
  let head = 0;
  let expansions = 0;
  const isGoal = (boxes, rep) => {
    for (const b of boxes) if (!goalSet.has(b)) return false;
    if (!needExit) return true;
    const region = flood(W, H, localWall, new Set(boxes), rep);
    for (const e of entrySet) if (region[e]) return true;
    return false;
  };
  if (isGoal(startBoxes, startRep)) {
    return { pushes: [], finalPlayer: room.entries[0] };
  }
  while (head < nodes.length) {
    const node = nodes[head];
    const nodeId = head;
    head++;
    if (++expansions > budget) return null;
    const boxesAt = new Set(node.boxes);
    const region = flood(W, H, localWall, boxesAt, node.rep);
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      const bx = b % W, by = (b / W) | 0;
      for (let d = 0; d < 4; d++) {
        const stand = (by - DY[d]) * W + (bx - DX[d]);
        const dest = (by + DY[d]) * W + (bx + DX[d]);
        if (stand < 0 || stand >= W * H || dest < 0 || dest >= W * H) continue;
        if (!region[stand]) continue;
        if (localWall[dest] || boxesAt.has(dest)) continue;
        const nextBoxes = [...node.boxes];
        nextBoxes[bi] = dest;
        nextBoxes.sort((a, b) => a - b);
        // After the push the player stands on the box's old cell.
        const nextRegion = flood(W, H, localWall, new Set(nextBoxes), b);
        const nextRep = regionRep(nextRegion);
        const key = keyOf(nextBoxes, nextRep);
        if (seen.has(key)) continue;
        seen.set(key, nodes.length);
        nodes.push({ boxes: nextBoxes, rep: nextRep, parent: nodeId, from: b, dir: d });
        if (isGoal(nextBoxes, nextRep)) {
          const pushes = [];
          let cur = nodes[nodes.length - 1];
          while (cur.parent !== -1) {
            pushes.push({ from: cur.from, dir: cur.dir });
            cur = nodes[cur.parent];
          }
          pushes.reverse();
          return { pushes, finalPlayer: lastPlayerCell(L, room, pushes) };
        }
      }
    }
  }
  return null;
}

// Replay a room's pushes locally to find where the player ends up and to
// expand the pushes into a full move string. The player enters standing
// on the room's first entry cell.
function roomMoves(L, room, pushes) {
  const { W, H } = L;
  const open = new Uint8Array(W * H);
  for (const c of room.cells) open[c] = 1;
  for (const e of room.entries) open[e] = 1;
  const localWall = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) localWall[i] = open[i] ? 0 : 1;
  const boxesAt = new Set(room.boxes);
  let player = room.entries[0];
  let out = '';
  for (const p of pushes) {
    const bx = p.from % W, by = (p.from / W) | 0;
    const stand = (by - DY[p.dir]) * W + (bx - DX[p.dir]);
    const walk = walkPath(W, H, localWall, boxesAt, player, stand);
    if (walk === null) return null;
    out += walk + DCH[p.dir];
    boxesAt.delete(p.from);
    boxesAt.add((by + DY[p.dir]) * W + (bx + DX[p.dir]));
    player = p.from;
  }
  return { moves: out, finalPlayer: player };
}

function lastPlayerCell(L, room, pushes) {
  const r = roomMoves(L, room, pushes);
  return r ? r.finalPlayer : room.entries[0];
}

// -------------------------------------------------------------- assembly

function solveLevel(text) {
  const L = parseLevel(text);
  let rooms = decompose(L);
  if (!rooms) {
    // Fallback: the whole floor is one room, entered where the player is.
    const cells = [];
    for (let i = 0; i < L.W * L.H; i++) if (!L.wall[i]) cells.push(i);
    rooms = [{
      cells,
      cellSet: new Set(cells),
      boxes: L.boxes,
      goals: cells.filter((c) => L.goal[c]),
      entries: [L.player],
    }];
  }
  const BUDGET = 2000000;
  const solved = rooms.map((room) => ({
    room,
    res: solveRoom(L, room, false, BUDGET),
  }));
  for (const s of solved) if (!s.res) return '';
  // Decide play order: a room whose final player region cannot reach its
  // own door must be played last; if several are stuck, re-solve all but
  // one of them with the exit requirement.
  const canExit = solved.map((s) => {
    const boxesAt = new Set();
    for (const o of solved) for (const b of o.room.goals) boxesAt.add(b);
    const { finalPlayer } = roomMoves(L, s.room, s.res.pushes);
    const region = flood(L.W, L.H, L.wall, boxesAt, finalPlayer);
    return s.room.entries.some((e) => region[e]);
  });
  const stuckIdx = solved.map((s, i) => i).filter((i) => !canExit[i]);
  if (stuckIdx.length > 1) {
    for (let k = 1; k < stuckIdx.length; k++) {
      const s = solved[stuckIdx[k]];
      const res = solveRoom(L, s.room, true, BUDGET);
      if (res) s.res = res;
    }
  }
  const order = [
    ...solved.filter((s, i) => canExit[i]),
    ...solved.filter((s, i) => !canExit[i]),
  ];
  // Assemble: walk the hallway between rooms, then play each room.
  const boxesAt = new Set(L.boxes);
  let player = L.player;
  let out = '';
  for (const s of order) {
    const entry = s.room.entries[0];
    const walk = walkPath(L.W, L.H, L.wall, boxesAt, player, entry);
    if (walk === null) return '';
    out += walk;
    player = entry;
    const rm = roomMoves(L, s.room, s.res.pushes);
    if (!rm) return '';
    out += rm.moves;
    player = rm.finalPlayer;
    for (const b of s.room.boxes) boxesAt.delete(b);
    for (const g of s.room.goals) boxesAt.add(g);
  }
  return out;
}

// ------------------------------------------------------------------ main

for (const id of readdirSync(IN_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), moves + '\n');
}
