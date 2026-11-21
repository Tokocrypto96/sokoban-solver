// Sokoban solver — push-optimal per room, deterministic.
//
// Reads every level in /input/<id>/level.txt (XSB format) and writes the
// player's move string to /output/<id>/solution.txt.
//
// Method: a generated level is a one-square-wide hallway with rooms off it.
// A box pushed out of its room can never come back, so each room is an
// independent push-optimal subproblem and the level optimum is the sum of
// the room optima. Each room is solved push-optimally by an A* search
// over pushes (state = box cells + the player's reachable region), guided
// by a minimum-cost box-to-goal matching heuristic over precomputed push
// distances, with static dead-square pruning. The pushes are then
// expanded into full player moves and the rooms are chained by walking
// the player along the hallway. Search budgets are counted in expansions,
// never in wall-clock time, so runs are byte-for-byte deterministic.

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

// ------------------------------------------------------- room precompute

// Push-distance table per goal over the room grid (other boxes ignored):
// the minimum number of pushes that moves a box from a cell to that goal.
// Computed by reverse BFS from the goal over push moves: a box arrived at
// cell c pushed in direction d, so it came from p = c - d and the player
// stood at s = p - d; both must be open cells.
function pushDistances(W, H, localWall, goals) {
  const N = W * H;
  const tables = [];
  for (const g of goals) {
    const dist = new Int32Array(N).fill(-1);
    dist[g] = 0;
    const queue = [g];
    let head = 0;
    while (head < queue.length) {
      const c = queue[head++];
      const cx = c % W, cy = (c / W) | 0;
      for (let d = 0; d < 4; d++) {
        const px = cx - DX[d], py = cy - DY[d];
        const sx = px - DX[d], sy = py - DY[d];
        if (px < 0 || py < 0 || px >= W || py >= H) continue;
        if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
        const p = py * W + px, s = sy * W + sx;
        if (localWall[p] || localWall[s] || dist[p] !== -1) continue;
        dist[p] = dist[c] + 1;
        queue.push(p);
      }
    }
    tables.push(dist);
  }
  return tables;
}

// Minimum-cost assignment of boxes to distinct goals by table distance.
// Boxes and goals are few (at most 4), so try every permutation.
// Returns -1 when some box cannot reach any goal.
function matchCost(tables, boxes) {
  const n = boxes.length;
  if (n === 0) return 0;
  const perm = boxes.map((_, i) => i);
  let best = -1;
  const visit = (k) => {
    if (k === n) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = tables[perm[i]][boxes[i]];
        if (d < 0) return;
        sum += d;
        if (best !== -1 && sum >= best) return;
      }
      best = sum;
      return;
    }
    for (let i = k; i < n; i++) {
      const t = perm[k]; perm[k] = perm[i]; perm[i] = t;
      visit(k + 1);
      const u = perm[k]; perm[k] = perm[i]; perm[i] = u;
    }
  };
  visit(0);
  return best;
}

// ------------------------------------------------------------ room solver

// Solve one room push-optimally with A* over pushes, guided by the
// matching heuristic over push distances, with static dead-square
// pruning. The room grid is the room's cells plus its hallway entry
// cells; the player starts on an entry cell. When needExit is set, the
// final player region must contain an entry cell so the player can
// leave again. Returns { pushes, finalPlayer } in global cell indices,
// or null when unsolved within the expansion budget.
function solveRoom(L, room, needExit, budget) {
  const { W, H } = L;
  const open = new Uint8Array(W * H);
  for (const c of room.cells) open[c] = 1;
  for (const e of room.entries) open[e] = 1;
  const localWall = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) localWall[i] = open[i] ? 0 : 1;
  const goalSet = new Set(room.goals);
  const entrySet = new Set(room.entries);
  const goals = [...room.goals].sort((a, b) => a - b);
  const tables = pushDistances(W, H, localWall, goals);
  // Static dead cells: no goal is reachable from them in the push graph.
  const dead = new Uint8Array(W * H);
  for (let c = 0; c < W * H; c++) {
    if (localWall[c] || goalSet.has(c)) continue;
    let alive = false;
    for (const t of tables) if (t[c] >= 0) { alive = true; break; }
    if (!alive) dead[c] = 1;
  }

  const startBoxes = [...room.boxes].sort((a, b) => a - b);
  for (const b of startBoxes) if (dead[b]) return null;
  const h0 = matchCost(tables, startBoxes);
  if (h0 < 0) return null;
  const startRegion = flood(W, H, localWall, new Set(startBoxes), room.entries[0]);
  // The player may walk the hallway between this room's own entry cells.
  for (const e of room.entries) {
    const r = flood(W, H, localWall, new Set(startBoxes), e);
    for (let i = 0; i < r.length; i++) if (r[i]) startRegion[i] = 1;
  }
  const startRep = regionRep(startRegion);

  const keyOf = (boxes, rep) => boxes.join(',') + '|' + rep;
  const nodes = [{
    boxes: startBoxes, rep: startRep, g: 0, h: h0, parent: -1, from: -1, dir: -1,
  }];
  const best = new Map([[keyOf(startBoxes, startRep), 0]]);
  // Binary heap of node ids ordered by (f, h, id): deterministic.
  const heap = [0];
  const less = (a, b) => {
    const fa = nodes[a].g + nodes[a].h, fb = nodes[b].g + nodes[b].h;
    if (fa !== fb) return fa < fb;
    if (nodes[a].h !== nodes[b].h) return nodes[a].h < nodes[b].h;
    return a < b;
  };
  const heapPush = (id) => {
    heap.push(id);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && less(heap[l], heap[m])) m = l;
        if (r < heap.length && less(heap[r], heap[m])) m = r;
        if (m === i) break;
        const t = heap[i]; heap[i] = heap[m]; heap[m] = t;
        i = m;
      }
    }
    return top;
  };
  const isGoal = (boxes, rep) => {
    for (const b of boxes) if (!goalSet.has(b)) return false;
    if (!needExit) return true;
    const region = flood(W, H, localWall, new Set(boxes), rep);
    for (const e of entrySet) if (region[e]) return true;
    return false;
  };
  let expansions = 0;
  while (heap.length) {
    const nodeId = heapPop();
    const node = nodes[nodeId];
    if (node.g !== best.get(keyOf(node.boxes, node.rep))) continue;
    if (isGoal(node.boxes, node.rep)) {
      const pushes = [];
      let cur = node;
      while (cur.parent !== -1) {
        pushes.push({ from: cur.from, dir: cur.dir });
        cur = nodes[cur.parent];
      }
      pushes.reverse();
      return { pushes, finalPlayer: lastPlayerCell(L, room, pushes) };
    }
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
        if (localWall[dest] || boxesAt.has(dest) || dead[dest]) continue;
        const nextBoxes = [...node.boxes];
        nextBoxes[bi] = dest;
        nextBoxes.sort((a, b) => a - b);
        const ng = node.g + 1;
        // After the push the player stands on the box's old cell.
        const nextRegion = flood(W, H, localWall, new Set(nextBoxes), b);
        const nextRep = regionRep(nextRegion);
        const key = keyOf(nextBoxes, nextRep);
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= ng) continue;
        const nh = matchCost(tables, nextBoxes);
        if (nh < 0) continue;
        best.set(key, ng);
        nodes.push({
          boxes: nextBoxes, rep: nextRep, g: ng, h: nh,
          parent: nodeId, from: b, dir: d,
        });
        heapPush(nodes.length - 1);
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
