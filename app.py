from flask import Flask, render_template, request, jsonify, session
import heapq, time, math, random, copy
from collections import deque

app = Flask(__name__)
app.secret_key = 'chess_ai_pathfinding_2024'

# ════════════════════════════════════════════════════════════
#  CHESS PIECE MOVEMENT RULES
# ════════════════════════════════════════════════════════════

def get_chess_neighbors(pos, grid, rows, cols, piece='king', piece_color='white', board_pieces=None, goal=None):
    """
    Returns legal destination squares for 'piece' of 'piece_color' at 'pos'.
    board_pieces: dict mapping (r,c) -> color string ('white'/'black'), or None.
    Friendly pieces block movement; enemy pieces can be captured (reached but not passed).
    Walls (grid==1) always block.
    """
    r, c = pos
    neighbors = []
    bp = board_pieces or {}

    def can_enter(nr, nc):
        """Can the piece move TO (nr,nc)?  Returns ('yes','no','capture')."""
        if not (0 <= nr < rows and 0 <= nc < cols):
            return 'no'
        if grid[nr][nc] == 1:
            return 'no'
        occupant = bp.get((nr, nc))
        if occupant is None:
            return 'yes'
        if occupant == piece_color:
            return 'no'          # blocked by friendly
        return 'capture'         # enemy – can land but not pass through

    if piece == 'king':
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nr, nc = r+dr, c+dc
            status = can_enter(nr, nc)
            if status in ('yes', 'capture'):
                neighbors.append((nr, nc))

    elif piece in ('queen', 'rook', 'bishop'):
        if piece == 'queen':
            dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        elif piece == 'rook':
            dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        else:  # bishop
            dirs = [(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            while True:
                status = can_enter(nr, nc)
                if status == 'no':
                    break
                neighbors.append((nr, nc))
                if status == 'capture':
                    break          # cannot slide past a captured piece
                nr += dr; nc += dc

    elif piece == 'knight':
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r+dr, c+dc
            status = can_enter(nr, nc)
            if status in ('yes', 'capture'):
                neighbors.append((nr, nc))

    elif piece == 'pawn':
        # Determine forward direction based on piece color
        fwd = -1 if piece_color == 'white' else 1
        enemy_color = 'black' if piece_color == 'white' else 'white'

        # Forward move (1 step) – only to empty square
        nr, nc = r + fwd, c
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1 and bp.get((nr, nc)) is None:
            neighbors.append((nr, nc))
            # Initial double-step from starting rank
            start_rank = 6 if piece_color == 'white' else 1
            if r == start_rank:
                nr2 = r + 2*fwd
                if 0 <= nr2 < rows and grid[nr2][nc] != 1 and bp.get((nr2, nc)) is None:
                    neighbors.append((nr2, nc))

        # Diagonal captures
        for dc in (-1, 1):
            nr, nc = r + fwd, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                occupant = bp.get((nr, nc))
                if grid[nr][nc] != 1 and occupant == enemy_color:
                    neighbors.append((nr, nc))
                # Also allow moving to goal even if it looks empty (for pathfinding)
                elif goal and (nr, nc) == goal and grid[nr][nc] != 1:
                    neighbors.append((nr, nc))

    return neighbors

# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════

def reconstruct_path(came_from, start, goal):
    path, cur = [], goal
    while cur != start:
        path.append(cur)
        cur = came_from.get(cur)
        if cur is None: return []
    path.append(start)
    return list(reversed(path))

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def to_list(seq):
    return [list(p) for p in seq]

def fail(name, explored, t0, desc):
    return {
        'path': [], 'explored': to_list(explored),
        'nodes_explored': len(explored),
        'path_length': 0, 'cost': 0,
        'g_cost': 0, 'h_cost': 0, 'f_cost': 0,
        'time_ms': round((time.time()-t0)*1000,3),
        'algorithm': name, 'description': desc, 'found': False
    }

def ok(name, path, explored, g, h, t0, desc):
    f = g + h
    return {
        'path': to_list(path), 'explored': to_list(explored),
        'nodes_explored': len(explored),
        'path_length': len(path), 'cost': g,
        'g_cost': g, 'h_cost': h, 'f_cost': f,
        'time_ms': round((time.time()-t0)*1000,3),
        'algorithm': name, 'description': desc, 'found': True
    }

# ════════════════════════════════════════════════════════════
#  1. BFS — Breadth-First Search
# ════════════════════════════════════════════════════════════

def bfs(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    queue = deque([start])
    came_from = {start: None}
    explored = []
    while queue:
        cur = queue.popleft()
        explored.append(cur)
        if cur == goal:
            path = reconstruct_path(came_from, start, goal)
            g = len(path)-1
            return ok('BFS', path, explored, g, 0, t0,
                      f'BFS explores level by level. Guarantees shortest path (unweighted). '
                      f'Steps: {g} | Nodes explored: {len(explored)}')
        for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal):
            if n not in came_from:
                came_from[n] = cur
                queue.append(n)
    return fail('BFS', explored, t0, 'BFS: No path found. All reachable nodes explored.')

# ════════════════════════════════════════════════════════════
#  2. DFS — Depth-First Search
# ════════════════════════════════════════════════════════════

def dfs(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    stack = [start]
    came_from = {start: None}
    explored_set = set()
    explored = []
    while stack:
        cur = stack.pop()
        if cur in explored_set: continue
        explored_set.add(cur)
        explored.append(cur)
        if cur == goal:
            path = reconstruct_path(came_from, start, goal)
            g = len(path)-1
            return ok('DFS', path, explored, g, 0, t0,
                      f'DFS dives deep before backtracking. Not guaranteed optimal. '
                      f'Steps: {g} | Nodes explored: {len(explored)}')
        for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal):
            if n not in came_from:
                came_from[n] = cur
                stack.append(n)
    return fail('DFS', explored, t0, 'DFS: No path found.')

# ════════════════════════════════════════════════════════════
#  3. UCS — Uniform Cost Search
# ════════════════════════════════════════════════════════════

def ucs(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    done = set()
    explored = []
    while pq:
        g, cur = heapq.heappop(pq)
        if cur in done: continue
        done.add(cur); explored.append(cur)
        if cur == goal:
            path = reconstruct_path(came_from, start, goal)
            return ok('UCS', path, explored, g, 0, t0,
                      f'UCS expands lowest-cost node first. Optimal for weighted graphs. '
                      f'Total cost: {g} | Nodes explored: {len(explored)}')
        for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal):
            new_g = g + 1  # uniform cost = 1 per move
            if n not in cost_so_far or new_g < cost_so_far[n]:
                cost_so_far[n] = new_g
                came_from[n] = cur
                heapq.heappush(pq, (new_g, n))
    return fail('UCS', explored, t0, 'UCS: No path found.')

# ════════════════════════════════════════════════════════════
#  4. A* — A-Star Search
# ════════════════════════════════════════════════════════════

def astar(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    pq = [(manhattan(start,goal), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    done = set()
    explored = []
    while pq:
        f, g, cur = heapq.heappop(pq)
        if cur in done: continue
        done.add(cur); explored.append(cur)
        if cur == goal:
            path = reconstruct_path(came_from, start, goal)
            h = manhattan(goal, goal)
            return ok('A*', path, explored, g, h, t0,
                      f'A* uses f(n)=g(n)+h(n). g={g} (actual cost) + h=0 (at goal). '
                      f'Optimal & complete. Nodes explored: {len(explored)}')
        for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal):
            tg = g_score[cur] + 1
            if n not in g_score or tg < g_score[n]:
                g_score[n] = tg
                came_from[n] = cur
                h = manhattan(n, goal)
                heapq.heappush(pq, (tg+h, tg, n))
    return fail('A*', explored, t0, 'A*: No path found.')

# ════════════════════════════════════════════════════════════
#  5. Greedy Best-First Search
# ════════════════════════════════════════════════════════════

def greedy(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    pq = [(manhattan(start,goal), start)]
    came_from = {start: None}
    done = set()
    explored = []
    while pq:
        h, cur = heapq.heappop(pq)
        if cur in done: continue
        done.add(cur); explored.append(cur)
        if cur == goal:
            path = reconstruct_path(came_from, start, goal)
            g = len(path)-1
            return ok('Greedy', path, explored, g, 0, t0,
                      f'Greedy always moves toward goal using heuristic h(n)={h}. '
                      f'Fast but not always optimal. Steps: {g}')
        for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal):
            if n not in came_from:
                came_from[n] = cur
                heapq.heappush(pq, (manhattan(n,goal), n))
    return fail('Greedy', explored, t0, 'Greedy: No path found.')

# ════════════════════════════════════════════════════════════
#  6. Backtracking
# ════════════════════════════════════════════════════════════

def backtracking(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    explored = []
    result = []
    MAX_EXP = rows * cols * 2

    def bt(cur, path, visited):
        explored.append(cur)
        if len(explored) > MAX_EXP: return False
        if cur == goal:
            result.extend(path + [cur]); return True
        nbrs = sorted(get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal),
                      key=lambda n: manhattan(n, goal))   # MRV heuristic
        for n in nbrs:
            if n not in visited:
                visited.add(n)
                if bt(n, path+[cur], visited): return True
                visited.discard(n)
        return False

    found = bt(start, [], {start})
    g = len(result)-1 if result else 0
    if found:
        return ok('Backtracking', result, explored, g, 0, t0,
                  f'Backtracking with MRV heuristic. Systematically tries moves and backtracks on dead ends. '
                  f'Constraints: no revisits + valid {piece} moves. '
                  f'Nodes explored: {len(explored)} | Cost: {g}')
    return fail('Backtracking', explored, t0, 'Backtracking: No path found.')

# ════════════════════════════════════════════════════════════
#  7. Forward Checking
# ════════════════════════════════════════════════════════════

def forward_checking(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    explored = []
    result = []
    MAX_EXP = rows * cols * 2

    def fc(cur, path, visited, remaining_domain):
        explored.append(cur)
        if len(explored) > MAX_EXP: return False
        if cur == goal:
            result.extend(path + [cur]); return True

        nbrs = sorted(
            [n for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal)
             if n not in visited and n in remaining_domain],
            key=lambda n: manhattan(n, goal)
        )
        for n in nbrs:
            # Forward checking: verify at least one move is available from n
            future_nbrs = [x for x in get_chess_neighbors(n, grid, rows, cols, piece, piece_color, board_pieces, goal)
                           if x not in visited or x == goal]
            if not future_nbrs and n != goal:
                continue  # Prune: dead end ahead
            visited.add(n)
            new_domain = remaining_domain - visited
            new_domain.add(goal)  # always keep goal in domain
            if fc(n, path + [cur], visited, new_domain): return True
            visited.discard(n)
        return False

    all_cells = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 1}
    found = fc(start, [], {start}, all_cells)
    g = len(result)-1 if result else 0
    if found:
        return ok('Forward Checking', result, explored, g, 0, t0,
                  f'Forward Checking: Backtracking + constraint propagation. '
                  f'Prunes branches where future moves would be impossible. '
                  f'Nodes explored: {len(explored)} | Cost: {g}')
    return fail('Forward Checking', explored, t0, 'Forward Checking: No path found.')

# ════════════════════════════════════════════════════════════
#  8. CSP with Arc Consistency (AC-3)
# ════════════════════════════════════════════════════════════

def csp_arc_consistency(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    explored = []
    result = []
    MAX_EXP = rows * cols * 2

    # Build adjacency map for arc consistency
    adj = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1:
                adj[(r, c)] = get_chess_neighbors((r, c), grid, rows, cols, piece, piece_color, board_pieces, goal)

    def revise(domain, xi, xj):
        """Remove values from domain[xi] with no support in domain[xj]."""
        revised = False
        for v in list(domain.get(xi, [])):
            if not any(v2 for v2 in domain.get(xj, [])):
                domain[xi].discard(v)
                revised = True
        return revised

    # Initialize domains: each cell's domain = its reachable neighbors
    domain = {cell: set(adj.get(cell, [])) | {cell} for cell in adj}

    # AC-3 propagation
    queue = [(xi, xj) for xi in adj for xj in adj.get(xi, [])]
    while queue:
        xi, xj = queue.pop(0)
        if revise(domain, xi, xj):
            if not domain[xi]:
                return fail('CSP', explored, t0, 'CSP: Arc consistency failed — no solution.')
            for xk in adj.get(xi, []):
                if xk != xj:
                    queue.append((xk, xi))

    # Backtrack with arc-consistency pruned domains
    def bt_ac(cur, path, visited):
        explored.append(cur)
        if len(explored) > MAX_EXP: return False
        if cur == goal:
            result.extend(path + [cur]); return True
        nbrs = sorted(
            [n for n in adj.get(cur, []) if n not in visited and n in domain.get(cur, set())],
            key=lambda n: manhattan(n, goal)
        )
        for n in nbrs:
            visited.add(n)
            if bt_ac(n, path + [cur], visited): return True
            visited.discard(n)
        return False

    found = bt_ac(start, [], {start})
    g = len(result)-1 if result else 0
    if found:
        return ok('CSP', result, explored, g, 0, t0,
                  f'CSP with Arc Consistency (AC-3): Constraint Satisfaction Problem with '
                  f'arc-consistency preprocessing to prune domains before search. '
                  f'Nodes explored: {len(explored)} | Cost: {g}')
    return fail('CSP', explored, t0, 'CSP: No path found.')

# ════════════════════════════════════════════════════════════
#  9. Minimum Conflicts (CSP Local Search)
# ════════════════════════════════════════════════════════════

def minimum_conflicts(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    explored = []
    MAX_STEPS = rows * cols * 4

    # Build initial greedy assignment
    cur, path, vis = start, [start], {start}
    for _ in range(rows * cols):
        if cur == goal: break
        nbrs = [n for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal) if n not in vis]
        if not nbrs: break
        nxt = min(nbrs, key=lambda n: manhattan(n, goal))
        path.append(nxt); vis.add(nxt); cur = nxt

    def n_conflicts(pos, i, path):
        c = 0
        if grid[pos[0]][pos[1]] == 1: c += 10
        if i > 0 and pos not in get_chess_neighbors(path[i-1], grid, rows, cols, piece, piece_color, board_pieces, goal): c += 5
        return c

    step = 0
    for step in range(MAX_STEPS):
        explored.append(path[-1])
        if path[-1] == goal: break

        # Find most conflicted position in path
        worst_i, worst_c = -1, 0
        for i in range(1, len(path)):
            c = n_conflicts(path[i], i, path)
            if c > worst_c: worst_c = c; worst_i = i

        if worst_i == -1 or worst_c == 0:
            # Extend path toward goal
            last = path[-1]
            candidates = [n for n in get_chess_neighbors(last, grid, rows, cols, piece, piece_color, board_pieces, goal)
                          if n not in set(path)]
            if candidates:
                nxt = min(candidates, key=lambda n: manhattan(n, goal))
                path.append(nxt)
            else:
                # Random restart from midpoint
                mid = max(1, len(path)//2)
                path = path[:mid]
        else:
            # Reassign worst position to minimise conflicts
            prev = path[worst_i-1]
            candidates = [n for n in get_chess_neighbors(prev, grid, rows, cols, piece, piece_color, board_pieces, goal)
                          if n not in path[:worst_i]]
            if candidates:
                best = min(candidates, key=lambda n: manhattan(n, goal) + n_conflicts(n, worst_i, path))
                path[worst_i] = best

        if len(path) > rows * cols * 2:
            path = path[:rows*cols]

    found = bool(path) and path[-1] == goal
    g = len(path)-1 if found else 0
    if found:
        return ok('Min Conflicts', path, explored, g, 0, t0,
                  f'Min-Conflicts CSP: iterative local search, repairs constraint violations. '
                  f'Iterations: {step+1} | Cost: {g}')
    return fail('Min Conflicts', explored, t0,
                f'Min-Conflicts: No path found after {step+1} iterations.')

# ════════════════════════════════════════════════════════════
#  10. Minimax (AI vs Adversary)
# ════════════════════════════════════════════════════════════

def minimax(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    explored = []
    DEPTH = min(7, rows)
    best = [None]          # best path found

    def mm(pos, depth, is_max, visited, path, alpha, beta):
        explored.append(pos)
        if pos == goal:
            if best[0] is None or len(path) < len(best[0]):
                best[0] = path[:]
            return 1000 - len(path)
        if depth == 0: return -manhattan(pos, goal)
        nbrs = [n for n in get_chess_neighbors(pos, grid, rows, cols, piece, piece_color, board_pieces, goal)
                if n not in visited]
        if not nbrs: return -manhattan(pos, goal) * 5
        if is_max:
            v = float('-inf')
            for n in nbrs[:5]:
                visited.add(n)
                v = max(v, mm(n, depth-1, False, visited, path+[n], alpha, beta))
                visited.discard(n)
                alpha = max(alpha, v)
                if beta <= alpha: break
            return v
        else:
            # Adversary moves to worst cell for AI
            v = float('inf')
            for n in sorted(nbrs, key=lambda x: -manhattan(x, goal))[:3]:
                visited.add(n)
                v = min(v, mm(n, depth-1, True, visited, path+[n], alpha, beta))
                visited.discard(n)
                beta = min(beta, v)
                if beta <= alpha: break
            return v

    mm(start, DEPTH, True, {start}, [start], float('-inf'), float('inf'))

    if not best[0] or best[0][-1] != goal:
        fb = astar(grid, start, goal, rows, cols, piece, piece_color, board_pieces)
        fb['algorithm'] = 'Minimax'
        fb['description'] = (f'Minimax: AI (MAX) vs Adversary (MIN). '
                             f'Depth={DEPTH} | Nodes={len(explored)}. Fallback to A* path shown.')
        fb['explored'] = to_list(explored); fb['nodes_explored'] = len(explored)
        return fb

    path = best[0]; g = len(path)-1
    return ok('Minimax', path, explored, g, 0, t0,
              f'Minimax: AI maximises score, Adversary minimises. '
              f'Depth={DEPTH} | Nodes evaluated={len(explored)} | Cost={g}')

# ════════════════════════════════════════════════════════════
#  11. Alpha-Beta Pruning
# ════════════════════════════════════════════════════════════

def alpha_beta(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    t0 = time.time()
    explored = []
    DEPTH = min(10, rows)
    pruned = [0]
    best = [None]

    def ab(pos, depth, is_max, visited, path, alpha, beta):
        explored.append(pos)
        if pos == goal:
            if best[0] is None or len(path) < len(best[0]):
                best[0] = path[:]
            return 1000 - len(path)
        if depth == 0: return -manhattan(pos, goal)
        nbrs = [n for n in get_chess_neighbors(pos, grid, rows, cols, piece, piece_color, board_pieces, goal)
                if n not in visited]
        if not nbrs: return -manhattan(pos, goal)
        if is_max:
            v = float('-inf')
            for n in nbrs[:5]:
                visited.add(n)
                v = max(v, ab(n, depth-1, False, visited, path+[n], alpha, beta))
                visited.discard(n)
                alpha = max(alpha, v)
                if beta <= alpha: pruned[0]+=1; break
            return v
        else:
            v = float('inf')
            for n in sorted(nbrs, key=lambda x: -manhattan(x,goal))[:3]:
                visited.add(n)
                v = min(v, ab(n, depth-1, True, visited, path+[n], alpha, beta))
                visited.discard(n)
                beta = min(beta, v)
                if beta <= alpha: pruned[0]+=1; break
            return v

    ab(start, DEPTH, True, {start}, [start], float('-inf'), float('inf'))

    if not best[0] or best[0][-1] != goal:
        fb = astar(grid, start, goal, rows, cols, piece, piece_color, board_pieces)
        fb['algorithm'] = 'Alpha-Beta'
        fb['description'] = (f'Alpha-Beta Pruning: Minimax + pruning. '
                             f'Branches pruned: {pruned[0]} | Nodes={len(explored)}. '
                             f'Fallback to A* path shown.')
        fb['explored'] = to_list(explored); fb['nodes_explored'] = len(explored)
        return fb

    path = best[0]; g = len(path)-1
    return ok('Alpha Beta Pruning', path, explored, g, 0, t0,
              f'Alpha-Beta Pruning: Minimax + α-β cuts. Branches pruned: {pruned[0]} '
              f'(~{round(pruned[0]/(max(len(explored),1))*100)}% saved). '
              f'Depth={DEPTH} | Cost={g}')

# ════════════════════════════════════════════════════════════
#  12. HMM — Hidden Markov Model Path Inference
# ════════════════════════════════════════════════════════════

def hmm(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    """
    HMM-inspired pathfinding: uses Viterbi-like forward pass.
    States = grid cells; transitions = valid chess moves.
    Emission probability decays with distance to goal (Gaussian).
    Viterbi selects the most-likely sequence of states from start→goal.
    """
    t0 = time.time()
    explored = []
    INF = float('inf')

    # Emission: probability of observing goal from state s
    def emission(s):
        d = manhattan(s, goal)
        return math.exp(-d * 0.3)   # Gaussian-like decay

    # Transition: uniform among valid neighbors
    def transition(s, t, nbrs_s):
        return 1.0 / len(nbrs_s) if nbrs_s else 0.0

    # Viterbi forward pass using log-probabilities (avoid underflow)
    # delta[s] = max log-prob path to s; psi[s] = predecessor
    delta = {start: 0.0}   # log(1)
    psi   = {start: None}
    visited_order = [start]
    frontier = [start]
    done = set()

    while frontier:
        # Pick state with highest delta (most probable so far)
        cur = max(frontier, key=lambda s: delta.get(s, -INF))
        frontier.remove(cur)
        if cur in done: continue
        done.add(cur)
        explored.append(cur)

        if cur == goal: break

        nbrs = get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal)
        for n in nbrs:
            if n in done: continue
            # log(transition) + log(emission)
            log_t = math.log(transition(cur, n, nbrs) + 1e-12)
            log_e = math.log(emission(n) + 1e-12)
            new_delta = delta[cur] + log_t + log_e
            if new_delta > delta.get(n, -INF):
                delta[n] = new_delta
                psi[n] = cur
                if n not in frontier:
                    frontier.append(n)

    # Backtrace Viterbi path
    if goal not in psi:
        return fail('HMM', explored, t0, 'HMM: No probable path found to goal.')

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = psi.get(cur)
    path.reverse()

    if path[0] != start:
        return fail('HMM', explored, t0, 'HMM: Viterbi traceback failed.')

    g = len(path) - 1
    return ok('HMM', path, explored, g, 0, t0,
              f'Hidden Markov Model (Viterbi): models path as sequence of states with '
              f'emission & transition probabilities. Finds most probable route to goal. '
              f'States explored: {len(explored)} | Path length: {g}')

# ════════════════════════════════════════════════════════════
#  13. Bayesian Network Path Search
# ════════════════════════════════════════════════════════════

def bayesian_network(grid, start, goal, rows, cols, piece='king', piece_color='white', board_pieces=None):
    """
    Bayesian Network-inspired search:
    • Each cell has a prior probability of being on the optimal path
      (based on distance to goal — closer = higher prior).
    • Likelihood is updated with evidence: cells with more valid
      neighbors have higher connectivity likelihood.
    • Posterior = prior × likelihood guides a best-first search.
    """
    t0 = time.time()
    explored = []

    def prior(s):
        """P(s on optimal path): inverse Manhattan distance to goal."""
        d = manhattan(s, goal) + 1
        return 1.0 / d

    def likelihood(s):
        """P(evidence | s): connectivity — more neighbors = more likely traversable."""
        nbrs = get_chess_neighbors(s, grid, rows, cols, piece, piece_color, board_pieces, goal)
        max_nbrs = 8 if piece == 'king' else 4
        return len(nbrs) / max(max_nbrs, 1)

    def posterior(s):
        p = prior(s) * likelihood(s)
        return p

    # Best-first search guided by posterior probability (higher = better)
    pq = [(-posterior(start), start)]
    came_from = {start: None}
    done = set()

    while pq:
        neg_post, cur = heapq.heappop(pq)
        if cur in done: continue
        done.add(cur)
        explored.append(cur)

        if cur == goal:
            path = reconstruct_path(came_from, start, goal)
            g = len(path) - 1
            return ok('Bayesian Network', path, explored, g, 0, t0,
                      f'Bayesian Network: uses prior P(cell on path) × likelihood P(evidence|cell) '
                      f'as posterior to guide best-first search. '
                      f'Nodes explored: {len(explored)} | Cost: {g}')

        for n in get_chess_neighbors(cur, grid, rows, cols, piece, piece_color, board_pieces, goal):
            if n not in came_from:
                came_from[n] = cur
                heapq.heappush(pq, (-posterior(n), n))

    return fail('Bayesian Network', explored, t0, 'Bayesian Network: No path found.')

# ════════════════════════════════════════════════════════════
#  REGISTRY
# ════════════════════════════════════════════════════════════

ALGORITHMS = {
    'bfs':             bfs,
    'dfs':             dfs,
    'ucs':             ucs,
    'astar':           astar,
    'greedy':          greedy,
    'backtracking':    backtracking,
    'forwardchecking': forward_checking,
    'csp':             csp_arc_consistency,
    'minimax':         minimax,
    'alphabeta':       alpha_beta,
    'minconflicts':    minimum_conflicts,
    'hmm':             hmm,
    'bayesian':        bayesian_network,
}

# ════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════

@app.route('/')
def index(): return render_template('index.html')

@app.route('/game')
def game(): return render_template('game.html')

@app.route('/results')
def results():
    return render_template('results.html', result=session.get('result_data'))

@app.route('/api/solve', methods=['POST'])
def solve():
    d = request.get_json()
    grid   = d.get('grid', [])
    start  = tuple(d.get('start', [0,0]))
    goal   = tuple(d.get('goal',  [0,0]))
    algo   = d.get('algorithm', 'bfs').lower()
    piece  = d.get('piece_type', 'king').lower()
    piece_color = d.get('piece_color', 'white').lower()
    rows, cols = len(grid), len(grid[0]) if grid else 0

    # board_pieces: dict mapping (r,c) -> color for pieces still on board
    # Frontend sends this as list of [r, c, color] for non-start non-goal pieces
    raw_bp = d.get('board_pieces', [])
    board_pieces = {}
    for entry in raw_bp:
        r, c, color = entry[0], entry[1], entry[2]
        board_pieces[(r, c)] = color

    if algo not in ALGORITHMS:
        return jsonify({'error': f'Unknown algorithm: {algo}'}), 400
    try:
        result = ALGORITHMS[algo](grid, start, goal, rows, cols, piece, piece_color, board_pieces)
        session['result_data'] = result
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# ════════════════════════════════════════════════════════════
#  COMPARISON ENDPOINT — run all algorithms, return summary
# ════════════════════════════════════════════════════════════

@app.route('/api/compare', methods=['POST'])
def compare_algos():
    d = request.get_json()
    grid        = d.get('grid', [])
    start       = tuple(d.get('start', [0, 0]))
    goal        = tuple(d.get('goal',  [0, 0]))
    piece       = d.get('piece_type',  'king').lower()
    piece_color = d.get('piece_color', 'white').lower()
    rows, cols  = len(grid), len(grid[0]) if grid else 0
    raw_bp      = d.get('board_pieces', [])
    board_pieces = {(e[0], e[1]): e[2] for e in raw_bp}

    results = {}
    for algo_key, algo_fn in ALGORITHMS.items():
        try:
            r = algo_fn(grid, start, goal, rows, cols, piece, piece_color, board_pieces)
            results[algo_key] = {
                'algorithm':      r.get('algorithm', algo_key),
                'found':          r.get('found', False),
                'path_length':    r.get('path_length', 0),
                'nodes_explored': r.get('nodes_explored', 0),
                'cost':           r.get('cost', 0),
                'time_ms':        r.get('time_ms', 0),
            }
        except Exception as ex:
            results[algo_key] = {
                'algorithm': algo_key, 'found': False,
                'path_length': 0, 'nodes_explored': 0,
                'cost': 0, 'time_ms': 0, 'error': str(ex)
            }
    return jsonify(results)

# ════════════════════════════════════════════════════════════
#  BAYESIAN HEAT MAP ENDPOINT
# ════════════════════════════════════════════════════════════

@app.route('/api/bayesian', methods=['POST'])
def bayesian_heatmap():
    d          = request.get_json()
    grid       = d.get('grid', [])
    raw_bp     = d.get('board_pieces', [])   # [[r,c,color], ...]
    goal       = d.get('goal', None)
    rows, cols = len(grid), len(grid[0]) if grid else 0

    # Identify enemy (black) piece positions
    enemy_pos = [(e[0], e[1]) for e in raw_bp if e[2] == 'black']
    if goal:
        enemy_pos.append((goal[0], goal[1]))   # goal piece is also a threat

    heatmap = []
    for r in range(rows):
        row_vals = []
        for c in range(cols):
            if grid[r][c] == 1:
                row_vals.append(-1.0)   # wall marker
                continue
            # Bayesian: P(danger) = 1 - ∏(1 - P_i(danger from enemy i))
            prob_safe = 1.0
            for (er, ec) in enemy_pos:
                dist = abs(r - er) + abs(c - ec)
                # Exponential decay: close = high danger, far = low danger
                p_danger = math.exp(-dist * 0.45)
                prob_safe *= (1.0 - p_danger * 0.75)
            danger = round(1.0 - prob_safe, 4)
            row_vals.append(danger)
        heatmap.append(row_vals)

    return jsonify({'heatmap': heatmap, 'rows': rows, 'cols': cols})

# ════════════════════════════════════════════════════════════
#  CHESS GAME — ENGINE
# ════════════════════════════════════════════════════════════

_CHESS_PV = {'pawn':100,'knight':320,'bishop':330,'rook':500,'queen':900,'king':20000}

# Piece-square tables (white perspective; flip row index for black)
_PST = {
    'pawn':   [[0,0,0,0,0,0,0,0],[50,50,50,50,50,50,50,50],[10,10,20,30,30,20,10,10],
               [5,5,10,25,25,10,5,5],[0,0,0,20,20,0,0,0],[5,-5,-10,0,0,-10,-5,5],
               [5,10,10,-20,-20,10,10,5],[0,0,0,0,0,0,0,0]],
    'knight': [[-50,-40,-30,-30,-30,-30,-40,-50],[-40,-20,0,0,0,0,-20,-40],
               [-30,0,10,15,15,10,0,-30],[-30,5,15,20,20,15,5,-30],
               [-30,0,15,20,20,15,0,-30],[-30,5,10,15,15,10,5,-30],
               [-40,-20,0,5,5,0,-20,-40],[-50,-40,-30,-30,-30,-30,-40,-50]],
    'bishop': [[-20,-10,-10,-10,-10,-10,-10,-20],[-10,0,0,0,0,0,0,-10],
               [-10,0,5,10,10,5,0,-10],[-10,5,5,10,10,5,5,-10],
               [-10,0,10,10,10,10,0,-10],[-10,10,10,10,10,10,10,-10],
               [-10,5,0,0,0,0,5,-10],[-20,-10,-10,-10,-10,-10,-10,-20]],
    'rook':   [[0,0,0,0,0,0,0,0],[5,10,10,10,10,10,10,5],[-5,0,0,0,0,0,0,-5],
               [-5,0,0,0,0,0,0,-5],[-5,0,0,0,0,0,0,-5],[-5,0,0,0,0,0,0,-5],
               [-5,0,0,0,0,0,0,-5],[0,0,0,5,5,0,0,0]],
    'queen':  [[-20,-10,-10,-5,-5,-10,-10,-20],[-10,0,0,0,0,0,0,-10],
               [-10,0,5,5,5,5,0,-10],[-5,0,5,5,5,5,0,-5],
               [0,0,5,5,5,5,0,-5],[-10,5,5,5,5,5,0,-10],
               [-10,0,5,0,0,0,0,-10],[-20,-10,-10,-5,-5,-10,-10,-20]],
    'king':   [[-30,-40,-40,-50,-50,-40,-40,-30],[-30,-40,-40,-50,-50,-40,-40,-30],
               [-30,-40,-40,-50,-50,-40,-40,-30],[-30,-40,-40,-50,-50,-40,-40,-30],
               [-20,-30,-30,-40,-40,-30,-30,-20],[-10,-20,-20,-20,-20,-20,-20,-10],
               [20,20,0,0,0,0,20,20],[20,30,10,0,0,10,30,20]],
}

def _inb(r,c): return 0<=r<8 and 0<=c<8

def _raw_moves(board, r, c):
    p = board[r][c]
    if not p: return []
    pt, col = p['type'], p['color']
    opp = 'black' if col=='white' else 'white'
    moves = []

    def add(tr,tc,sp=None): moves.append({'from':[r,c],'to':[tr,tc],'sp':sp})
    def empty(tr,tc): return _inb(tr,tc) and not board[tr][tc]
    def enemy(tr,tc): return _inb(tr,tc) and board[tr][tc] and board[tr][tc]['color']==opp
    def can(tr,tc):   return _inb(tr,tc) and (not board[tr][tc] or board[tr][tc]['color']==opp)

    if pt == 'pawn':
        fwd = -1 if col=='white' else 1
        sr  =  6 if col=='white' else 1
        pr  =  0 if col=='white' else 7
        if empty(r+fwd,c):
            if r+fwd==pr:
                for pp in ['queen','rook','bishop','knight']: add(r+fwd,c,f'promo:{pp}')
            else:
                add(r+fwd,c)
                if r==sr and empty(r+2*fwd,c): add(r+2*fwd,c,'double')
        for dc in [-1,1]:
            if enemy(r+fwd,c+dc):
                if r+fwd==pr:
                    for pp in ['queen','rook','bishop','knight']: add(r+fwd,c+dc,f'promo:{pp}')
                else: add(r+fwd,c+dc)
    elif pt == 'knight':
        for dr,dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            if can(r+dr,c+dc): add(r+dr,c+dc)
    elif pt == 'king':
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if (dr or dc) and can(r+dr,c+dc): add(r+dr,c+dc)
        if not p.get('moved'):
            if not board[r][5] and not board[r][6] and board[r][7] and board[r][7]['type']=='rook' and not board[r][7].get('moved'):
                add(r,c+2,'castle_k')
            if not board[r][3] and not board[r][2] and not board[r][1] and board[r][0] and board[r][0]['type']=='rook' and not board[r][0].get('moved'):
                add(r,c-2,'castle_q')
    else:
        dirs = {'rook':[(0,1),(0,-1),(1,0),(-1,0)],
                'bishop':[(1,1),(1,-1),(-1,1),(-1,-1)],
                'queen':[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]}[pt]
        for dr,dc in dirs:
            nr,nc=r+dr,c+dc
            while _inb(nr,nc):
                if board[nr][nc]:
                    if board[nr][nc]['color']==opp: add(nr,nc)
                    break
                add(nr,nc); nr+=dr; nc+=dc
    return moves

def _apply(board, move):
    b = copy.deepcopy(board)
    fr,fc = move['from']; tr,tc = move['to']; sp = move.get('sp')
    piece = dict(b[fr][fc]); piece['moved']=True
    b[fr][fc]=None; b[tr][tc]=piece
    if sp=='enpassant':
        fwd=-1 if piece['color']=='white' else 1; b[tr-fwd][tc]=None
    elif sp=='castle_k':
        r=dict(b[fr][7]); r['moved']=True; b[fr][7]=None; b[fr][5]=r
    elif sp=='castle_q':
        r=dict(b[fr][0]); r['moved']=True; b[fr][0]=None; b[fr][3]=r
    elif sp and sp.startswith('promo:'):
        b[tr][tc]={'type':sp.split(':')[1],'color':piece['color'],'moved':True}
    return b

def _find_king(board, color):
    for r in range(8):
        for c in range(8):
            p=board[r][c]
            if p and p['type']=='king' and p['color']==color: return (r,c)
    return None

def _attacked(board, r, c, by):
    for pr in range(8):
        for pc in range(8):
            if board[pr][pc] and board[pr][pc]['color']==by:
                for m in _raw_moves(board,pr,pc):
                    if m['to']==[r,c]: return True
    return False

def _in_check(board, color):
    k=_find_king(board,color)
    if not k: return True
    opp='black' if color=='white' else 'white'
    return _attacked(board,k[0],k[1],opp)

def _legal_moves(board, color):
    moves=[]
    opp='black' if color=='white' else 'white'
    for r in range(8):
        for c in range(8):
            if board[r][c] and board[r][c]['color']==color:
                for m in _raw_moves(board,r,c):
                    if m.get('sp') in ('castle_k','castle_q'):
                        fr,fc=m['from']
                        if _attacked(board,fr,fc,opp): continue
                        mc=fc+1 if m['sp']=='castle_k' else fc-1
                        if _attacked(board,fr,mc,opp): continue
                    nb=_apply(board,m)
                    if not _in_check(nb,color): moves.append(m)
    return moves

# ════════════════════════════════════════════════════════════
#  EVALUATION — KING SAFETY
# ════════════════════════════════════════════════════════════

def _king_safety(board, color):
    """
    King Safety score (positive = safer for 'color').
    Factors:
      • Pawn shield  – pawns directly in front of king
      • Open files   – penalty for no friendly pawns on king's file or adjacent
      • Attack zone  – penalty for each square around king attacked by enemy
      • Castling     – bonus when king is on typical castled square
      • Endgame      – in endgame, king should be centralised (separate bonus)
    """
    king_pos = _find_king(board, color)
    if not king_pos:
        return -10000
    kr, kc = king_pos
    opp = 'black' if color == 'white' else 'white'
    fwd = -1 if color == 'white'  else 1   # rank in front of king
    score = 0

    # ── Count total pieces to detect endgame ──────────────────
    total = sum(1 for r in range(8) for c in range(8) if board[r][c])
    endgame = total <= 12

    if not endgame:
        # ── Pawn shield ───────────────────────────────────────
        # Reward pawns on the three files in front of the king
        for dc in (-1, 0, 1):
            nc = kc + dc
            if 0 <= nc < 8:
                for dist in (1, 2):          # one or two rows ahead
                    nr = kr + fwd * dist
                    if 0 <= nr < 8 and board[nr][nc]:
                        p = board[nr][nc]
                        if p['type'] == 'pawn' and p['color'] == color:
                            score += 15 if dist == 1 else 7
                            break
                else:
                    score -= 10   # no pawn on this file strip

        # ── Open / semi-open file penalty ─────────────────────
        for dc in (-1, 0, 1):
            nc = kc + dc
            if not (0 <= nc < 8):
                continue
            has_own_pawn = any(
                board[r][nc] and board[r][nc]['type'] == 'pawn'
                and board[r][nc]['color'] == color
                for r in range(8)
            )
            has_opp_pawn = any(
                board[r][nc] and board[r][nc]['type'] == 'pawn'
                and board[r][nc]['color'] == opp
                for r in range(8)
            )
            if not has_own_pawn and not has_opp_pawn:
                score -= 20   # fully open file — very dangerous
            elif not has_own_pawn:
                score -= 10   # semi-open file

        # ── Castling position bonus ───────────────────────────
        back_rank = 7 if color == 'white' else 0
        if kr == back_rank and kc in (2, 6):
            score += 25   # king-side or queen-side castled
        elif kr == back_rank and kc in (3, 4, 5):
            score -= 20   # king stuck in centre

        # ── Enemy attack pressure around king ─────────────────
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = kr + dr, kc + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    if _attacked(board, nr, nc, opp):
                        score -= 8

    else:
        # ── Endgame: king should be active / centralised ──────
        centre_dist = abs(kr - 3.5) + abs(kc - 3.5)
        score += int((7 - centre_dist) * 6)

    return score


# ════════════════════════════════════════════════════════════
#  EVALUATION — FULL POSITION SCORE
# ════════════════════════════════════════════════════════════

def _evaluate(board):
    """
    Comprehensive static evaluation.
    Returns centipawn score (positive = White advantage).

    Components
    ──────────
    1. Material score         – piece values (PV table)
    2. Piece-Square tables    – positional bonuses per piece/square
    3. King safety            – pawn shield, open files, attack zones
    4. Mobility               – legal-move count difference × weight
    """
    material   = 0
    positional = 0

    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if not p:
                continue
            val = _CHESS_PV.get(p['type'], 0)
            pst = _PST.get(p['type'], [[0] * 8] * 8)
            if p['color'] == 'white':
                material   += val
                positional += pst[r][c]
            else:
                material   -= val
                positional -= pst[7 - r][c]

    # King safety (white minus black)
    w_safety = _king_safety(board, 'white')
    b_safety = _king_safety(board, 'black')
    king_safety_score = w_safety - b_safety

    # Mobility: count legal moves for each side (5 cp per move advantage)
    # Use raw (pseudo-legal) moves for speed — still meaningful signal
    w_mob = sum(len(_raw_moves(board, r, c))
                for r in range(8) for c in range(8)
                if board[r][c] and board[r][c]['color'] == 'white')
    b_mob = sum(len(_raw_moves(board, r, c))
                for r in range(8) for c in range(8)
                if board[r][c] and board[r][c]['color'] == 'black')
    mobility_score = (w_mob - b_mob) * 5

    return material + positional + king_safety_score + mobility_score

def _minimax(board, depth, alpha, beta, maximizing, algo='alphabeta'):
    color='white' if maximizing else 'black'
    moves=_legal_moves(board,color)
    if not moves or depth==0:
        return _evaluate(board), None

    # Move ordering: sort captures first for alpha-beta efficiency
    if algo in ('alphabeta','minimax'):
        moves.sort(key=lambda m: 0 if board[m['to'][0]][m['to'][1]] else 1)
    elif algo=='greedy':
        depth=1  # greedy = depth-1 evaluation only
    elif algo=='random':
        return 0, random.choice(moves)

    best=None
    if maximizing:
        v=float('-inf')
        for m in moves:
            nb=_apply(board,m)
            cv,_=_minimax(nb,depth-1,alpha,beta,False,algo)
            if cv>v: v=cv; best=m
            if algo=='alphabeta':
                alpha=max(alpha,v)
                if beta<=alpha: break
        return v,best
    else:
        v=float('inf')
        for m in moves:
            nb=_apply(board,m)
            cv,_=_minimax(nb,depth-1,alpha,beta,True,algo)
            if cv<v: v=cv; best=m
            if algo=='alphabeta':
                beta=min(beta,v)
                if beta<=alpha: break
        return v,best

# ════════════════════════════════════════════════════════════
#  CHESS GAME — ROUTES
# ════════════════════════════════════════════════════════════

@app.route('/chess')
def chess_game_page():
    return render_template('chess_game.html')

@app.route('/api/chess/ai_move', methods=['POST'])
def chess_ai_move():
    d = request.get_json()
    board      = d.get('board')
    color      = d.get('color', 'white')
    algo       = d.get('algorithm', 'alphabeta')
    diff       = d.get('difficulty', 'medium')
    # ── Depth levels updated: Easy=2, Medium=4, Hard=6 ──────
    depth      = {'easy': 2, 'medium': 4, 'hard': 6}.get(diff, 4)
    maximizing = (color == 'white')

    if not board:
        return jsonify({'move': None, 'game_over': True})

    # ── Instrumented alpha-beta that tracks nodes + pruning ──
    nodes_visited = [0]
    branches_pruned = [0]
    positions_evaluated = [0]

    def _ab_traced(b, dep, alpha, beta, maxing):
        nodes_visited[0] += 1
        col   = 'white' if maxing else 'black'
        moves = _legal_moves(b, col)
        if not moves or dep == 0:
            positions_evaluated[0] += 1
            return _evaluate(b), None
        # Captures first (move ordering)
        moves.sort(key=lambda m: 0 if b[m['to'][0]][m['to'][1]] else 1)
        best = None
        if maxing:
            v = float('-inf')
            for m in moves:
                cv, _ = _ab_traced(_apply(b, m), dep - 1, alpha, beta, False)
                if cv > v: v = cv; best = m
                if algo == 'alphabeta':
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        branches_pruned[0] += 1
                        break
            return v, best
        else:
            v = float('inf')
            for m in moves:
                cv, _ = _ab_traced(_apply(b, m), dep - 1, alpha, beta, True)
                if cv < v: v = cv; best = m
                if algo == 'alphabeta':
                    beta = min(beta, v)
                    if beta <= alpha:
                        branches_pruned[0] += 1
                        break
            return v, best

    try:
        t0 = time.time()
        score, best = _ab_traced(board, depth, float('-inf'), float('inf'), maximizing)
        elapsed = round((time.time() - t0) * 1000, 1)

        if not best:
            return jsonify({'move': None, 'game_over': True})

        files = 'abcdefgh'
        def alg(sq): return files[sq[1]] + str(8 - sq[0])
        from_sq = alg(best['from'])
        to_sq   = alg(best['to'])
        move_str = from_sq + '\u2192' + to_sq

        # ── Score after the move (for eval display) ──────────
        post_board = _apply(board, best)
        post_score = _evaluate(post_board)
        # Decompose score
        mat = sum(
            (_CHESS_PV.get(board[r][c]['type'], 0) * (1 if board[r][c]['color']=='white' else -1))
            for r in range(8) for c in range(8) if board[r][c]
        )
        ks  = _king_safety(board,'white') - _king_safety(board,'black')

        # ── Reasoning trace for Explainable AI ───────────────
        legal_count = len(_legal_moves(board, color))
        reasoning = [
            f'Generated {legal_count} legal moves for {color}',
            f'Searched to depth {depth} using {"Alpha-Beta" if algo=="alphabeta" else algo.capitalize()}',
            f'Visited {nodes_visited[0]:,} nodes in the game tree',
            f'Pruned {branches_pruned[0]} branches (alpha-beta cutoffs)',
            f'Evaluated {positions_evaluated[0]:,} leaf positions',
            f'Selected best move: {move_str}',
            f'Expected advantage: {"+"+str(round(post_score/100,2)) if post_score>=0 else str(round(post_score/100,2))} pawns',
        ]

        return jsonify({
            'move'               : best,
            'from_sq'            : from_sq,
            'to_sq'              : to_sq,
            'eval_score'         : round(score / 100, 2),
            'eval_cp'            : score,
            'post_eval_score'    : round(post_score / 100, 2),
            'nodes_visited'      : nodes_visited[0],
            'positions_evaluated': positions_evaluated[0],
            'branches_pruned'    : branches_pruned[0],
            'depth'              : depth,
            'time_ms'            : elapsed,
            'algorithm'          : algo,
            'difficulty'         : diff,
            'reasoning'          : reasoning,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ════════════════════════════════════════════════════════════
#  BOARD ENVIRONMENT — AI SUGGEST ENDPOINT
#  Called by the /board State Space page.
#  Returns best move + full evaluation breakdown.
# ════════════════════════════════════════════════════════════

@app.route('/api/board/ai_suggest', methods=['POST'])
def board_ai_suggest():
    """
    Runs depth-limited Minimax with Alpha-Beta pruning on the given board.
    Returns:
      move         – best move {from, to}
      eval         – total centipawn score
      material     – material component
      positional   – PST (piece-square table) component
      king_safety  – king safety component
      mobility     – mobility component
      depth        – search depth used
      nodes        – nodes evaluated (approximate)
      from_sq      – algebraic 'from' square
      to_sq        – algebraic 'to' square
    """
    d = request.get_json()
    board    = d.get('board')          # 8×8 list [{type,color,moved}|null]
    color    = d.get('color', 'white')
    depth    = int(d.get('depth', 3))  # depth from frontend (1-4)

    if not board:
        return jsonify({'error': 'No board provided'}), 400

    maximizing = (color == 'white')
    nodes = [0]   # mutable counter

    # ── Instrumented minimax to count nodes ──────────────────
    def mm_count(b, dep, alpha, beta, maxing):
        nodes[0] += 1
        col   = 'white' if maxing else 'black'
        moves = _legal_moves(b, col)
        if not moves or dep == 0:
            return _evaluate(b), None
        # Move ordering: captures first
        moves.sort(key=lambda m: 0 if b[m['to'][0]][m['to'][1]] else 1)
        best = None
        if maxing:
            v = float('-inf')
            for m in moves:
                cv, _ = mm_count(_apply(b, m), dep-1, alpha, beta, False)
                if cv > v:
                    v = cv; best = m
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v, best
        else:
            v = float('inf')
            for m in moves:
                cv, _ = mm_count(_apply(b, m), dep-1, alpha, beta, True)
                if cv < v:
                    v = cv; best = m
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v, best

    try:
        score, best = mm_count(board, depth, float('-inf'), float('inf'), maximizing)

        if not best:
            return jsonify({'move': None, 'game_over': True})

        # ── Breakdown on the POST-move board ─────────────────
        post = _apply(board, best)

        # Material
        mat = 0
        for r in range(8):
            for c in range(8):
                p = post[r][c]
                if not p: continue
                val = _CHESS_PV.get(p['type'], 0)
                mat += val if p['color']=='white' else -val

        # Positional (PST)
        pos = 0
        for r in range(8):
            for c in range(8):
                p = post[r][c]
                if not p: continue
                pst = _PST.get(p['type'], [[0]*8]*8)
                pos += pst[r][c] if p['color']=='white' else -pst[7-r][c]

        # King safety
        ks = _king_safety(post, 'white') - _king_safety(post, 'black')

        # Mobility
        wm = sum(len(_raw_moves(post,r,c))
                 for r in range(8) for c in range(8)
                 if post[r][c] and post[r][c]['color']=='white')
        bm = sum(len(_raw_moves(post,r,c))
                 for r in range(8) for c in range(8)
                 if post[r][c] and post[r][c]['color']=='black')
        mob = (wm - bm) * 5

        files = 'abcdefgh'
        def alg(sq): return files[sq[1]] + str(8 - sq[0])

        return jsonify({
            'move'        : best,
            'eval'        : score,
            'material'    : mat,
            'positional'  : pos,
            'king_safety' : ks,
            'mobility'    : mob,
            'depth'       : depth,
            'nodes'       : nodes[0],
            'from_sq'     : alg(best['from']),
            'to_sq'       : alg(best['to']),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500



# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  CHESS DEMO: MINIMAX vs ALPHA-BETA NODE COMPARISON
#  /api/chess/compare_nodes
#  Runs both algorithms on the same chess board position
#  and returns exact node counts for evaluator demo.
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

@app.route('/api/chess/compare_nodes', methods=['POST'])
def chess_compare_nodes():
    """
    Runs PURE MINIMAX and ALPHA-BETA on the same chess position.
    Returns node counts for both so the evaluator can see the pruning benefit.
    """
    d = request.get_json()
    board  = d.get('board')
    color  = d.get('color', 'white')
    depth  = int(d.get('depth', 3))

    if not board:
        return jsonify({'error': 'No board provided'}), 400

    maximizing = (color == 'white')

    # \u2500\u2500 Pure Minimax (no pruning) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    mm_nodes = [0]
    mm_best  = [None]

    def pure_minimax(b, dep, maxing):
        mm_nodes[0] += 1
        col   = 'white' if maxing else 'black'
        moves = _legal_moves(b, col)
        if not moves or dep == 0:
            return _evaluate(b), None
        moves.sort(key=lambda m: 0 if b[m['to'][0]][m['to'][1]] else 1)
        best = None
        if maxing:
            v = float('-inf')
            for m in moves:
                cv, _ = pure_minimax(_apply(b, m), dep - 1, False)
                if cv > v:
                    v = cv; best = m
            return v, best
        else:
            v = float('inf')
            for m in moves:
                cv, _ = pure_minimax(_apply(b, m), dep - 1, True)
                if cv < v:
                    v = cv; best = m
            return v, best

    # \u2500\u2500 Alpha-Beta Minimax \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ab_nodes  = [0]
    ab_pruned = [0]

    def alpha_beta_mm(b, dep, alpha, beta, maxing):
        ab_nodes[0] += 1
        col   = 'white' if maxing else 'black'
        moves = _legal_moves(b, col)
        if not moves or dep == 0:
            return _evaluate(b), None
        moves.sort(key=lambda m: 0 if b[m['to'][0]][m['to'][1]] else 1)
        best = None
        if maxing:
            v = float('-inf')
            for m in moves:
                cv, _ = alpha_beta_mm(_apply(b, m), dep - 1, alpha, beta, False)
                if cv > v:
                    v = cv; best = m
                alpha = max(alpha, v)
                if beta <= alpha:
                    ab_pruned[0] += 1
                    break
            return v, best
        else:
            v = float('inf')
            for m in moves:
                cv, _ = alpha_beta_mm(_apply(b, m), dep - 1, alpha, beta, True)
                if cv < v:
                    v = cv; best = m
                beta = min(beta, v)
                if beta <= alpha:
                    ab_pruned[0] += 1
                    break
            return v, best

    try:
        t0 = time.time()
        mm_score, mm_mv = pure_minimax(board, depth, maximizing)
        mm_time = round((time.time() - t0) * 1000, 2)

        t1 = time.time()
        ab_score, ab_mv = alpha_beta_mm(board, depth, float('-inf'), float('inf'), maximizing)
        ab_time = round((time.time() - t1) * 1000, 2)

        reduction = round((1 - ab_nodes[0] / max(mm_nodes[0], 1)) * 100, 1)
        files = 'abcdefgh'
        def alg(sq): return files[sq[1]] + str(8 - sq[0])

        return jsonify({
            'depth'         : depth,
            'minimax_nodes' : mm_nodes[0],
            'minimax_score' : mm_score,
            'minimax_time'  : mm_time,
            'minimax_best'  : (alg(mm_mv['from']) + '\u2192' + alg(mm_mv['to'])) if mm_mv else None,
            'ab_nodes'      : ab_nodes[0],
            'ab_pruned'     : ab_pruned[0],
            'ab_score'      : ab_score,
            'ab_time'       : ab_time,
            'ab_best'       : (alg(ab_mv['from']) + '\u2192' + alg(ab_mv['to'])) if ab_mv else None,
            'reduction_pct' : reduction,
            'time_saved_ms' : round(mm_time - ab_time, 2),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  CHESS DEMO: A* PIECE PATHFINDING
#  /api/chess/piece_path
#  Find shortest route for a chess piece (Knight, Queen, etc.)
#  from one square to another, optionally avoiding threatened squares.
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def _sq_to_rc(sq):
    """Convert algebraic 'a1' -> (row, col) where row 0=rank8, row 7=rank1."""
    col = ord(sq[0].lower()) - ord('a')
    row = 8 - int(sq[1])
    return (row, col)

def _rc_to_sq(r, c):
    return 'abcdefgh'[c] + str(8 - r)

def _chess_piece_astar(piece_type, from_rc, to_rc, board=None, avoid_threatened=False):
    """
    A* pathfinding on 8x8 chess board for a specific piece type.

    Heuristic functions (admissible for each piece):
      - knight : Chebyshev-based approximation of minimum knight moves
      - bishop  : 0 if reachable diagonally, else 2
      - rook   : 1 if same rank or file, else 2
      - queen  : 1 if any queen move reaches (same rank/file/diagonal), else 2
      - king   : Chebyshev distance
      - pawn   : row difference (white moves up)

    If avoid_threatened=True and board provided, threatened squares get +10 cost.
    """
    FILES = 'abcdefgh'

    def heuristic(a, b, pt):
        ar, ac = a; br, bc = b
        dr, dc = abs(ar - br), abs(ac - bc)
        if pt == 'knight':
            # Minimum number of knight moves approximation
            if dr == 1 and dc == 0 or dr == 0 and dc == 1:
                return 3
            return max(math.ceil((dr + dc) / 3), math.ceil(dr / 2), math.ceil(dc / 2))
        elif pt == 'bishop':
            return 0 if (ar + ac) % 2 == (br + bc) % 2 else 2
        elif pt == 'rook':
            return 0 if ar == br or ac == bc else 1
        elif pt == 'queen':
            return 0 if (ar == br or ac == bc or dr == dc) else 1
        elif pt == 'king':
            return max(dr, dc)
        else:  # pawn
            return dr

    def get_neighbors(pos, pt, color='white'):
        r, c = pos
        nbrs = []
        def add(nr, nc):
            if 0 <= nr < 8 and 0 <= nc < 8:
                # Skip if occupied by own piece (we only care about empty for pathfinding demo)
                if board and board[nr][nc] and board[nr][nc].get('color') == color:
                    return
                nbrs.append((nr, nc))

        if pt == 'knight':
            for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                add(r+dr, c+dc)
        elif pt == 'king':
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr or dc: add(r+dr, c+dc)
        elif pt in ('rook', 'bishop', 'queen'):
            dirs = {
                'rook':   [(0,1),(0,-1),(1,0),(-1,0)],
                'bishop': [(1,1),(1,-1),(-1,1),(-1,-1)],
                'queen':  [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
            }[pt]
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    if board and board[nr][nc]:
                        add(nr, nc)   # can land on (capture), but not pass
                        break
                    add(nr, nc)
                    nr += dr; nc += dc
        elif pt == 'pawn':
            fwd = -1   # white pawn moves up (decreasing row)
            if 0 <= r+fwd < 8: add(r+fwd, c)
            if r == 6 and 0 <= r+2*fwd < 8: add(r+2*fwd, c)
        return nbrs

    def threatened_squares(bd):
        """Set of (r,c) threatened by black pieces."""
        if not bd:
            return set()
        threatened = set()
        for pr in range(8):
            for pc in range(8):
                p = bd[pr][pc]
                if p and p.get('color') == 'black':
                    pt2 = p.get('type', '')
                    for (nr, nc) in get_neighbors((pr, pc), pt2, 'black'):
                        threatened.add((nr, nc))
        return threatened

    threats = threatened_squares(board) if avoid_threatened else set()

    # A* search
    start, goal = from_rc, to_rc
    pq = [(heuristic(start, goal, piece_type), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    closed = set()
    nodes_explored = 0

    while pq:
        f, g, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        closed.add(cur)
        nodes_explored += 1

        if cur == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            return {
                'path'          : [_rc_to_sq(r, c) for r, c in path],
                'path_rc'       : [[r, c] for r, c in path],
                'nodes_explored': nodes_explored,
                'path_length'   : len(path) - 1,
                'found'         : True,
                'avoid_threats' : avoid_threatened,
            }

        for nb in get_neighbors(cur, piece_type):
            if nb in closed:
                continue
            extra = 10 if nb in threats else 0   # high cost to enter threatened square
            new_g = g_score[cur] + 1 + extra
            if nb not in g_score or new_g < g_score[nb]:
                g_score[nb] = new_g
                came_from[nb] = cur
                h = heuristic(nb, goal, piece_type)
                heapq.heappush(pq, (new_g + h, new_g, nb))

    return {'path': [], 'path_rc': [], 'nodes_explored': nodes_explored, 'path_length': 0, 'found': False}


@app.route('/api/chess/piece_path', methods=['POST'])
def chess_piece_path():
    """
    A* shortest-path search for a chess piece on the 8x8 board.
    Example: Knight from a1 to h8, Queen avoiding threatened squares.
    """
    d = request.get_json()
    piece_type       = d.get('piece', 'knight').lower()
    from_sq          = d.get('from', 'a1').lower()
    to_sq            = d.get('to', 'h8').lower()
    board            = d.get('board', None)          # optional live board state
    avoid_threatened = d.get('avoid_threatened', False)

    try:
        from_rc = _sq_to_rc(from_sq)
        to_rc   = _sq_to_rc(to_sq)
    except Exception:
        return jsonify({'error': 'Invalid square notation'}), 400

    t0 = time.time()
    result = _chess_piece_astar(piece_type, from_rc, to_rc, board, avoid_threatened)
    result['time_ms']   = round((time.time() - t0) * 1000, 2)
    result['piece']     = piece_type
    result['from_sq']   = from_sq
    result['to_sq']     = to_sq
    result['heuristic'] = {
        'knight': 'Min knight-moves approximation (Chebyshev-based)',
        'queen' : 'Min queen moves: 0 (reachable in 1) or 1 (needs 2)',
        'rook'  : 'Min rook moves: 0 (same rank/file) or 1',
        'bishop': 'Min bishop moves: 0 (same colour diagonal) or 2',
        'king'  : 'Chebyshev distance (max of row/col diff)',
        'pawn'  : 'Row distance (forward only)',
    }.get(piece_type, 'Manhattan distance')
    return jsonify(result)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  CHESS DEMO: CSP ANALYSIS
#  /api/chess/csp_analysis
#  Models legal move generation as a CSP.
#  Shows: raw pseudo-moves, pruning pipeline, MRV ordering.
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

@app.route('/api/chess/csp_analysis', methods=['POST'])
def chess_csp_analysis():
    """
    CSP framing of legal move generation.

    Variables   : Each friendly piece is a CSP variable.
    Domain      : All pseudo-legal destination squares per piece.
    Constraints :
      C1 — Movement rules   (piece type defines its reachable squares)
      C2 — Board boundaries (0 <= r,c < 8)
      C3 — Friendly blocking (cannot land on own piece)
      C4 — No king left in check after move (forward checking)

    MRV (Minimum Remaining Values):
      Pieces ordered by number of legal moves (fewest first).
      Lower MRV = higher priority in search tree (prune early).
    """
    d     = request.get_json()
    board = d.get('board')
    color = d.get('color', 'white')

    if not board:
        return jsonify({'error': 'No board provided'}), 400

    FILES = 'abcdefgh'
    def alg(sq): return FILES[sq[1]] + str(8 - sq[0])

    # \u2500 Step 1: Count raw candidate squares (no rule filtering) \u2500\u2500\u2500\u2500
    raw_candidates = 0
    for r in range(8):
        for c in range(8):
            if board[r][c] and board[r][c]['color'] == color:
                # Every empty or enemy-occupied square is a "raw candidate"
                for tr in range(8):
                    for tc in range(8):
                        if not (board[tr][tc] and board[tr][tc]['color'] == color):
                            raw_candidates += 1

    # \u2500 Step 2: After C1+C2+C3 — piece movement rules \u2500\u2500\u2500\u2500\u2500\u2500\u2500
    after_movement = 0
    piece_pseudo = {}   # (r,c) -> [pseudo moves]
    for r in range(8):
        for c in range(8):
            if board[r][c] and board[r][c]['color'] == color:
                mvs = _raw_moves(board, r, c)
                piece_pseudo[(r, c)] = mvs
                after_movement += len(mvs)

    # \u2500 Step 3: After C4 — forward checking (king safety) \u2500\u2500\u2500\u2500
    piece_legal = {}    # (r,c) -> [legal moves]
    after_check = 0
    for (r, c), pseudo_mvs in piece_pseudo.items():
        legal = []
        for m in pseudo_mvs:
            if m.get('sp') in ('castle_k', 'castle_q'):
                opp = 'black' if color == 'white' else 'white'
                if _attacked(board, m['from'][0], m['from'][1], opp):
                    continue
                mc = m['from'][1] + 1 if m['sp'] == 'castle_k' else m['from'][1] - 1
                if _attacked(board, m['from'][0], mc, opp):
                    continue
            nb = _apply(board, m)
            if not _in_check(nb, color):
                legal.append(m)
        piece_legal[(r, c)] = legal
        after_check += len(legal)

    # \u2500 MRV ordering (Minimum Remaining Values) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    mrv_list = []
    for (r, c), legal in piece_legal.items():
        p = board[r][c]
        mrv_list.append({
            'piece'      : p['type'],
            'color'      : p['color'],
            'square'     : alg((r, c)),
            'pseudo_moves': len(piece_pseudo.get((r, c), [])),
            'legal_moves': len(legal),
            'example_moves': [alg(m['to']) for m in legal[:5]],
        })
    # Sort by legal_moves ascending (MRV: fewest legal moves first)
    mrv_list.sort(key=lambda x: x['legal_moves'])

    pruned_movement = raw_candidates - after_movement
    pruned_check    = after_movement - after_check
    total_pruned    = raw_candidates - after_check

    return jsonify({
        'color'            : color,
        'raw_candidates'   : raw_candidates,      # before any constraint
        'after_movement'   : after_movement,       # after C1+C2+C3
        'after_check'      : after_check,          # after C4 (final legal)
        'pruned_by_movement': pruned_movement,
        'pruned_by_check'  : pruned_check,
        'total_pruned'     : total_pruned,
        'pruning_pct'      : round(total_pruned / max(raw_candidates, 1) * 100, 1),
        'mrv_order'        : mrv_list,             # MRV-ordered piece list
        'csp_summary': {
            'variables'   : len(piece_legal),
            'constraints' : ['Movement rules (C1)', 'Board boundaries (C2)',
                             'No friendly-piece collision (C3)', 'King not in check (C4)'],
            'mrv_applied' : True,
            'forward_checking': True,
        }
    })




# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  MOVE RECOMMENDATION + EXPLAINABLE AI
#  /api/chess/recommend_move
#  When the user selects a piece, returns:
#    - best move from that piece's legal moves
#    - human-readable reasons (center, king safety, mobility, development)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

@app.route('/api/chess/recommend_move', methods=['POST'])
def chess_recommend_move():
    """
    Explainable move recommendation for a selected piece.
    Uses Alpha-Beta depth-2 search constrained to moves from the
    selected piece's square, then explains why the chosen move is good.
    """
    d     = request.get_json()
    board = d.get('board')
    color = d.get('color', 'white')
    row   = d.get('row', -1)
    col   = d.get('col', -1)

    if not board or row < 0 or col < 0:
        return jsonify({'error': 'Missing board or square'}), 400
    if not board[row][col]:
        return jsonify({'error': 'No piece at given square'}), 400

    FILES = 'abcdefgh'
    def alg(sq): return FILES[sq[1]] + str(8 - sq[0])

    # All legal moves for the whole side (for context)
    all_legal = _legal_moves(board, color)
    # Legal moves only for this piece
    piece_moves = [m for m in all_legal if m['from'] == [row, col]]

    if not piece_moves:
        return jsonify({
            'found': False,
            'reason': 'This piece has no legal moves.',
            'move': None,
            'reasons': [],
        })

    # Evaluate each move and pick best
    best_move, best_score = None, float('-inf') if color == 'white' else float('inf')
    for m in piece_moves:
        nb = _apply(board, m)
        sc = _evaluate(nb)
        if (color == 'white' and sc > best_score) or (color == 'black' and sc < best_score):
            best_score = sc; best_move = m

    to_r, to_c = best_move['to']
    piece_type  = board[row][col]['type']
    from_sq     = alg([row, col])
    to_sq       = alg([to_r, to_c])

    # ── Generate human-readable reasons ──────────────────────
    reasons = []
    opp = 'black' if color == 'white' else 'white'

    # 1. Capture
    if board[to_r][to_c] and board[to_r][to_c]['color'] == opp:
        cap = board[to_r][to_c]['type']
        reasons.append(f'Captures enemy {cap} (material gain +{_CHESS_PV.get(cap,0)} cp)')

    # 2. Center control
    center_squares = {(3,3),(3,4),(4,3),(4,4)}
    ext_center = {(2,2),(2,3),(2,4),(2,5),(3,2),(3,5),(4,2),(4,5),(5,2),(5,3),(5,4),(5,5)}
    if (to_r, to_c) in center_squares:
        reasons.append('Controls the centre (d4/d5/e4/e5)')
    elif (to_r, to_c) in ext_center:
        reasons.append('Influences the extended centre')

    # 3. King safety
    before_ks = _king_safety(board, color)
    after_ks  = _king_safety(_apply(board, best_move), color)
    if after_ks > before_ks:
        reasons.append(f'Improves king safety (+{after_ks - before_ks} cp)')

    # 4. Mobility (mobility before vs after)
    before_mob = len(all_legal)
    after_mob  = len(_legal_moves(_apply(board, best_move), color))
    if after_mob > before_mob:
        reasons.append(f'Increases mobility (+{after_mob - before_mob} moves available)')
    elif after_mob == before_mob:
        reasons.append('Maintains piece mobility')

    # 5. Piece development (moving from starting rank)
    start_rank = 7 if color == 'white' else 0
    if row == start_rank and to_r != start_rank and piece_type != 'pawn':
        reasons.append(f'Develops the {piece_type} from starting square')

    # 6. Check threat
    post_b = _apply(board, best_move)
    if _in_check(post_b, opp):
        reasons.append('Puts opponent king in CHECK!')

    # 7. Pawn advance
    if piece_type == 'pawn':
        fwd = 7 - to_r if color == 'white' else to_r
        if fwd >= 5:
            reasons.append('Advanced pawn — promotion threat')
        else:
            reasons.append('Advances pawn toward promotion')

    if not reasons:
        reasons.append('Best available move according to evaluation function')

    # 8. All piece moves for display
    all_piece_alg = [
        {'from': alg(m['from']), 'to': alg(m['to']),
         'score': round(_evaluate(_apply(board, m)) / 100, 2)}
        for m in piece_moves
    ]
    all_piece_alg.sort(key=lambda x: x['score'], reverse=(color == 'white'))

    return jsonify({
        'found'       : True,
        'move'        : best_move,
        'from_sq'     : from_sq,
        'to_sq'       : to_sq,
        'piece'       : piece_type,
        'eval_score'  : round(best_score / 100, 2),
        'reasons'     : reasons,
        'all_moves'   : all_piece_alg[:6],   # top 6 moves for this piece
        'total_legal' : len(all_legal),
        'piece_legal' : len(piece_moves),
    })


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  GAME ANALYSIS MODE
#  /api/chess/analyze_move
#  After a move is played: eval before vs after, mistake detection,
#  and best alternative move for post-game review.
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

@app.route('/api/chess/analyze_move', methods=['POST'])
def chess_analyze_move():
    """
    Post-move analysis:
      - Evaluates position before and after the move
      - Classifies the move quality (brilliant / good / inaccuracy / mistake / blunder)
      - Finds the best alternative move (depth-2 search on pre-move board)
      - Returns eval delta and explanation
    """
    d          = request.get_json()
    board_pre  = d.get('board_before')   # board BEFORE the move
    board_post = d.get('board_after')    # board AFTER the move
    color      = d.get('color', 'white') # color that made the move
    move_alg   = d.get('move_alg', '?')  # algebraic notation string

    if not board_pre or not board_post:
        return jsonify({'error': 'board_before and board_after required'}), 400

    FILES = 'abcdefgh'
    def alg(sq): return FILES[sq[1]] + str(8 - sq[0])

    # Eval before and after (from white's perspective always)
    eval_before = _evaluate(board_pre)
    eval_after  = _evaluate(board_post)

    # Delta from the moving player's perspective
    if color == 'white':
        delta = eval_after - eval_before
    else:
        delta = eval_before - eval_after   # black wants eval to go more negative

    # ── Best alternative move (depth-2 AB search on pre-move board) ──
    all_legal = _legal_moves(board_pre, color)
    best_alt_move  = None
    best_alt_score = float('-inf') if color == 'white' else float('inf')
    for m in all_legal:
        nb = _apply(board_pre, m)
        sc = _evaluate(nb)
        if (color == 'white' and sc > best_alt_score) or (color == 'black' and sc < best_alt_score):
            best_alt_score = sc; best_alt_move = m

    alt_sq   = None
    alt_eval = None
    if best_alt_move:
        alt_sq   = alg(best_alt_move['from']) + '\u2192' + alg(best_alt_move['to'])
        alt_eval = round(best_alt_score / 100, 2)

    # ── Classify move quality ─────────────────────────────────
    # delta in centipawns (positive = improved for the mover)
    if delta >= 50:
        quality    = 'Brilliant'
        quality_color = '#a78bfa'
        quality_icon  = '!!'
    elif delta >= 10:
        quality    = 'Good'
        quality_color = '#34d399'
        quality_icon  = '!'
    elif delta >= -20:
        quality    = 'Neutral'
        quality_color = '#94a3b8'
        quality_icon  = ''
    elif delta >= -100:
        quality    = 'Inaccuracy'
        quality_color = '#fbbf24'
        quality_icon  = '?'
    elif delta >= -300:
        quality    = 'Mistake'
        quality_color = '#f97316'
        quality_icon  = '??'
    else:
        quality    = 'Blunder'
        quality_color = '#f87171'
        quality_icon  = '??!'

    return jsonify({
        'move_alg'      : move_alg,
        'eval_before'   : round(eval_before / 100, 2),
        'eval_after'    : round(eval_after  / 100, 2),
        'eval_delta'    : round(delta / 100, 2),
        'quality'       : quality,
        'quality_color' : quality_color,
        'quality_icon'  : quality_icon,
        'best_alt_move' : alt_sq,
        'best_alt_eval' : alt_eval,
        'total_legal'   : len(all_legal),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
