from flask import Flask, render_template, request, jsonify, session
import heapq, time, math, random
from collections import deque

app = Flask(__name__)
app.secret_key = 'chess_ai_pathfinding_2024'

# ════════════════════════════════════════════════════════════
#  CHESS PIECE MOVEMENT RULES
# ════════════════════════════════════════════════════════════

def get_chess_neighbors(pos, grid, rows, cols, piece='king'):
    r, c = pos
    neighbors = []

    if piece == 'king':
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
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
            while 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 1: break
                neighbors.append((nr, nc))
                nr += dr; nc += dc

    elif piece == 'knight':
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                neighbors.append((nr, nc))

    elif piece == 'pawn':
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
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

def bfs(grid, start, goal, rows, cols, piece='king'):
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
        for n in get_chess_neighbors(cur, grid, rows, cols, piece):
            if n not in came_from:
                came_from[n] = cur
                queue.append(n)
    return fail('BFS', explored, t0, 'BFS: No path found. All reachable nodes explored.')

# ════════════════════════════════════════════════════════════
#  2. DFS — Depth-First Search
# ════════════════════════════════════════════════════════════

def dfs(grid, start, goal, rows, cols, piece='king'):
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
        for n in get_chess_neighbors(cur, grid, rows, cols, piece):
            if n not in came_from:
                came_from[n] = cur
                stack.append(n)
    return fail('DFS', explored, t0, 'DFS: No path found.')

# ════════════════════════════════════════════════════════════
#  3. UCS — Uniform Cost Search
# ════════════════════════════════════════════════════════════

def ucs(grid, start, goal, rows, cols, piece='king'):
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
        for n in get_chess_neighbors(cur, grid, rows, cols, piece):
            new_g = g + 1  # uniform cost = 1 per move
            if n not in cost_so_far or new_g < cost_so_far[n]:
                cost_so_far[n] = new_g
                came_from[n] = cur
                heapq.heappush(pq, (new_g, n))
    return fail('UCS', explored, t0, 'UCS: No path found.')

# ════════════════════════════════════════════════════════════
#  4. A* — A-Star Search
# ════════════════════════════════════════════════════════════

def astar(grid, start, goal, rows, cols, piece='king'):
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
        for n in get_chess_neighbors(cur, grid, rows, cols, piece):
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

def greedy(grid, start, goal, rows, cols, piece='king'):
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
        for n in get_chess_neighbors(cur, grid, rows, cols, piece):
            if n not in came_from:
                came_from[n] = cur
                heapq.heappush(pq, (manhattan(n,goal), n))
    return fail('Greedy', explored, t0, 'Greedy: No path found.')

# ════════════════════════════════════════════════════════════
#  6. CSP Backtracking
# ════════════════════════════════════════════════════════════

def csp_backtracking(grid, start, goal, rows, cols, piece='king'):
    t0 = time.time()
    explored = []
    result = []
    MAX_EXP = rows * cols * 2

    def bt(cur, path, visited):
        explored.append(cur)
        if len(explored) > MAX_EXP: return False
        if cur == goal:
            result.extend(path + [cur]); return True
        nbrs = sorted(get_chess_neighbors(cur, grid, rows, cols, piece),
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
        return ok('CSP Backtracking', result, explored, g, 0, t0,
                  f'CSP Backtracking with MRV heuristic. Constraints: no revisits + valid {piece} moves. '
                  f'Nodes explored: {len(explored)} | Cost: {g}')
    return fail('CSP Backtracking', explored, t0, 'CSP Backtracking: No path found.')

# ════════════════════════════════════════════════════════════
#  7. Minimum Conflicts (CSP Local Search)
# ════════════════════════════════════════════════════════════

def minimum_conflicts(grid, start, goal, rows, cols, piece='king'):
    t0 = time.time()
    explored = []
    MAX_STEPS = rows * cols * 4

    # Build initial greedy assignment
    cur, path, vis = start, [start], {start}
    for _ in range(rows * cols):
        if cur == goal: break
        nbrs = [n for n in get_chess_neighbors(cur, grid, rows, cols, piece) if n not in vis]
        if not nbrs: break
        nxt = min(nbrs, key=lambda n: manhattan(n, goal))
        path.append(nxt); vis.add(nxt); cur = nxt

    def n_conflicts(pos, i, path):
        c = 0
        if grid[pos[0]][pos[1]] == 1: c += 10
        if i > 0 and pos not in get_chess_neighbors(path[i-1], grid, rows, cols, piece): c += 5
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
            candidates = [n for n in get_chess_neighbors(last, grid, rows, cols, piece)
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
            candidates = [n for n in get_chess_neighbors(prev, grid, rows, cols, piece)
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
#  8. Minimax 2-Player (AI vs Adversary)
# ════════════════════════════════════════════════════════════

def minimax_2player(grid, start, goal, rows, cols, piece='king'):
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
        nbrs = [n for n in get_chess_neighbors(pos, grid, rows, cols, piece)
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
        fb = astar(grid, start, goal, rows, cols, piece)
        fb['algorithm'] = 'Minimax 2P'
        fb['description'] = (f'2-Player Minimax: AI (MAX) vs Adversary (MIN). '
                             f'Depth={DEPTH} | Nodes={len(explored)}. Fallback to A* path shown.')
        fb['explored'] = to_list(explored); fb['nodes_explored'] = len(explored)
        return fb

    path = best[0]; g = len(path)-1
    return ok('Minimax 2P', path, explored, g, 0, t0,
              f'2-Player Minimax: AI maximises score, Adversary minimises. '
              f'Depth={DEPTH} | Nodes evaluated={len(explored)} | Cost={g}')

# ════════════════════════════════════════════════════════════
#  9. Minimax 3-Player (Paranoid: AI vs 2 Adversaries)
# ════════════════════════════════════════════════════════════

def minimax_3player(grid, start, goal, rows, cols, piece='king'):
    t0 = time.time()
    explored = []
    DEPTH = min(6, rows)
    best = [None]

    def mm3(pos, depth, player, visited, path, alpha, beta):
        explored.append(pos)
        if pos == goal:
            if best[0] is None or len(path) < len(best[0]):
                best[0] = path[:]
            return 1000 - len(path)
        if depth == 0: return -manhattan(pos, goal)
        nbrs = [n for n in get_chess_neighbors(pos, grid, rows, cols, piece)
                if n not in visited]
        if not nbrs: return -manhattan(pos, goal) * 5
        next_p = (player % 3) + 1
        if player == 1:    # MAX — AI
            v = float('-inf')
            for n in nbrs[:4]:
                visited.add(n)
                v = max(v, mm3(n, depth-1, next_p, visited, path+[n], alpha, beta))
                visited.discard(n)
                alpha = max(alpha, v)
                if beta <= alpha: break
            return v
        else:              # MIN — Adversary P2 or P3 (paranoid)
            v = float('inf')
            for n in sorted(nbrs, key=lambda x: -manhattan(x, goal))[:3]:
                visited.add(n)
                v = min(v, mm3(n, depth-1, next_p, visited, path+[n], alpha, beta))
                visited.discard(n)
                beta = min(beta, v)
                if beta <= alpha: break
            return v

    mm3(start, DEPTH, 1, {start}, [start], float('-inf'), float('inf'))

    if not best[0] or best[0][-1] != goal:
        fb = astar(grid, start, goal, rows, cols, piece)
        fb['algorithm'] = 'Minimax 3P'
        fb['description'] = (f'3-Player Paranoid Minimax: AI vs 2 adversaries. '
                             f'Depth={DEPTH} | Nodes={len(explored)}. Fallback to A* path shown.')
        fb['explored'] = to_list(explored); fb['nodes_explored'] = len(explored)
        return fb

    path = best[0]; g = len(path)-1
    return ok('Minimax 3P', path, explored, g, 0, t0,
              f'3-Player Paranoid Minimax: P1(AI/MAX) vs P2+P3(Adversaries/MIN). '
              f'Paranoid assumption: both enemies gang up on AI. '
              f'Depth={DEPTH} | Nodes={len(explored)} | Cost={g}')

# ════════════════════════════════════════════════════════════
#  10. Alpha-Beta Pruning
# ════════════════════════════════════════════════════════════

def alpha_beta(grid, start, goal, rows, cols, piece='king'):
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
        nbrs = [n for n in get_chess_neighbors(pos, grid, rows, cols, piece)
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
        fb = astar(grid, start, goal, rows, cols, piece)
        fb['algorithm'] = 'Alpha-Beta'
        fb['description'] = (f'Alpha-Beta Pruning: Minimax + pruning. '
                             f'Branches pruned: {pruned[0]} | Nodes={len(explored)}. '
                             f'Fallback to A* path shown.')
        fb['explored'] = to_list(explored); fb['nodes_explored'] = len(explored)
        return fb

    path = best[0]; g = len(path)-1
    return ok('Alpha-Beta', path, explored, g, 0, t0,
              f'Alpha-Beta Pruning: Minimax + α-β cuts. Branches pruned: {pruned[0]} '
              f'(~{round(pruned[0]/(max(len(explored),1))*100)}% saved). '
              f'Depth={DEPTH} | Cost={g}')

# ════════════════════════════════════════════════════════════
#  11. Hill Climbing (Local Search)
# ════════════════════════════════════════════════════════════

def hill_climbing(grid, start, goal, rows, cols, piece='king'):
    t0 = time.time()
    explored = []
    RESTARTS = 5
    best_path = []

    for _ in range(RESTARTS):
        cur = start
        path = [cur]
        vis = {cur}
        while cur != goal:
            explored.append(cur)
            nbrs = [n for n in get_chess_neighbors(cur, grid, rows, cols, piece) if n not in vis]
            if not nbrs: break
            nxt = min(nbrs, key=lambda n: manhattan(n, goal))
            if manhattan(nxt, goal) >= manhattan(cur, goal):
                nxt = random.choice(nbrs)  # escape local optimum
            cur = nxt; vis.add(cur); path.append(cur)
        if cur == goal: best_path = path; break
        if len(path) > len(best_path): best_path = path

    found = bool(best_path) and best_path[-1] == goal
    g = len(best_path)-1 if found else 0
    if found:
        return ok('Hill Climbing', best_path, explored, g, 0, t0,
                  f'Steepest-Ascent Hill Climbing with {RESTARTS} random restarts to escape local optima. '
                  f'Cost: {g}')
    return fail('Hill Climbing', explored, t0,
                f'Hill Climbing stuck in local optimum after {RESTARTS} restarts.')

# ════════════════════════════════════════════════════════════
#  REGISTRY
# ════════════════════════════════════════════════════════════

ALGORITHMS = {
    'bfs':          bfs,
    'dfs':          dfs,
    'ucs':          ucs,
    'astar':        astar,
    'greedy':       greedy,
    'csp':          csp_backtracking,
    'minconflicts': minimum_conflicts,
    'minimax2':     minimax_2player,
    'minimax3':     minimax_3player,
    'alphabeta':    alpha_beta,
    'hillclimbing': hill_climbing,
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
    rows, cols = len(grid), len(grid[0]) if grid else 0

    if algo not in ALGORITHMS:
        return jsonify({'error': f'Unknown algorithm: {algo}'}), 400
    try:
        result = ALGORITHMS[algo](grid, start, goal, rows, cols, piece)
        session['result_data'] = result
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
