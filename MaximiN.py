from collections import defaultdict
import json
import math
import random
import time
from typing import List, Tuple, Dict, Optional

from ortools.sat.python import cp_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---------- Metrics ----------
def maximin_value(order: List[str], edges: List[Tuple[str, str]]) -> int:
    """Минимальное расстояние между парами в order для рёбер edges."""
    pos = {name: i + 1 for i, name in enumerate(order)}
    return min(abs(pos[a] - pos[b]) for a, b in edges)


def minimax_value(order: List[str], edges: List[Tuple[str, str]]) -> int:
    """Максимальное расстояние между парами в order для рёбер edges."""
    pos = {name: i + 1 for i, name in enumerate(order)}
    return max(abs(pos[a] - pos[b]) for a, b in edges)


# ---------- Local searches: 2-opt / 3-opt ----------
def local_search_2opt(order: List[str], edges: List[Tuple[str, str]],
                      func, maximize: bool = True,
                      skip_prob: float = 0.0, max_no_improve: int = 200) -> List[str]:
    n = len(order)
    best = order[:]
    best_val = func(best, edges)
    no_improve = 0

    while True:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        best_local_swap = None
        best_local_val = best_val

        for idx_i in range(n - 1):
            i = indices[idx_i]
            j_candidates = indices[idx_i + 1 :]
            random.shuffle(j_candidates)
            for j in j_candidates:
                if skip_prob > 0 and random.random() < skip_prob:
                    continue
                new_order = best[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_val = func(new_order, edges)
                if (maximize and new_val > best_local_val) or (not maximize and new_val < best_local_val):
                    best_local_val = new_val
                    best_local_swap = (i, j, new_order)
                    # early exit if clearly better
                    if (maximize and new_val >= best_val + 2) or (not maximize and new_val <= best_val - 2):
                        break
            if best_local_swap and ((maximize and best_local_val >= best_val + 2) or (not maximize and best_local_val <= best_val - 2)):
                break

        if best_local_swap:
            _, _, best = best_local_swap
            best_val = best_local_val
            improved = True
            no_improve = 0
        else:
            no_improve += 1

        if not improved:
            if no_improve >= max_no_improve:
                # focused shake: shuffle a small problematic segment
                pos = {name: i for i, name in enumerate(best)}
                dist_list = [(abs(pos[a] - pos[b]), pos[a], pos[b]) for a, b in edges]
                dist_list.sort()
                bad = set()
                for _, i, j in dist_list[: max(1, len(dist_list) // 6)]:
                    bad.add(i); bad.add(j)
                bad = sorted(bad)
                if len(bad) >= 2:
                    a, b = min(bad), max(bad)
                    seg_len = max(2, min(len(best) // 4, b - a + 1))
                    seg = best[a : a + seg_len]
                    random.shuffle(seg)
                    best[a : a + seg_len] = seg
                else:
                    for _ in range(max(1, len(best) // 10)):
                        i, j = random.sample(range(len(best)), 2)
                        best[i], best[j] = best[j], best[i]
                no_improve = 0
                continue
            else:
                break

    return best


def local_search_3opt(order: List[str], edges: List[Tuple[str, str]], func, maximize: bool = True,
                      max_checks: int = 2000) -> List[str]:
    """Sampled 3-opt — пробуем несколько перестановок троек и принимаем первое улучшение."""
    n = len(order)
    best = order[:]
    best_val = func(best, edges)
    checks = 0

    while checks < max_checks:
        i, j, k = sorted(random.sample(range(n), 3))
        variants = []
        v = best[:]
        v[i], v[j], v[k] = best[j], best[k], best[i]
        variants.append(v)
        v = best[:]
        v[i], v[j], v[k] = best[k], best[i], best[j]
        variants.append(v)
        v = best[:]
        v[j], v[k] = best[k], best[j]
        variants.append(v)

        for new_order in variants:
            checks += 1
            new_val = func(new_order, edges)
            if (maximize and new_val > best_val) or (not maximize and new_val < best_val):
                best = new_order
                best_val = new_val
                break
        if checks >= max_checks:
            break

    return best

# ---------- Adaptive utilities ----------
def adaptive_shuffle_indices(order: List[str], edges: List[Tuple[str, str]], worst_fraction: float = 0.2) -> List[int]:
    """Возвращает индексы вершин, которые стоит перемешать (на основе "дефицитов")."""
    pos = {name: i + 1 for i, name in enumerate(order)}
    deficits: Dict[str, int] = defaultdict(int)
    for a, b in edges:
        d = abs(pos[a] - pos[b])
        if d < 12:
            deficits[a] += (12 - d)
            deficits[b] += (12 - d)
    if not deficits:
        n = len(order)
        rcount = max(1, int(n * 0.05))
        return random.sample(list(range(n)), rcount)

    sorted_vertices = sorted(deficits.items(), key=lambda x: -x[1])
    m = max(1, int(len(edges) * worst_fraction))
    selected = [v for v, _ in sorted_vertices[:m]]
    neighbors = set(selected)
    for v in selected:
        for a, b in edges:
            if a == v:
                neighbors.add(b)
            elif b == v:
                neighbors.add(a)

    idx = [order.index(v) for v in neighbors if v in order]
    n = len(order)
    min_needed = max(1, int(n * 0.05))
    if len(idx) < min_needed:
        extras = [i for i in range(n) if i not in idx]
        random.shuffle(extras)
        idx.extend(extras[: min_needed - len(idx)])
    random.shuffle(idx)
    return idx

def accept_prob(delta: float, temperature: float) -> float:
    """Вероятность принятия хуже-решения (для SA)."""
    if delta >= 0:
        return 1.0
    try:
        return math.exp(delta / max(1e-12, temperature))
    except OverflowError:
        return 0.0

# ---------- Two-phase memetic search (для maximin) ----------
def two_phase_search(order: List[str], edges: List[Tuple[str, str]], func, maximize: bool = True) -> List[str]:
    """Простая меметическая (GA + локальный поиск) схема для максимизации func."""
    pop_size = 14
    generations = 40
    mutation_rate = 0.28
    elite_frac = 0.25
    n = len(order)

    # population
    population = [order[:]]
    degrees = defaultdict(int)
    for a, b in edges:
        degrees[a] += 1; degrees[b] += 1
    deg_order = sorted(order, key=lambda x: degrees[x], reverse=True)
    population.append(deg_order[:])
    population.append(sorted(order, key=lambda x: -degrees[x])[:])
    while len(population) < pop_size:
        p = order[:]
        random.shuffle(p)
        population.append(p)

    for i in range(len(population)):
        population[i] = local_search_2opt(population[i], edges, func, maximize, skip_prob=0.12)

    def fitness(p: List[str]) -> int:
        return func(p, edges)

    for gen in range(generations):
        scored = sorted([(fitness(p), p) for p in population], key=lambda x: x[0], reverse=maximize)
        elites_count = max(2, int(len(population) * elite_frac))
        elites = [p for _, p in scored[:elites_count]]
        new_pop = elites[:]

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            cut = random.randint(1, n - 2)
            child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
            if random.random() < mutation_rate:
                a, b = random.sample(range(n), 2)
                child[a], child[b] = child[b], child[a]
            if random.random() < 0.12:
                idx = random.sample(range(n), max(2, n // 12))
                vals = [child[i] for i in idx]
                random.shuffle(vals)
                for i, v in zip(idx, vals):
                    child[i] = v
            child = local_search_2opt(child, edges, func, maximize, skip_prob=0.15)
            new_pop.append(child)

        population = new_pop
        if gen % 8 == 0:
            for _ in range(max(1, pop_size // 8)):
                r = order[:]
                random.shuffle(r)
                r = local_search_2opt(r, edges, func, maximize, skip_prob=0.25)
                population[-1] = r

    best = max(population, key=fitness) if maximize else min(population, key=fitness)
    return best

# ---------- Iterated 2-opt with SA (для minimax) ----------
def iterated_2opt(order: List[str], edges: List[Tuple[str, str]], func, maximize: bool = True,
                  iterations: int = 150, shuffle_fraction: float = 0.35, skip_prob: float = 0.02) -> List[str]:
    best = order[:]
    best_val = func(best, edges)
    n = len(order)
    temperature = max(1.0, best_val / 2)

    current, current_val = best[:], best_val
    stagnation = 0

    for _ in range(iterations):
        new_order = current[:]
        if random.random() < 0.4:
            a, b = sorted(random.sample(range(n), 2))
            new_order[a:b] = reversed(new_order[a:b])
        else:
            idx = random.sample(range(n), max(2, int(n * shuffle_fraction)))
            vals = [new_order[i] for i in idx]
            random.shuffle(vals)
            for i, v in zip(idx, vals):
                new_order[i] = v

        new_order = local_search_2opt(new_order, edges, func, maximize=maximize, skip_prob=skip_prob)
        new_order = local_search_3opt(new_order, edges, func, maximize=maximize)

        new_val = func(new_order, edges)
        delta = (new_val - current_val) if maximize else (current_val - new_val)

        if delta > 0 or random.random() < math.exp(delta / max(1e-9, temperature)):
            current, current_val = new_order[:], new_val
            stagnation = 0
            if (maximize and current_val > best_val) or (not maximize and current_val < best_val):
                best, best_val = current[:], current_val
        else:
            stagnation += 1

        if stagnation > max(5, iterations // 20):
            a, b = sorted(random.sample(range(n), 2))
            current[a:b] = reversed(current[a:b])
            current_val = func(current, edges)
            stagnation = 0

        temperature *= 0.97

    return best

# ---------- OR-Tools exact checks ----------
def feasible_k(vertices: List[str], edges: List[Tuple[str, str]], k: int, time_limit: float = 60.0) -> Optional[List[str]]:
    """Проверяет достижимость min-edge >= k (maximin) с помощью CP-SAT."""
    model = cp_model.CpModel()
    n = len(vertices)
    order_vars = {v: model.NewIntVar(1, n, f"ord_{v}") for v in vertices}
    model.AddAllDifferent(order_vars.values())

    for u, v in edges:
        diff = model.NewIntVar(-n, n, f"diff_{u}_{v}")
        model.Add(diff == order_vars[u] - order_vars[v])
        absdiff = model.NewIntVar(0, n, f"abs_{u}_{v}")
        model.AddAbsEquality(absdiff, diff)
        model.Add(absdiff >= k)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        order = sorted(vertices, key=lambda v: solver.Value(order_vars[v]))
        return order
    return None

def feasible_k_minimax(vertices: List[str], edges: List[Tuple[str, str]], k: int, time_limit: float = 60.0) -> Optional[List[str]]:
    """Проверяет достижимость max-edge <= k (minimax) с помощью CP-SAT."""
    model = cp_model.CpModel()
    n = len(vertices)
    order_vars = {v: model.NewIntVar(1, n, f"ord_{v}") for v in vertices}
    model.AddAllDifferent(order_vars.values())

    for u, v in edges:
        diff = model.NewIntVar(-n, n, f"diff_{u}_{v}")
        model.Add(diff == order_vars[u] - order_vars[v])
        absdiff = model.NewIntVar(0, n, f"abs_{u}_{v}")
        model.AddAbsEquality(absdiff, diff)
        model.Add(absdiff <= k)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        order = sorted(vertices, key=lambda v: solver.Value(order_vars[v]))
        return order
    return None

def maximin_by_csp(vertices: List[str], edges: List[Tuple[str, str]], time_per_k: float = 6.0,
                   overall_time_limit: float = 60.0) -> Tuple[Optional[List[str]], Optional[int]]:
    start = time.time()
    n = len(vertices)
    lo, hi = 1, n - 1
    for k_try in range(hi, lo - 1, -1):
        elapsed = time.time() - start
        if elapsed > overall_time_limit:
            break
        extra = (k_try - lo) * 0.5
        tlimit = min(overall_time_limit - elapsed, max(1.0, time_per_k + extra))
        print(f"Проверка k = {k_try} с таймаутом {tlimit:.2f}s ...")
        order = feasible_k(vertices, edges, k_try, time_limit=tlimit)
        if order is not None:
            print(f"✅ CSP нашёл порядок с min-edge >= {k_try}")
            return order, k_try
    print("⚠️ CSP не нашёл решение в пределах лимита.")
    return None, None

def minimax_by_csp(vertices: List[str], edges: List[Tuple[str, str]], time_per_k: float = 6.0,
                   overall_time_limit: float = 60.0) -> Tuple[Optional[List[str]], Optional[int]]:
    start = time.time()
    n = len(vertices)
    lo, hi = 1, n - 1
    best_order = None
    best_k = hi

    print("=== Точный поиск minimax через OR-Tools ===")
    while lo <= hi and (time.time() - start) < overall_time_limit:
        mid = (lo + hi) // 2
        tlimit = min(time_per_k, overall_time_limit - (time.time() - start))
        print(f"Пробуем k = {mid} (таймаут {tlimit:.1f}s)...", end=" ")
        order = feasible_k_minimax(vertices, edges, mid, time_limit=tlimit)
        if order is not None:
            print("✅ FEASIBLE")
            best_order, best_k = order, mid
            hi = mid - 1
        else:
            print("❌ UNSAT")
            lo = mid + 1

    if best_order is not None:
        print(f"✅ Оптимальное значение minimax ≤ {best_k}")
    else:
        print("⚠️ Решение не найдено в лимите.")
    return best_order, best_k

# ---------- Initial orders ----------
def reverse_cuthill_mckee(vertices: List[str], edges_named: List[Tuple[str, str]]) -> List[str]:
    G = {v: set() for v in vertices}
    for a, b in edges_named:
        G[a].add(b); G[b].add(a)
    visited = set()
    order: List[str] = []

    def bfs(start: str) -> None:
        queue = [start]
        visited.add(start)
        while queue:
            u = queue.pop(0)
            order.append(u)
            neigh = sorted(list(G[u] - visited), key=lambda x: len(G[x]))
            for v in neigh:
                visited.add(v)
            queue.extend(neigh)

    remaining = set(vertices)
    while remaining:
        start = min(remaining, key=lambda x: len(G[x]))
        bfs(start)
        remaining -= set(order)
    return list(reversed(order))

def greedy_initial_order(vertices: List[str], edges_named: List[Tuple[str, str]]) -> List[str]:
    G = {v: set() for v in vertices}
    for a, b in edges_named:
        G[a].add(b); G[b].add(a)
    remaining = set(vertices)
    current = min(vertices, key=lambda x: len(G[x]))
    order = [current]
    remaining.remove(current)
    while remaining:
        next_v = max(remaining, key=lambda x: sum(1 for y in order if y in G[x]))
        order.append(next_v)
        remaining.remove(next_v)
    return order


def initial_orders(vertices: List[str], edges_named: List[Tuple[str, str]]) -> List[List[str]]:
    names = list(vertices)
    degrees = {v: sum(1 for a, b in edges_named if a == v or b == v) for v in names}
    return [
        reverse_cuthill_mckee(names, edges_named),
        sorted(names, key=lambda x: degrees[x], reverse=True),
        random.sample(names, len(names)),
        greedy_initial_order(names, edges_named),
    ]

# ---------- Printing & Visualization ----------
def print_numbering_maximin(order: List[str], edges: List[Tuple[str, str]]) -> None:
    pos = {name: i + 1 for i, name in enumerate(order)}
    min_dist, max_dist = float("inf"), -1
    min_edge = max_edge = None
    print("Подсчёт maximin:")
    for a, b in edges:
        d = abs(pos[a] - pos[b])
        print(f"Ребро ({pos[a]}, {pos[b]}) = {d}")
        if d < min_dist: min_dist, min_edge = d, (pos[a], pos[b])
        if d > max_dist: max_dist, max_edge = d, (pos[a], pos[b])
    print(f"Минимальное ребро (maximin) = {min_edge} с длиной {min_dist}")
    print(f"Максимальное ребро = {max_edge} с длиной {max_dist}\n")

def print_numbering_minimax(order: List[str], edges: List[Tuple[str, str]]) -> None:
    pos = {name: i + 1 for i, name in enumerate(order)}
    max_dist, min_dist = -1, float("inf")
    max_edge = min_edge = None
    print("Подсчёт minimax:")
    for a, b in edges:
        d = abs(pos[a] - pos[b])
        print(f"Ребро ({pos[a]}, {pos[b]}) = {d}")
        if d > max_dist: max_dist, max_edge = d, (pos[a], pos[b])
        if d < min_dist: min_dist, min_edge = d, (pos[a], pos[b])
    print(f"Максимальное ребро (minimax) = {max_edge} с длиной {max_dist}")
    print(f"Минимальное ребро = {min_edge} с длиной {min_dist}\n")


def visualize_graph(vertices: List[Dict], edges: List[Dict], order: List[str], title: str,
                    filename: str, maximize: bool = True, draw_numbers: bool = True) -> None:
    """Сохраняет изображение графа с пронумерованными вершинами в порядке order."""

    # Сопоставление имени вершины с номером
    name_to_num = {name: i + 1 for i, name in enumerate(order)}
    node_color = "#ffa3a3" if maximize else "#a3d5ff"

    # Позиции вершин (с инвертированным y)
    positions = {}
    for v in vertices:
        name = v["name"]
        num = name_to_num[name]
        x, y = v.get("x", 0), v.get("y", 0)
        positions[num] = (x, -y)

    # Длины ребер
    edge_lengths = [
        (abs(name_to_num[vertices[e["vertex1"]]["name"]] - name_to_num[vertices[e["vertex2"]]["name"]]),
         (name_to_num[vertices[e["vertex1"]]["name"]], name_to_num[vertices[e["vertex2"]]["name"]]))
        for e in edges
    ]
    min_len, min_edge = min(edge_lengths, key=lambda x: x[0])
    max_len, max_edge = max(edge_lengths, key=lambda x: x[0])

    # Подпись
    if maximize:
        label_text = f"Минимальное ребро (maximin) = {min_edge} с длиной {min_len}\nМаксимальное ребро = {max_edge} с длиной {max_len}"
    else:
        label_text = f"Максимальное ребро (minimax) = {max_edge} с длиной {max_len}\nМинимальное ребро = {min_edge} с длиной {min_len}"

    fig, ax = plt.subplots(figsize=(12, 8))

    # Рисуем ребра
    for e in edges:
        v1 = vertices[e["vertex1"]]["name"]
        v2 = vertices[e["vertex2"]]["name"]
        u, v = name_to_num[v1], name_to_num[v2]
        x1, y1 = positions[u];
        x2, y2 = positions[v]
        control = e.get("controlStep", 0)
        color = e.get("color", "#000000")
        width = e.get("lineWidth", 2)
        if control == 0:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)
        else:
            rad = control / 300.0
            patch = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle="-", color=color, linewidth=width, zorder=2)
            ax.add_patch(patch)

    # Рисуем вершины
    for num, (x, y) in positions.items():
        ax.scatter(x, y, s=1000, color=node_color, edgecolors="black", zorder=3)
        if draw_numbers:
            ax.text(x, y, str(num), fontsize=12, weight="bold", ha="center", va="center", zorder=4)

    plt.title(f"{title}\n{label_text}", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Автоматическое масштабирование с отступами
    xs = [x for x, y in positions.values()]
    ys = [y for x, y in positions.values()]
    x_margin = (max(xs) - min(xs)) * 0.15 + 20  # 15% + 20 пикселей
    y_margin = (max(ys) - min(ys)) * 0.15 + 20
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

    fig.tight_layout(pad=0)
    plt.savefig(filename, bbox_inches="tight", dpi=300, pad_inches=0.2)
    plt.close()
    print(f"✅ Граф сохранён в {filename}")
# ---------- Main ----------
import json
import random
import time

def print_edges(order, edges_named):
    """Выводит список рёбер с разницей номеров между вершинами"""
    name_to_num = {name: i + 1 for i, name in enumerate(order)}
    print("\n--- Рёбра (v1 (num) - v2 (num) = dist) ---")
    total = 0
    for v1, v2 in edges_named:
        if v1 not in name_to_num or v2 not in name_to_num:
            continue
        n1 = name_to_num[v1]
        n2 = name_to_num[v2]
        dist = abs(n1 - n2)
        total += dist
        print(f"{v1} ({n1}) - {v2} ({n2}) = {dist}")
    print(f"Итого (сумма расстояний по рёбрам): {total}\n")

import json
import random
import time

def print_edges(order, edges_named):
    """Выводит список рёбер с разницей номеров между вершинами"""
    name_to_num = {name: i + 1 for i, name in enumerate(order)}
    print("\n--- Рёбра (v1 (num) - v2 (num) = dist) ---")
    total = 0
    for v1, v2 in edges_named:
        if v1 not in name_to_num or v2 not in name_to_num:
            continue
        n1 = name_to_num[v1]
        n2 = name_to_num[v2]
        dist = abs(n1 - n2)
        total += dist
        print(f"{v1} ({n1}) - {v2} ({n2}) = {dist}")
    print(f"Итого (сумма расстояний по рёбрам): {total}\n")

def main(file_path, choice="2") -> None:
    random.seed(time.time())
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vertices = data["vertices"]
    edges_raw = data["edges"]
    edges_named = [(vertices[e["vertex1"]]["name"], vertices[e["vertex2"]]["name"]) for e in edges_raw]
    vertex_names = [v["name"] for v in vertices]


    # === Режим поиска (основной алгоритм) ===
    print("=== Ищем maximin ===")
    order_csp, k_csp = maximin_by_csp(vertex_names, edges_named, time_per_k=6.0, overall_time_limit=120.0)
    if order_csp is not None:
        best_maximin = order_csp
    else:
        print("CSP не нашёл решение — используем эвристику two_phase_search.")
        starts = initial_orders(vertex_names, edges_named)
        best_val = float("-inf")
        best_ord = None
        for st in starts:
            cand = two_phase_search(st, edges_named, maximin_value, maximize=True)
            v = maximin_value(cand, edges_named)
            if v > best_val:
                best_val = v
                best_ord = cand
        best_maximin = best_ord

    print_numbering_maximin(best_maximin, edges_named)
    visualize_graph(vertices, edges_raw, best_maximin, "Maximin", "graph_maximin.png", maximize=True)

    print("=== Ищем minimax ===")
    order_csp_min, k_csp_min = minimax_by_csp(vertex_names, edges_named, time_per_k=8.0, overall_time_limit=120.0)
    if order_csp_min is not None:
        best_minimax = order_csp_min
        print_numbering_minimax(best_minimax, edges_named)
    else:
        print("CSP не нашёл решение — используем эвристику iterated_2opt.")
        starts = initial_orders(vertex_names, edges_named)
        best_minimax = None
        best_val_minimax = float("inf")
        for st in starts:
            cand = iterated_2opt(st, edges_named, minimax_value, maximize=False,
                                 iterations=180, shuffle_fraction=0.35, skip_prob=0.02)
            val = minimax_value(cand, edges_named)
            if val < best_val_minimax:
                best_minimax = cand
                best_val_minimax = val
        print_numbering_minimax(best_minimax, edges_named)

    visualize_graph(vertices, edges_raw, best_minimax, "Minimax", "graph_minimax.png", maximize=False)


if __name__ == "__main__":
    main("graph33.graph")
