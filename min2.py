import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# === Подсчёт длины нумерации ===
def numbering_length(order, edges, verbose=True):
    name_to_number = {name: i + 1 for i, name in enumerate(order)}
    total = 0
    max_edge = (None, 0)
    min_edge = (None, float("inf"))
    for v1, v2 in edges:
        dist = abs(name_to_number[v1] - name_to_number[v2])
        total += dist
        if dist > max_edge[1]:
            max_edge = ((v1, v2), dist)
        if dist < min_edge[1]:
            min_edge = ((v1, v2), dist)
        if verbose:
            print(f"{name_to_number[v1]} - {name_to_number[v2]} = {dist}")
    if verbose:
        print(f"Итоговая длина нумерации: {total}")
        print(f"Минимальное ребро: {min_edge[0]} = {min_edge[1]}")
        print(f"Максимальное ребро: {max_edge[0]} = {max_edge[1]}")
    return total, name_to_number


# === Локальный поиск 2-opt ===
def local_search_2opt(order, edges, maximize=False):
    improved = True
    best_order = order[:]
    best_length, _ = numbering_length(best_order, edges, verbose=False)
    while improved:
        improved = False
        for i in range(len(best_order)):
            for j in range(i + 1, len(best_order)):
                new_order = best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_length, _ = numbering_length(new_order, edges, verbose=False)
                if (maximize and new_length > best_length) or (not maximize and new_length < best_length):
                    best_order = new_order
                    best_length = new_length
                    improved = True
    return best_length, best_order


# === Iterated 2-opt ===
def iterated_2opt(order, edges, maximize=False, iterations=50, shuffle_fraction=0.4):
    best_order = order[:]
    best_length, _ = numbering_length(best_order, edges, verbose=False)
    for _ in range(iterations):
        new_order = best_order[:]
        n_shuffle = max(1, int(len(order) * shuffle_fraction))
        idx = random.sample(range(len(order)), n_shuffle)
        vals = [new_order[i] for i in idx]
        random.shuffle(vals)
        for i, v in zip(idx, vals):
            new_order[i] = v
        new_length, new_order = local_search_2opt(new_order, edges, maximize=maximize)
        if (maximize and new_length > best_length) or (not maximize and new_length < best_length):
            best_order = new_order
            best_length = new_length
    return best_length, best_order


# === Визуализация ===
def visualize_graph(vertices, edges, name_to_number, length, filename, maximize=False):
    color = "#ffa3a3" if maximize else "#a3d5ff"
    title_text = "Max" if maximize else "Min"

    positions = {name_to_number[v["name"]]: (v["x"], -v["y"]) for v in vertices}
    fig, ax = plt.subplots(figsize=(14, 12))

    for num, (x, y) in positions.items():
        ax.scatter(x, y, s=1200, color=color, edgecolors="black", zorder=3)
        ax.text(x, y, str(num), fontsize=14, weight="bold", ha="center", va="center", zorder=4)

    for e in edges:
        v1 = vertices[e["vertex1"]]["name"]
        v2 = vertices[e["vertex2"]]["name"]
        u, v = name_to_number[v1], name_to_number[v2]
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        control_step = e.get("controlStep", 0)
        edge_color = e.get("color", "black")
        width = e.get("lineWidth", 2)
        if control_step == 0:
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=width, zorder=1)
        else:
            rad = control_step / 200.0
            patch = FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle="-", color=edge_color,
                                    linewidth=width, zorder=2)
            ax.add_patch(patch)

    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    padding_x = dx * 0.6 if dx > 0 else 200
    padding_y = dy * 0.6 if dy > 0 else 200

    ax.set_xlim(min(xs) - padding_x, max(xs) + padding_x)
    ax.set_ylim(-max(ys) - padding_y, -min(ys) + padding_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.title(f"{title_text}\nДлина нумерации = {length}", fontsize=14, fontweight="bold")
    plt.savefig(filename, bbox_inches="tight", dpi=300, pad_inches=1.0)
    plt.close()
    print(f"✅ Граф сохранён в {filename}")

def main(file_path, choice="2"):

    with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    vertices = data["vertices"]
    edges_raw = data["edges"]
    edges_named = [(vertices[e["vertex1"]]["name"], vertices[e["vertex2"]]["name"]) for e in edges_raw]
    vertex_names = [v["name"] for v in vertices]

    if choice == "1":
        # Используем номера из поля name
        try:
            order_from_file = [v["name"] for v in sorted(vertices, key=lambda x: int(x["name"]))]
            length_file, name_to_number_file = numbering_length(order_from_file, edges_named, verbose=True)
            visualize_graph(vertices, edges_raw, name_to_number_file, length_file, "graph_from_file.png")

            print(f"\n✅ Готовая нумерация посчитана: длина = {length_file}")

        except Exception as e:
            print(f"Ошибка при чтении нумерации из файла: {e}")


    else:
        # --- Минимальная нумерация ---
        order_min = vertex_names[:]
        _, order_min = iterated_2opt(order_min, edges_named, maximize=False)
        length_min, name_to_number_min = numbering_length(order_min, edges_named, verbose=True)
        visualize_graph(vertices, edges_raw, name_to_number_min, length_min, "graph_min.png", maximize=False)

        # --- Максимальная нумерация ---
        order_max = random.sample(vertex_names, len(vertex_names))
        _, order_max = iterated_2opt(order_max, edges_named, maximize=True)
        length_max, name_to_number_max = numbering_length(order_max, edges_named, verbose=True)
        visualize_graph(vertices, edges_raw, name_to_number_max, length_max, "graph_max.png", maximize=True)

        print(f"\nМинимальная нумерация: длина = {length_min}")
        print(f"Максимальная нумерация: длина = {length_max}")

if __name__ == "__main__":
    main("example.graph")