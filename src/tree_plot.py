import heapq
import itertools
import matplotlib.pyplot as plt
import numpy as np

def n_samples(node):
    return int(sum(node.class_counts.values())) if node.class_counts else 0


def is_internal(node):
    return len(node.children) > 0 or node.left is not None or node.right is not None


def kids(node, categorical, max_children):
    # Deskripsi node : label_cabang, anak, jumlah_kategori_tersembunyi)

    if node.feature in categorical:
        items = sorted(node.children.items(), key=lambda kv: - n_samples(kv[1]))
        shown = items[:max_children]
        return [(f"= {k}", ch) for k, ch in shown], len(items) - len(shown)
    out = []
    if node.left is not None:
        out.append((f"<= {node.threshold:.3g}", node.left))
    if node.right is not None:
        out.append((f"> {node.threshold:.3g}", node.right))
    return out, 0


def plot_tree(model, top_n=10, feature_names=None, class_names=("Tidak batal (0)", "Batal (1)"), max_children=4, path=None, dpi=150, show=True):

    if model.root is None:
        raise ValueError("Model belum di-fit.")
    categorical = set(model.categorical_idx)
    name_of = (lambda i: feature_names[i]) if feature_names is not None else (lambda i: f"X[{i}]")

    # pilih node yang diperluas (best-first berdasarkan jumlah sampel)
    counter = itertools.count()
    heap = [(-n_samples(model.root), next(counter), model.root)]
    expanded = {}  
    while heap and len(expanded) < top_n:
        _, _, node = heapq.heappop(heap)
        if not is_internal(node):
            continue
        child_list, hidden = kids(node, categorical, max_children)
        expanded[id(node)] = (child_list, hidden)
        for _, ch in child_list:
            heapq.heappush(heap, (-n_samples(ch), next(counter), ch))

    # tata letak: x = urutan daun, y = -kedalaman
    pos, edges, nodes = {}, [], []
    leaf_x = itertools.count()

    def layout(node, depth, branch):
        key = id(node)
        if key in expanded:
            kids, _ = expanded[key]
            xs = []
            for lab, ch in kids:
                cx = layout(ch, depth + 1, lab)
                xs.append(cx)
                edges.append((key, id(ch), lab))
            x = float(np.mean(xs))
        else:
            x = float(next(leaf_x))
        pos[key] = (x, -depth)
        nodes.append(node)
        return x

    layout(model.root, 0, None)
    n_leaf = next(leaf_x)
    depth_max = max(-y for _, y in pos.values())

    # gambar tree
    fig_w = max(8.0, 2.3 * n_leaf)
    fig_h = max(4.0, 2.0 * (depth_max + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("RdBu_r")
    for a, b, lab in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color="#6b7280", lw=1, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, lab, fontsize=7, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=2)

    total = n_samples(model.root)
    for node in nodes:
        n = n_samples(node)
        p1 = node.class_counts.get(1, 0) / n if n else 0.0
        c0 = int(node.class_counts.get(0, 0))
        c1 = int(node.class_counts.get(1, 0))
        if id(node) in expanded:
            hidden = expanded[id(node)][1]
            head = f"{name_of(node.feature)}"
            if node.feature not in categorical:
                head += f" <= {node.threshold:.3g}?"
            if hidden:
                head += f"\n(+{hidden} kategori lain)"
        else:
            head = (f"{class_names[1] if p1 >= 0.5 else class_names[0]}"
                    + (" ..." if is_internal(node) else ""))
        txt = f"{head}\nn={n} ({n / total:.0%})\n{class_names[0]}={c0} | {class_names[1]}={c1}"
        x, y = pos[id(node)]
        ax.text(x, y, txt, fontsize=7, ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", fc=cmap(0.15 + 0.7 * p1), ec="#374151",
                          alpha=0.85))

    ax.set_xlim(-0.8, max(n_leaf - 1, 0) + 0.8)
    ax.set_ylim(-depth_max - 0.7, 0.7)
    ax.axis("off")
    ax.set_title(f"Decision Tree (C4.5 scratch) - top {len(expanded)} percabangan"
                 f"\n(warna: biru = {class_names[0]}, merah = {class_names[1]})", fontsize=10)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig