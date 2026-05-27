"""
TP03 – Paralelização da Geração de Fractais
Programação Concorrente e Distribuída – UCB 2026/1
"""

import matplotlib
matplotlib.use('Agg')  # backend sem janela (compatível com threads)

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
import threading

# ---------------------------------------------------------------------------
# Diretório de saída – área de trabalho do usuário
# ---------------------------------------------------------------------------
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(DESKTOP, exist_ok=True)

# Lock global para operações do matplotlib que não são thread-safe
_plt_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Utilitário IFS
# ---------------------------------------------------------------------------
def gerar_fractal(transformacoes, probabilidades, iteracoes=100_000):
    if not abs(sum(probabilidades) - 1.0) < 1e-6:
        raise ValueError("As probabilidades devem somar 1.")
    x, y = 0.0, 0.0
    pontos = []
    for _ in range(iteracoes):
        r = random.random()
        acumulado = 0.0
        for i, prob in enumerate(probabilidades):
            acumulado += prob
            if r < acumulado:
                x, y = transformacoes[i](x, y)
                break
        pontos.append((x, y))
    return pontos


# ---------------------------------------------------------------------------
# Funções de cada fractal
# ---------------------------------------------------------------------------

def sierpinski():
    """Triângulo de Sierpinski."""
    transformacoes = [
        lambda x, y: (0.5 * x,        0.5 * y),
        lambda x, y: (0.5 * x + 0.5,  0.5 * y),
        lambda x, y: (0.5 * x + 0.25, 0.5 * y + 0.5),
    ]
    probabilidades = [1/3, 1/3, 1/3]
    pontos = gerar_fractal(transformacoes, probabilidades)
    x_vals, y_vals = zip(*pontos)

    with _plt_lock:
        fig = plt.figure()
        plt.scatter(x_vals, y_vals, s=0.1, color='black', marker='.')
        plt.title("Triângulo de Sierpinski")
        plt.axis('off')
        plt.savefig(os.path.join(DESKTOP, "sierpinski.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] sierpinski.png")


def samambaia_barnsley():
    """Samambaia de Barnsley."""
    transformacoes = [
        lambda x, y: (0.0,                    0.16 * y),
        lambda x, y: (0.85*x + 0.04*y,       -0.04*x + 0.85*y + 1.6),
        lambda x, y: (0.2*x  - 0.26*y,        0.23*x + 0.22*y + 1.6),
        lambda x, y: (-0.15*x + 0.28*y,       0.26*x + 0.24*y + 0.44),
    ]
    probabilidades = [0.01, 0.85, 0.07, 0.07]
    pontos = gerar_fractal(transformacoes, probabilidades)
    x_vals, y_vals = zip(*pontos)

    with _plt_lock:
        fig = plt.figure()
        plt.scatter(x_vals, y_vals, s=0.1, color='green', marker='.')
        plt.title("Samambaia de Barnsley")
        plt.axis('off')
        plt.savefig(os.path.join(DESKTOP, "samambaia_barnsley.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] samambaia_barnsley.png")


def mandelbrot(width=800, height=800, max_iter=100):
    """Conjunto de Mandelbrot."""
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    image = np.zeros((height, width))
    for row in range(height):
        for col in range(width):
            c = complex(x_min + (x_max - x_min) * col / width,
                        y_min + (y_max - y_min) * row / height)
            z = 0.0j
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z * z + c
                n += 1
            image[row, col] = n

    with _plt_lock:
        fig = plt.figure()
        plt.imshow(image,
                   extent=(x_min, x_max, y_min, y_max),
                   cmap='hot', interpolation='bilinear')
        plt.title("Conjunto de Mandelbrot")
        plt.axis('off')
        plt.savefig(os.path.join(DESKTOP, "mandelbrot.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] mandelbrot.png")


def julia(c=-0.7 + 0.27015j, width=800, height=800, max_iter=100):
    """Conjunto de Julia."""
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    image = np.zeros((height, width))
    for row in range(height):
        for col in range(width):
            z = complex(x_min + (x_max - x_min) * col / width,
                        y_min + (y_max - y_min) * row / height)
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z * z + c
                n += 1
            image[row, col] = n

    with _plt_lock:
        fig = plt.figure()
        plt.imshow(image,
                   extent=(x_min, x_max, y_min, y_max),
                   cmap='twilight_shifted', interpolation='bilinear')
        plt.title("Conjunto de Julia")
        plt.axis('off')
        plt.savefig(os.path.join(DESKTOP, "julia.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] julia.png")


def koch_curve(order=4, size=300):
    """Curva de Koch."""
    def koch_recursive(points, order):
        if order == 0:
            return points
        new_points = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            new_points.append(p1)
            new_points.append((p1[0] + dx / 3,
                                p1[1] + dy / 3))
            new_points.append((p1[0] + dx / 2 - dy * np.sqrt(3) / 6,
                                p1[1] + dy / 2 + dx * np.sqrt(3) / 6))
            new_points.append((p1[0] + 2 * dx / 3,
                                p1[1] + 2 * dy / 3))
        new_points.append(points[-1])
        return koch_recursive(new_points, order - 1)

    points = [(0, 0), (size, 0)]
    points = koch_recursive(points, order)
    x_vals, y_vals = zip(*points)

    with _plt_lock:
        fig = plt.figure()
        plt.plot(x_vals, y_vals, color='blue', linewidth=1)
        plt.title("Curva de Koch")
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(os.path.join(DESKTOP, "koch_curve.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] koch_curve.png")


def fractal_tree():
    """Árvore Fractal."""
    def draw_tree(ax, x, y, length, angle, depth):
        if depth == 0:
            return
        x_end = x + length * np.cos(np.radians(angle))
        y_end = y + length * np.sin(np.radians(angle))
        ax.plot([x, x_end], [y, y_end], color='brown', linewidth=1)
        draw_tree(ax, x_end, y_end, length * 0.7, angle - 30, depth - 1)
        draw_tree(ax, x_end, y_end, length * 0.7, angle + 30, depth - 1)

    with _plt_lock:
        fig, ax = plt.subplots()
        draw_tree(ax, 0, 0, 100, 90, 8)
        plt.title("Árvore Fractal")
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(os.path.join(DESKTOP, "fractal_tree.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] fractal_tree.png")


def sierpinski_carpet(size=3, iterations=4):
    """Tapete de Sierpinski."""
    carpet = np.ones((size**iterations, size**iterations))

    def recursive_remove(grid, x, y, size, iteration):
        if iteration == 0:
            return
        sub_size = size // 3
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    grid[x + sub_size:x + 2*sub_size,
                         y + sub_size:y + 2*sub_size] = 0
                else:
                    recursive_remove(grid,
                                     x + i * sub_size,
                                     y + j * sub_size,
                                     sub_size, iteration - 1)

    recursive_remove(carpet, 0, 0, size**iterations, iterations)

    with _plt_lock:
        fig = plt.figure()
        plt.imshow(carpet, cmap='gray_r')
        plt.title("Tapete de Sierpinski")
        plt.axis('off')
        plt.savefig(os.path.join(DESKTOP, "sierpinski_carpet.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] sierpinski_carpet.png")


def menger_sponge(iterations=2):
    """Esponja de Menger."""
    def generate_sponge(grid, x, y, z, size, iteration):
        if iteration == 0:
            return
        sub_size = size // 3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if (i == 1 and j == 1) or \
                       (i == 1 and k == 1) or \
                       (j == 1 and k == 1):
                        continue
                    generate_sponge(grid,
                                    x + i * sub_size,
                                    y + j * sub_size,
                                    z + k * sub_size,
                                    sub_size, iteration - 1)

    grid_size = 3**iterations
    grid = np.ones((grid_size, grid_size, grid_size))
    generate_sponge(grid, 0, 0, 0, grid_size, iterations)

    with _plt_lock:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(grid, edgecolor='k')
        plt.title("Esponja de Menger")
        plt.savefig(os.path.join(DESKTOP, "menger_sponge.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    print("  [OK] menger_sponge.png")


# ---------------------------------------------------------------------------
# Execução SEQUENCIAL (para medir tempo de referência)
# ---------------------------------------------------------------------------
def gerar_todos_sequencial():
    funcoes = [
        sierpinski,
        samambaia_barnsley,
        mandelbrot,
        julia,
        koch_curve,
        fractal_tree,
        sierpinski_carpet,
        menger_sponge,
    ]
    for f in funcoes:
        print(f"  Gerando {f.__name__}...")
        f()


# ---------------------------------------------------------------------------
# Execução PARALELA com threads
# ---------------------------------------------------------------------------
def gerar_todos_paralelo():
    """
    Cria uma thread independente para cada fractal e aguarda todas
    terminarem antes de retornar (uso de thread.join()).
    """
    funcoes = [
        sierpinski,
        samambaia_barnsley,
        mandelbrot,
        julia,
        koch_curve,
        fractal_tree,
        sierpinski_carpet,
        menger_sponge,
    ]

    threads = []
    for f in funcoes:
        t = threading.Thread(target=f, name=f"Thread-{f.__name__}")
        threads.append(t)

    # Inicia todas as threads
    for t in threads:
        print(f"  Iniciando {t.name}...")
        t.start()

    # Aguarda todas terminarem (barrier implícita via join sequencial)
    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# main – mede e compara os dois modos
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  TP03 – Geração Paralela de Fractais")
    print("  Programação Concorrente e Distribuída – UCB 2026/1")
    print("=" * 60)

    # --- Sequencial ---
    print("\n[1/2] Execução SEQUENCIAL")
    t0 = time.perf_counter()
    gerar_todos_sequencial()
    t_seq = time.perf_counter() - t0
    print(f"  Tempo sequencial : {t_seq:.2f}s")

    # --- Paralelo ---
    print("\n[2/2] Execução PARALELA (threads)")
    t0 = time.perf_counter()
    gerar_todos_paralelo()
    t_par = time.perf_counter() - t0
    print(f"  Tempo paralelo   : {t_par:.2f}s")

    # --- Comparação ---
    speedup = t_seq / t_par if t_par > 0 else float('inf')
    print("\n" + "=" * 60)
    print(f"  Speedup obtido   : {speedup:.2f}x")
    print(f"  Imagens salvas em: {DESKTOP}")
    print("=" * 60)


if __name__ == "__main__":
    main()