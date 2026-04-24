"""
Семінарське заняття 4.

Завдання 1: Розподіл ціни акцій через t=1,2,3 кроки
Завдання 2: Середня вартість акції через t=1,2,3 кроки
Завдання 3: Графік залежності ціни від ймовірності p
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

# Параметри 
S0_task1 = 100       # Початкова вартість (Завдання 1)
S0_task2 = 1000      # Початкова вартість (Завдання 2)
ALPHA = -0.1         # Зниження ціни
BETA = 0.15          # Зростання ціни
P_VALUES = [0.1, 0.25, 0.5]
N_SIM = 1000         # Кількість реалізацій для моделювання


# Допоміжні функції

def binomial_coeff(n, k):
    """Біноміальний коефіцієнт C(n, k)."""
    from math import comb
    return comb(n, k)

def stock_price(S0, ups, downs, beta=BETA, alpha=ALPHA):
    """Ціна акції після ups зростань і downs падінь."""
    return S0 * (1 + beta) ** ups * (1 + alpha) ** downs

def distribution_at_step(S0, t, p, alpha=ALPHA, beta=BETA):
    """
    Повертає список (ціна, ймовірність) для всіх можливих станів на кроці t.
    Стани: k зростань (beta) та (t-k) падінь (alpha), k=0..t
    """
    q = 1 - p
    states = []
    for k in range(t + 1):
        downs = t - k
        ups = k
        price = stock_price(S0, ups, downs, beta, alpha)
        prob = binomial_coeff(t, k) * (q ** k) * (p ** downs)
        states.append((price, prob))
    return states

def mean_price(S0, t, p, alpha=ALPHA, beta=BETA):
    """Аналітичне середнє значення ціни: S0*(1 + alpha*p + beta*(1-p))^t"""
    q = 1 - p
    mu = 1 + alpha * p + beta * q
    return S0 * mu ** t

def simulate_price(S0, t, p, n_sim=N_SIM, alpha=ALPHA, beta=BETA):
    """Моделювання N реалізацій ціни акції на кроці t (метод Монте-Карло)."""
    rng = np.random.default_rng(42)
    prices = []
    for _ in range(n_sim):
        price = S0
        for _ in range(t):
            if rng.random() < p:
                price *= (1 + alpha)
            else:
                price *= (1 + beta)
        prices.append(price)
    return np.array(prices)


# ЗАВДАННЯ 1 

def task1():
    print("=" * 65)
    print("ЗАВДАННЯ 1. Розподіл ціни акцій (S0 = 100)")
    print(f"  alpha = {ALPHA},  beta = {BETA}")
    print("=" * 65)

    for t in range(1, 4):
        print(f"\n--- Крок t = {t} ---")
        header = f"{'p':>6}  " + "  ".join(
            [f"S{t}_{k+1} = {stock_price(S0_task1, k, t-k):.2f}" for k in range(t+1)]
        )
        print(header)
        print("-" * len(header))
        for p in P_VALUES:
            states = distribution_at_step(S0_task1, t, p)
            row = f"{p:>6.2f}  " + "  ".join([f"{'P='+str(round(prob,4)):>14}" for _, prob in states])
            print(row)

        # Моделювання
        print(f"\n  [Моделювання, N={N_SIM} реалізацій, p=0.1]")
        sim = simulate_price(S0_task1, t, 0.1)
        unique, counts = np.unique(np.round(sim, 2), return_counts=True)
        for price, cnt in zip(unique, counts):
            print(f"    Ціна {price:7.2f}: {cnt:4d} разів")
        print(f"    Середнє: {sim.mean():.4f},  Дисперсія: {sim.var():.4f}")


# ЗАВДАННЯ 2 

def task2():
    print("\n" + "=" * 65)
    print("ЗАВДАННЯ 2. Середня вартість акції (S0 = 1000)")
    print("=" * 65)

    scenarios = [
        ("Сценарій 1: alpha=-0.1, beta=+0.1", -0.1, 0.1),
        ("Сценарій 2: alpha=-0.2, beta=+0.1", -0.2, 0.1),
    ]

    for label, alpha2, beta2 in scenarios:
        print(f"\n{label}")
        header = f"  {'p':>6}  {'t=1':>10}  {'t=2':>10}  {'t=3':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for p in P_VALUES:
            means = [mean_price(S0_task2, t, p, alpha2, beta2) for t in range(1, 4)]
            print(f"  {p:>6.2f}  " + "  ".join([f"{m:>10.4f}" for m in means]))


# ЗАВДАННЯ 3 

def task3():
    """Будує графіки залежності середньої ціни акцій від ймовірності p."""
    print("\n" + "=" * 65)
    print("ЗАВДАННЯ 3. Графік залежності ціни від p")
    print("=" * 65)

    p_range = np.linspace(0, 1, 200)
    scenarios = [
        ("Сценарій 1: α=−0.1, β=+0.1", -0.1, 0.1),
        ("Сценарій 2: α=−0.2, β=+0.1", -0.2, 0.1),
    ]
    colors = ['#2563eb', '#16a34a', '#dc2626']
    linestyles = ['-', '--', ':']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Середня ціна акцій залежно від ймовірності p\n(S₀ = 1000)",
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, (label, alpha2, beta2) in zip(axes, scenarios):
        for t, color, ls in zip([1, 2, 3], colors, linestyles):
            means = [mean_price(S0_task2, t, p, alpha2, beta2) for p in p_range]
            ax.plot(p_range, means, color=color, linestyle=ls, linewidth=2,
                    label=f"t = {t}")

        # Позначення p з умови задачі
        for p_mark in P_VALUES:
            ax.axvline(p_mark, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Ймовірність p (падіння)", fontsize=10)
        ax.set_ylabel("Середня ціна акції", fontsize=10)
        ax.legend(title="Крок", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("task3_graph.png", dpi=150, bbox_inches='tight')
    print("  Графік збережено: task3_graph.png")
    plt.close()


def plot_histograms():
    """Гістограми розподілу змодельованих цін для t=1,2,3 (p=0.1)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Гістограми розподілу цін акцій (S₀=100, p=0.1, N=1000)",
                 fontsize=13, fontweight='bold')
    colors_bar = ['#93c5fd', '#6ee7b7', '#fca5a5']
    for idx, t in enumerate([1, 2, 3]):
        ax = axes[idx]
        sim = simulate_price(S0_task1, t, 0.1)
        unique, counts = np.unique(np.round(sim, 2), return_counts=True)
        freqs = counts / counts.sum()
        states = distribution_at_step(S0_task1, t, 0.1)
        theo_prices = [s[0] for s in states]
        theo_probs  = [s[1] for s in states]
        ax.bar(range(len(unique)), freqs, color=colors_bar[idx],
               edgecolor='steelblue', linewidth=0.8, label='Симуляція', zorder=3)
        ax.plot(range(len(theo_prices)), theo_probs, 'o--',
                color='#1e40af', markersize=7, linewidth=1.5, label='Теорія', zorder=4)
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels([f"{p:.1f}" for p in unique], fontsize=9)
        ax.set_title(f"t = {t}", fontsize=11)
        ax.set_xlabel("Ціна акції", fontsize=9)
        ax.set_ylabel("Частота / Ймовірність", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("histograms.png", dpi=150, bbox_inches='tight')
    print("  Гістограми збережено: histograms.png")
    plt.close()


def variance_analysis():
    """Аналіз зростання дисперсії з кроками (p=0.1, S0=100)."""
    print("\n--- Аналіз дисперсії (p=0.1) ---")
    print(f"  {'Крок t':>8}  {'Теор. середнє':>16}  {'Симул. середнє':>16}  "
          f"{'Симул. дисперсія':>18}  {'СКВ':>10}")
    print("  " + "-" * 76)
    for t in range(1, 4):
        sim = simulate_price(S0_task1, t, 0.1)
        theo_mean = mean_price(S0_task1, t, 0.1, ALPHA, BETA)
        print(f"  {t:>8}  {theo_mean:>16.4f}  {sim.mean():>16.4f}  "
              f"{sim.var():>18.4f}  {sim.std():>10.4f}")


# ДЕРЕВО ЦІН (Binomial Tree) 

def plot_binomial_tree():
    """Візуалізація біноміального дерева для t=0..3."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title("Біноміальне дерево ціни акції (S₀=100, t=0..3)",
                 fontsize=12, fontweight='bold')

    p_tree = 0.1
    node_positions = {}  # (t, k) -> (x, y)

    for t in range(4):
        for k in range(t + 1):
            x = t
            y = k - t / 2
            node_positions[(t, k)] = (x, y)

    # Ребра
    for t in range(3):
        for k in range(t + 1):
            x0, y0 = node_positions[(t, k)]
            # Зростання (k -> k+1)
            x1, y1 = node_positions[(t + 1, k + 1)]
            ax.plot([x0, x1], [y0, y1], 'k-', lw=1, alpha=0.5)
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mid_x - 0.12, mid_y + 0.08, f"q={1-p_tree}", fontsize=7, color='#16a34a')
            # Падіння (k -> k)
            x2, y2 = node_positions[(t + 1, k)]
            ax.plot([x0, x2], [y0, y2], 'k-', lw=1, alpha=0.5)
            mid_x2, mid_y2 = (x0 + x2) / 2, (y0 + y2) / 2
            ax.text(mid_x2 + 0.03, mid_y2 - 0.12, f"p={p_tree}", fontsize=7, color='#dc2626')

    # Вузли
    for (t, k), (x, y) in node_positions.items():
        ups = k
        downs = t - k
        price = stock_price(S0_task1, ups, downs)
        circle = plt.Circle((x, y), 0.22, color='#dbeafe', ec='#2563eb', lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f"{price:.1f}", ha='center', va='center',
                fontsize=8, fontweight='bold', color='#1e3a5f', zorder=4)

    # Підписи осі t
    for t in range(4):
        ax.text(t, -2.0, f"t = {t}", ha='center', fontsize=10, color='gray')

    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-2.3, 1.8)
    plt.tight_layout()
    plt.savefig("binomial_tree.png", dpi=150, bbox_inches='tight')
    print("  Дерево збережено: binomial_tree.png")
    plt.close()


# Головна програма 

if __name__ == "__main__":
    task1()
    variance_analysis()
    task2()
    task3()
    plot_histograms()
    plot_binomial_tree()
    print("\nГотово!")
    