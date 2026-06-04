import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Carica la Q-table
q_table = np.load(
    "blackjack_env/model/policy.npy",
    allow_pickle=True
).item()

# 0 = Stand, 1 = Hit
policy_no_ace = np.full((10, 10), np.nan)
policy_ace = np.full((10, 10), np.nan)

player_sums = range(12, 22)
dealer_cards = range(1, 11)

for i, player in enumerate(player_sums):
    for j, dealer in enumerate(dealer_cards):

        state = (player, dealer, 0)
        if state in q_table:
            policy_no_ace[i, j] = np.argmax(q_table[state])

        state = (player, dealer, 1)
        if state in q_table:
            policy_ace[i, j] = np.argmax(q_table[state])

# Colori professionali
# Blu = Stand
# Rosso = Hit
cmap = ListedColormap([
    "#2563EB",  # Stand
    "#DC2626"   # Hit
])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 8),
    constrained_layout=True
)

for ax, data, title in zip(
    axes,
    [policy_no_ace, policy_ace],
    ["No Usable Ace", "Usable Ace"]
):

    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="lower",
        aspect="equal"
    )

    # Griglia
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)

    ax.grid(
        which="minor",
        color="white",
        linewidth=2
    )

    ax.tick_params(which="minor", bottom=False, left=False)

    # Testo nelle celle
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            if np.isnan(data[i, j]):
                continue

            action = int(data[i, j])

            label = "S" if action == 0 else "H"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold"
            )

    ax.set_xticks(range(10))
    ax.set_xticklabels(range(1, 11), fontsize=11)

    ax.set_yticks(range(10))
    ax.set_yticklabels(range(12, 22), fontsize=11)

    ax.set_xlabel("Dealer Showing", fontsize=12)
    ax.set_ylabel("Player Sum", fontsize=12)

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold"
    )

# Legenda
from matplotlib.patches import Patch


fig.suptitle(
    "Blackjack Policy Learned with Q-Learning",
    fontsize=18,
    fontweight="bold"
)

plt.savefig(
    "blackjack_policy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()