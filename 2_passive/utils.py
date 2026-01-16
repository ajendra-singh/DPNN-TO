import matplotlib.pyplot as plt

def plotTO(xplot, yplot, rho_plot):

    # Plot
    _, ax = plt.subplots(figsize=(9, 5))

    ax.pcolor(
        xplot, yplot, rho_plot,
        cmap="gray_r", vmin=0, vmax=1,
        edgecolors="k", linewidths=0.25
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Optimized topology", fontsize=14)
    plt.show()