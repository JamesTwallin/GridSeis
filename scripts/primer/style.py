"""House plotting style shared by the grid-frequency primer figures (Noto Sans Display, bold upper-case left titles)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
for font in ['NotoSansDisplay-Regular.ttf', 'NotoSansDisplay-Bold.ttf', 'NotoSansSymbols-Regular.ttf', 'NotoSansMath-Regular.ttf']:
    fp = os.path.join(HERE, font)
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)

plt.rcParams.update({
    'figure.titlesize': 14, 'axes.titlesize': 12, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'font.size': 10, 'axes.unicode_minus': False,
    'font.family': 'Noto Sans Display', 'font.weight': 'normal',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Noto Sans Math', 'mathtext.it': 'Noto Sans Math', 'mathtext.bf': 'Noto Sans Math:bold',
})

INK = '#1f2937'
GREY = '#9ca3af'
# fuel colours, fixed order (spinning machines first, then inverter-connected)
FUEL_COLOURS = {
    'Gas': '#c2410c', 'Coal': '#374151', 'Nuclear': '#7c3aed', 'Biomass': '#92400e', 'Hydro': '#1d4ed8',
    'Wind': '#0f766e', 'Solar': '#eab308', 'Imports': '#9ca3af', 'Storage': '#db2777', 'Other': '#d1d5db',
}
HEAVY = '#374151'   # high-inertia / synchronous
LIGHT = '#0f766e'   # low-inertia / inverter


def tidy(ax, ygrid=True):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, zorder=0, alpha=0.3)
    ax.set_axisbelow(True)


def title(fig, text, y=0.95, size=18):
    fig.suptitle(text.upper(), x=0.02, y=y, horizontalalignment='left', verticalalignment='bottom', fontweight='bold',
                 transform=fig.transFigure, fontsize=size)


def source(ax, text):
    ax.text(0.0, -0.22, text, transform=ax.transAxes, fontsize=9, color='grey', verticalalignment='top', horizontalalignment='left')


def save(fig, name, top=0.86):
    fig.subplots_adjust(top=top)
    fig.savefig(os.path.join(HERE, name), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)
