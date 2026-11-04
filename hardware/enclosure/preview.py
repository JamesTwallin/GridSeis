"""Render previews of the weather house: the assembled house in three positions, and the print layout.

A small painter's-algorithm renderer: meshes are subdivided into short triangles, back faces are
culled, triangles are sorted far to near and drawn as flat-shaded polygons. Good enough for
previews without an OpenGL stack."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts', 'primer'))
from style import title, INK  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

WOOD, ROOF, BASE, TURB, CHIM, TRIM = '#e2c9a0', '#7a3b2e', '#a8a29e', '#f1f5f9', '#57534e', '#8a5a3b'


def tris(m, refine=5.0):
    try: m = m.refine_to_length(refine)
    except Exception: pass
    mesh = m.to_mesh(); v = np.asarray(mesh.vert_properties)[:, :3]; return v[np.asarray(mesh.tri_verts)]


def draw(ax, parts, view=(0.45, 1.0, -0.38), light=(-0.5, -0.8, 0.9)):
    """parts: list of (triangles Nx3x3, colour). view: direction the camera looks along (into the scene)."""
    d = np.array(view, float); d /= np.linalg.norm(d); up = np.array([0, 0, 1.0])
    right = np.cross(d, up); right /= np.linalg.norm(right); upv = np.cross(right, d)
    L = np.array(light, float); L /= np.linalg.norm(L)
    polys, cols, depth = [], [], []
    for t, colour in parts:
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        vis = n @ d < 0; t, n = t[vis], n[vis]
        shade = 0.5 + 0.5 * np.clip(n @ L, 0, 1); c = np.clip(np.array(to_rgb(colour))[None] * shade[:, None], 0, 1)
        polys.append(np.stack([t @ right, t @ upv], axis=-1)); cols.append(c); depth.append((t.mean(axis=1)) @ d)
    P, C, Z = np.concatenate(polys), np.concatenate(cols), np.concatenate(depth); o = np.argsort(-Z)
    ax.add_collection(PolyCollection(P[o], facecolors=C[o], edgecolors=C[o], linewidths=0.25, antialiased=True))
    ax.set_xlim(P[..., 0].min() - 5, P[..., 0].max() + 5); ax.set_ylim(P[..., 1].min() - 5, P[..., 1].max() + 5); ax.set_aspect('equal'); ax.set_axis_off()


def render(base, house, roof, ridge, figures_at, angle_max, outdir):
    """figures_at(angle) gives [(manifold, colour), ...] for the bar and figures at that servo angle."""
    shell = [(tris(base), BASE), (tris(house, 3.0), WOOD), (tris(roof, 3.0), ROOF), (tris(ridge, 3.0), TRIM)]
    fig = plt.figure(figsize=(12, 5.6))
    poses = [(-angle_max, 'Grid ringing loudly: low carbon'), (0.0, 'Middle of the week'), (angle_max, 'Grid quiet: high carbon')]
    for k, (a, lab) in enumerate(poses):
        ax = fig.add_axes([0.01 + k * 0.33, 0.02, 0.31, 0.78])
        draw(ax, shell + [(tris(m, 1.5), c) for m, c in figures_at(a)], view=(0.32, 1.0, -0.26))
        ax.set_title(lab, fontsize=11, color=INK)
    bbs = np.array([m.bounding_box() for m in (base, house, roof, ridge)])
    size = f'{bbs[:, 3].max() - bbs[:, 0].min():.0f} mm wide, {bbs[:, 4].max() - bbs[:, 1].min():.0f} mm deep, {bbs[:, 5].max() - bbs[:, 2].min():.0f} mm tall to the finials'
    title(fig, 'GridSeis weather house', y=0.95, size=18)
    fig.text(0.02, 0.905, 'An ESP32-S3 listens to the mains hum on a wire and turns a servo: the wind turbine walks out of the right-hand door when the grid rings loudly'
             + chr(10) + 'for the week, the power station out of the left-hand door when it is quiet. Rendered from the STL files; ' + size + '.', fontsize=10, color=INK, va='top')
    fig.savefig(os.path.join(outdir, 'weather_house.png'), dpi=170, bbox_inches='tight', pad_inches=0.2); plt.close(fig); print('saved weather_house.png')


def render_layout(stl_dir, outdir):
    import trimesh
    rows = [[('base.stl', BASE), ('house.stl', WOOD)], [('roof.stl', ROOF), ('bar.stl', TRIM),
            ('turbine_tower.stl', TURB), ('turbine_rotor.stl', TURB), ('power_station.stl', CHIM)]]
    parts, y = [], 0.0
    for row in rows:
        x, depth = 0.0, 0.0
        for nm, col in row:
            mm = trimesh.load(os.path.join(stl_dir, nm), force='mesh'); v = mm.vertices - mm.vertices.min(axis=0)
            v[:, 0] += x; v[:, 1] += y; x += np.ptp(mm.vertices[:, 0]) + 12; depth = max(depth, np.ptp(mm.vertices[:, 1]))
            parts.append((v[mm.faces], col))
        y -= depth + 15
    fig = plt.figure(figsize=(11, 6)); ax = fig.add_axes([0.0, 0.0, 1.0, 0.84])
    draw(ax, parts, view=(0.3, 0.9, -0.75))
    title(fig, 'Print layout: seven pieces, no supports', y=0.96, size=18)
    fig.text(0.02, 0.91, 'Top row: base and house, printed upright. Bottom row: roof standing on its front edge, bar and turbine rotor printed flat; turbine tower and power station upright.', fontsize=10, color=INK, va='top')
    fig.savefig(os.path.join(outdir, 'print_layout.png'), dpi=150, bbox_inches='tight', pad_inches=0.2); plt.close(fig); print('saved print_layout.png')
