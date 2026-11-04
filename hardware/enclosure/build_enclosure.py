"""GridSeis weather house: parametric enclosure, exported as STL files ready to slice.

An alpine weather house in the style of the Austrian Wetterhäuschen: a timber chalet on a stone ground
storey. The servo stands in the stone storey and turns a bar just above the floor, carrying two 3D
figures: a wind turbine and a power station. When the grid is ringing loudly (light, windy, low carbon)
the turbine walks out of the right-hand door onto the porch; when it is quiet (heavy, gas-heavy, high
carbon) the power station walks out of the left-hand door.

Parts (all print without supports):
  base.stl            stone storey: servo posts, ESP32-S3-Zero on two rails, USB-C opening, antenna-wire hole
  house.stl           stage floor with the shaft hole, boarded walls, framed arched doors, gables, porch
  roof.stl            the whole roof in one piece: two fish-scale panels and the ridge cap; sits loose on the gables,
                      printed standing on its front edge
  bar.stl             figure bar, pushes straight onto the MG996R's shaft and is held by its centre screw
  turbine_tower.stl   upright; the rotor's spigot slots into the hole in the front of the nacelle
  turbine_rotor.stl   flat, blades on the bed
  power_station.stl   upright

Also checks that the figures clear the house over the full servo travel, that they get fully out of
their doors and stay on the porch, and renders previews to hardware/images/.
Usage: python hardware/enclosure/build_enclosure.py
Units are millimetres. Coordinates: x across the front, y from the front wall (y=0) to the back, z up."""
import math, os, struct
import numpy as np
from manifold3d import CrossSection, Manifold, OpType

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'stl'); IMG = os.path.join(os.path.dirname(HERE), 'images')
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
SEG = 64

# ---------------------------------------------------------------- parameters
W, D, PORCH, T = 84.0, 54.0, 22.0, 2.0         # outer width, house depth, porch depth, wall thickness
FLOOR_T, WALL_H = 3.0, 37.0                     # stage floor thickness, wall height above it
PITCH = 35.0                                    # roof pitch, degrees
DOOR_X, DOOR_W, DOOR_TOP = 19.0, 30.0, 34.0     # door centres at +-DOOR_X, width, height of the arch top above the floor
PIVOT_Y = 14.0                                  # servo shaft position behind the front wall
ARM_R = 27.5                                    # radius from the shaft to each figure (28.0 rubbed the right-hand door with an off-centre servo)
BAR_Z = 1.2                                     # bar underside above the stage floor
BAR_T = 4.0
ARM_W = 5.0                                     # width of the bar arms
ANGLE_MAX = 60.0                                # servo travel either side of centre; firmware ANGLE_MAX must match
SPIGOT_R, SPIGOT_L, SPIGOT_CLEAR = 0.9, 3.0, 0.1  # the rotor's spigot, and the clearance of the hole it slots into
# MG996R servo standing in the base, shaft at the origin of its own frame, body along +x; the bar's hub reaches
# down through the stage floor onto the shaft (measure yours: SERVO_TAB_TOP and SHAFT_ABOVE_TAB set the floor height)
SERVO_BODY = (-10.2, 30.5, -9.85, 9.85)         # x0, x1, y0, y1 of the case around the shaft
SERVO_TABS = (-17.0, 37.3)                      # x reach of the mounting tabs
SERVO_HOLES = [(x, y) for x in (-14.6, 34.9) for y in (-5.0, 5.0)]   # the four tab screw holes
SERVO_TAB_TOP, SERVO_TAB_T = 28.5, 2.5          # case bottom to the top of the tabs; tab thickness
SERVO_ABOVE_TAB = 11.0                          # case top above the tabs, not counting the output shaft
SHAFT_ABOVE_TAB = 15.0                          # top of the splined output shaft above the top of the tabs
SHAFT_R = 2.85                                  # hole for the 6 mm splined shaft: a push fit, the splines cut their own grooves
SHAFT_GRIP = 3.0                                # how far the shaft goes up into the hole
HUB_R, HUB_DROP = 5.5, 4.8                      # the bar's hub round the shaft, and how far it reaches below the bar
SCREW_R, SCREW_HEAD_R, SCREW_LEFT = 1.6, 3.3, 2.0  # centre screw: clearance hole, head recess, plastic left under the head
BASE_FLOOR = 2.0
# ESP32-S3-Zero (about 23.5 x 18 mm), USB-C end against the back wall
BOARD_X0, BOARD_W, BOARD_L, BOARD_RAIL_H = -9.2, 18.4, 23.0, 5.5
BOARD_Z = 2.0 + 4.0                             # underside of the board: raised so a USB-C plug's moulding clears the floor
USB_Z = BOARD_Z + 1.0 + 1.6                     # centre of the USB-C socket (1 mm board, 3.2 mm socket)
STAGE = BASE_FLOOR + SERVO_TAB_TOP + SHAFT_ABOVE_TAB - (BAR_Z - HUB_DROP + SHAFT_GRIP)   # stage floor top: set by the servo standing on the base floor
FLOOR_Z = STAGE - FLOOR_T                       # stage floor bottom in the assembly
BASE_H = FLOOR_Z                                # the base is a stone ground storey around the servo and the board
EAVE = STAGE + WALL_H
RIDGE = EAVE + (W / 2) * math.tan(math.radians(PITCH))
CLEAR = 0.3
TANP = math.tan(math.radians(PITCH))
TONGUE_H, TONGUE_W, TONGUE_IN = 0.8, 1.0, 2.0     # tongues on the gable tops: height, width, set back from the ends
RIDGE_R, RIDGE_SEAT = 3.0, 1.0                  # ridge cap radius; it sits on a flat this far above the apex of the gables


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def cyl(x, y, z0, h, r, seg=SEG):
    return Manifold.cylinder(h, r, r, seg).translate((x, y, z0))


def ccw(points):
    """Counter-clockwise order: the CAD library treats clockwise outlines as empty."""
    a = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))
    return list(points) if a > 0 else list(reversed(points))


def prism_xz(points, y0, y1):
    """Extrude a polygon given in (x, z) between y0 and y1 (a proper rotation, no mirroring)."""
    m = CrossSection([ccw(points)]).extrude(y1 - y0)   # polygon in xy, thickness along z
    return m.transform(np.array([[1, 0, 0, 0], [0, 0, -1, y1], [0, 1, 0, 0]], float))


def prism_yz(points, x0, x1):
    """Extrude a polygon given in (y, z) between x0 and x1 (a proper rotation, no mirroring)."""
    m = CrossSection([ccw(points)]).extrude(x1 - x0)
    return m.transform(np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], float))


def door_profile(cx, grow=0.0):
    """Round-topped doorway in (x, z) relative to the stage floor top: straight sides, then a half circle. grow widens it all round."""
    r = DOOR_W / 2 + grow; rect = DOOR_TOP - DOOR_W / 2; pts = [(cx - r, -1.0), (cx + r, -1.0)]
    for i in range(0, 41):
        a = math.radians(i * 180 / 40); pts.append((cx + r * math.cos(a), rect + r * math.sin(a)))
    return pts


def heart(cx, cz, s):
    pts = []
    for i in range(80):
        t = 2 * math.pi * i / 80
        x = 16 * math.sin(t) ** 3; z = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + s * x, cz + s * z))
    return pts


# ---------------------------------------------------------------- base: a stone plinth
PEGS = [(-W / 2 + 6, -PORCH + 6), (W / 2 - 6, -PORCH + 6), (-W / 2 + 6, D - 6), (W / 2 - 6, D - 6)]
GROOVE_W, GROOVE_D = 0.6, 0.5                   # decorative grooves: width and depth


def make_base():
    foot = 2.0                                  # the foot flares out by this much at the bottom, at 45 degrees
    b = Manifold.batch_hull([box(-W / 2 - foot, W / 2 + foot, -PORCH - foot, D + foot, 0, 1.2), box(-W / 2, W / 2, -PORCH, D, 0, BASE_H)])
    b -= box(-W / 2 + T, W / 2 - T, -PORCH + T, D - T, BASE_FLOOR, BASE_H + 1)
    # coursed stone: horizontal joints, and vertical joints staggered course to course
    courses = [1.2 + foot] + [1.2 + foot + (BASE_H - 1.2 - foot) * k / 3 for k in (1, 2)] + [BASE_H]
    cuts = []
    for k, (z0, z1) in enumerate(zip(courses[:-1], courses[1:])):
        if k: cuts += [box(-W, W, -PORCH - 5, -PORCH + GROOVE_D, z0 - GROOVE_W / 2, z0 + GROOVE_W / 2), box(-W, W, D - GROOVE_D, D + 5, z0 - GROOVE_W / 2, z0 + GROOVE_W / 2),
                       box(-W / 2 - 5, -W / 2 + GROOVE_D, -PORCH - 5, D + 5, z0 - GROOVE_W / 2, z0 + GROOVE_W / 2), box(W / 2 - GROOVE_D, W / 2 + 5, -PORCH - 5, D + 5, z0 - GROOVE_W / 2, z0 + GROOVE_W / 2)]
        off = 7.0 * (k % 2)
        for x in np.arange(-W / 2 + 7 + off, W / 2 - 3, 14):
            cuts += [box(x - GROOVE_W / 2, x + GROOVE_W / 2, -PORCH - 5, -PORCH + GROOVE_D, z0, z1), box(x - GROOVE_W / 2, x + GROOVE_W / 2, D - GROOVE_D, D + 5, z0, z1)]
        for y in np.arange(-PORCH + 7 + off, D - 3, 14):
            cuts += [box(-W / 2 - 5, -W / 2 + GROOVE_D, y - GROOVE_W / 2, y + GROOVE_W / 2, z0, z1), box(W / 2 - GROOVE_D, W / 2 + 5, y - GROOVE_W / 2, y + GROOVE_W / 2, z0, z1)]
    b -= Manifold.batch_boolean(cuts, OpType.Add)
    # blind arches on the front of the stone storey: a cellar door between two small windows
    def arch(cx, w, z0, top):
        r = w / 2; return [(cx - r, z0), (cx + r, z0)] + [(cx + r * math.cos(math.radians(a)), top - r + r * math.sin(math.radians(a))) for a in range(0, 181, 6)]
    for cx, w, z0, top in [(0.0, 14.0, foot + 1.2, BASE_H - 7), (-26.0, 9.0, BASE_H * 0.45, BASE_H - 9), (26.0, 9.0, BASE_H * 0.45, BASE_H - 9)]:
        b -= prism_xz(arch(cx, w, z0, top), -PORCH - 1, -PORCH + 1.2)                 # niche, 1.2 mm deep
    # servo stands on the base floor; its tabs screw to two posts
    for x0, x1 in [(SERVO_TABS[0], SERVO_BODY[0] - 2.5), (SERVO_BODY[1] + 2.5, SERVO_TABS[1])]:   # 2.5 mm from the case: room for the cable
        b += box(x0, x1, PIVOT_Y - 7.5, PIVOT_Y + 7.5, BASE_FLOOR - 0.01, BASE_FLOOR + SERVO_TAB_TOP - SERVO_TAB_T)
    for hx, hy in SERVO_HOLES:
        b -= cyl(hx, PIVOT_Y + hy, BASE_FLOOR + SERVO_TAB_TOP - SERVO_TAB_T - 10, 11, 1.25, 24)   # M3 self-tapping, 10 mm deep
    for x in (BOARD_X0 - 1.6, BOARD_X0 + BOARD_W):     # guide rails either side of the board
        b += box(x, x + 1.6, D - T - BOARD_L, D - T, BASE_FLOOR, BASE_FLOOR + BOARD_RAIL_H)
    bc = BOARD_X0 + BOARD_W / 2                         # support under the middle of the board, clear of the pins along its edges
    b += box(bc - 4, bc + 4, D - T - BOARD_L + 3, D - T - 2, BASE_FLOOR - 0.01, BOARD_Z)
    b -= box(bc - 6.5, bc + 6.5, D - T - 1, D + 5, USB_Z - 4.25, USB_Z + 4.25)   # USB-C plug and its moulding
    b -= Manifold.cylinder(T + 6, 1.9, 1.9, 32).rotate((-90, 0, 0)).translate((-W / 2 + 12, D - T - 1, BASE_H / 2))                          # antenna wire
    for x, y in PEGS:
        b += cyl(x, y, BASE_FLOOR, BASE_H - BASE_FLOOR, 4.0) + cyl(x, y, BASE_H, 2.0, 1.9, 32)   # corner posts with locating pegs
    return b


# ---------------------------------------------------------------- house (stage floor, walls, gables)
def board_grooves_x(x_face, outward, y0, y1, z0, z1, pitch=5.0):
    """Vertical board joints on a wall facing +-x."""
    xa, xb = (x_face - GROOVE_D, x_face + 5) if outward > 0 else (x_face - 5, x_face + GROOVE_D)
    return [box(xa, xb, y - GROOVE_W / 2, y + GROOVE_W / 2, z0, z1) for y in np.arange(y0, y1, pitch)]


def board_grooves_y(y_face, outward, x0, x1, z0, z1, pitch=5.0):
    """Vertical board joints on a wall facing +-y."""
    ya, yb = (y_face - GROOVE_D, y_face + 5) if outward > 0 else (y_face - 5, y_face + GROOVE_D)
    return [box(x - GROOVE_W / 2, x + GROOVE_W / 2, ya, yb, z0, z1) for x in np.arange(x0, x1, pitch)]


def side_window():
    """Blind window with shutters and a sill on the right-hand wall (mirrored for the left). Returns (raised, recessed)."""
    xf, yc, zc, pw, ph = W / 2, D / 2 + 2, STAGE + 27, 10.0, 13.0
    pane = [box(xf - 0.6, xf + 1, ya, yb, za, zb) for ya, yb in [(yc - pw / 2, yc - 0.5), (yc + 0.5, yc + pw / 2)]
            for za, zb in [(zc - ph / 2, zc - 0.5), (zc + 0.5, zc + ph / 2)]]           # four panes, cross mullion left standing
    frame = box(xf - 0.01, xf + 0.8, yc - pw / 2 - 1.5, yc + pw / 2 + 1.5, zc - ph / 2 - 1.5, zc + ph / 2 + 1.5) - box(xf - 1, xf + 2, yc - pw / 2, yc + pw / 2, zc - ph / 2, zc + ph / 2)
    raised = [frame]
    for sy in (-1, 1):                                  # shutters, folded open, with louvre lines
        y0 = yc + sy * (pw / 2 + 2.0); y1 = y0 + sy * 5.5
        sh = box(xf - 0.01, xf + 0.8, min(y0, y1), max(y0, y1), zc - ph / 2 - 1.5, zc + ph / 2 + 1.5)
        for z in np.arange(zc - ph / 2, zc + ph / 2 + 0.1, 2.0):
            sh -= box(xf + 0.4, xf + 1, min(y0, y1) + 1, max(y0, y1) - 1, z - 0.25, z + 0.25)
        raised.append(sh)
    zs = zc - ph / 2 - 1.5                              # sill with a 45 degree underside
    raised.append(prism_xz([(xf - 0.01, zs - 2.0), (xf + 2.0, zs), (xf + 2.0, zs + 0.01), (xf - 0.01, zs + 0.01)], yc - pw / 2 - 3, yc + pw / 2 + 3))
    return raised, pane


def make_house():
    z0 = FLOOR_Z
    # stage floor with a lip all round, chamfered underneath so it prints without support
    h = Manifold.batch_hull([box(-W / 2, W / 2, -PORCH, D, z0, STAGE), box(-W / 2 - 1, W / 2 + 1, -PORCH - 1, D + 1, z0 + 1, STAGE)])
    walls = box(-W / 2, -W / 2 + T, 0, D, STAGE, EAVE) + box(W / 2 - T, W / 2, 0, D, STAGE, EAVE)
    gable = [(-W / 2, EAVE - 0.01), (W / 2, EAVE - 0.01), (0, RIDGE)]
    front = box(-W / 2, W / 2, 0, T, STAGE, EAVE) + prism_xz(gable, 0, T)
    back = box(-W / 2, W / 2, D - T, D, STAGE, EAVE) + prism_xz(gable, D - T, D)
    # timber boarding
    cuts = board_grooves_x(W / 2, 1, 2.5, D - 1, STAGE, EAVE) + board_grooves_x(-W / 2, -1, 2.5, D - 1, STAGE, EAVE)
    cuts += board_grooves_y(D, 1, -W / 2 + 2.5, W / 2 - 1, STAGE, RIDGE) + board_grooves_y(0, -1, -W / 2 + 2.5, W / 2 - 1, EAVE, RIDGE)
    house = walls + front + back - Manifold.batch_boolean(cuts, OpType.Add)
    for cx in (-DOOR_X, DOOR_X):
        house -= prism_xz([(x, z + STAGE) for x, z in door_profile(cx)], -1, T + 1)
    house -= prism_xz(heart(0, EAVE + 10, 0.36), -1, T + 1)
    house -= Manifold.cylinder(T + 2, 4, 4, SEG).rotate((-90, 0, 0)).translate((0, D - T - 1, EAVE + 9))
    # door frames with keystones, standing proud of the front wall
    add = []
    for cx in (-DOOR_X, DOOR_X):
        add.append(prism_xz([(x, z + STAGE) for x, z in door_profile(cx, 2.0)], -1.0, 0.01) - prism_xz([(x, z + STAGE) for x, z in door_profile(cx)], -2, 1))
        top = STAGE + DOOR_TOP
        add.append(prism_xz([(cx - 1.8, top - 0.5), (cx + 1.8, top - 0.5), (cx + 2.4, top + 2.6), (cx - 2.4, top + 2.6)], -1.6, 0.01))
    # band at the eaves, front and back, 45 degree underside
    under_roof = prism_xz([(-W / 2 - 3, EAVE - 5), (W / 2 + 3, EAVE - 5), (W / 2 + 3, EAVE - 3 * TANP), (0, RIDGE), (-W / 2 - 3, EAVE - 3 * TANP)], -5, D + 5)
    add.append(prism_yz([(0.01, EAVE - 2.4), (-1.2, EAVE - 1.2), (-1.2, EAVE + 1.2), (0.01, EAVE + 1.2)], -W / 2, W / 2) ^ under_roof)
    add.append(prism_yz([(D - 0.01, EAVE - 2.4), (D + 1.2, EAVE - 1.2), (D + 1.2, EAVE + 1.2), (D - 0.01, EAVE + 1.2)], -W / 2, W / 2) ^ under_roof)
    # tongues along the gables' sloping tops: they key into grooves under the roof panels
    nx, nz = math.sin(math.radians(PITCH)), math.cos(math.radians(PITCH))
    a_, b_ = (TONGUE_IN, RIDGE - TONGUE_IN * TANP), (W / 2 - TONGUE_IN, EAVE + TONGUE_IN * TANP)
    tongue = [a_, b_, (b_[0] + TONGUE_H * nx, b_[1] + TONGUE_H * nz), (a_[0] + TONGUE_H * nx, a_[1] + TONGUE_H * nz)]
    for yc in (T / 2, D - T / 2):
        t_r = prism_xz(tongue, yc - TONGUE_W / 2, yc + TONGUE_W / 2); add += [t_r, t_r.mirror((1, 0, 0))]
    # round window frame at the back
    add.append(Manifold.cylinder(0.8, 5.6, 5.6, SEG).rotate((-90, 0, 0)).translate((0, D - 0.01, EAVE + 9)) - Manifold.cylinder(3, 4, 4, SEG).rotate((-90, 0, 0)).translate((0, D - 1, EAVE + 9)))
    raised, panes = side_window()
    raised_all = Manifold.batch_boolean(raised, OpType.Add); panes_all = Manifold.batch_boolean(panes, OpType.Add)
    house = house - panes_all - panes_all.mirror((1, 0, 0))
    add += [raised_all, raised_all.mirror((1, 0, 0))]
    h += house + Manifold.batch_boolean(add, OpType.Add)
    # porch: plank joints across the boards, and turned posts at the front corners only, so the figures can walk out freely
    h -= Manifold.batch_boolean([box(-W / 2 - 2, W / 2 + 2, y - 0.3, y + 0.3, STAGE - 0.5, STAGE + 0.1) for y in np.arange(-PORCH + 4, -0.5, 4.0)], OpType.Add)
    for x in (-W / 2 + 3, W / 2 - 3):
        y = -PORCH + 3
        h += box(x - 2, x + 2, y - 2, y + 2, STAGE, STAGE + 1.2) + cyl(x, y, STAGE, 7.5, 1.4, 32) + Manifold.cylinder(1.2, 1.4, 2.2, 32).translate((x, y, STAGE + 5.5)) + Manifold.sphere(2.2, 32).translate((x, y, STAGE + 7.2))
    # round hole for the bar's hub; the bar covers it at every angle
    h -= cyl(0, PIVOT_Y, z0 - 1, FLOOR_T + 2, HUB_R + 1.0)
    for x, y in PEGS:                                   # blind holes for the base pegs, 0.6 mm of floor left above
        h -= cyl(x, y, z0 - 1, 1 + 2.4, 2.1, 32)
    return h


# ---------------------------------------------------------------- roof
ROOF_T = 2.4


def make_roof_panel():
    """One roof panel, flat on the bed: x runs down the slope from the ridge (0) to the eave (S), y along the ridge."""
    run = (W / 2) / math.cos(math.radians(PITCH)); L, S = D + 12, run + 5
    p = box(0, S, 0, L, 0, ROOF_T)
    # fish-scale shingles: rows of half-round tile edges, staggered row to row
    arcs, pitch, rad = [], 4.4, 2.2
    for k, xb in enumerate(np.arange(S - 1.0, 4.0, -3.6)):
        for yc in np.arange(-pitch + (pitch / 2) * (k % 2), L + pitch, pitch):
            ring = CrossSection.circle(rad + 0.3, 24) - CrossSection.circle(rad - 0.3, 24)
            half = ring ^ CrossSection.square((rad + 1, 2 * rad + 2)).translate((0, -rad - 1))
            arcs.append(half.translate((xb - rad, yc)))
    scales = CrossSection.batch_boolean(arcs, OpType.Add) ^ CrossSection.square((S, L))
    p -= scales.extrude(1.0).translate((0, 0, ROOF_T - 0.5))
    # scalloped bargeboard edges at the gables and along the eave
    notches = [CrossSection.circle(1.1, 20).translate((x, y)) for x in np.arange(6.0, S - 2, 3.6) for y in (0.0, L)]
    notches += [CrossSection.circle(1.1, 20).translate((S, y)) for y in np.arange(1.8, L - 1, 3.6)]
    p -= CrossSection.batch_boolean(notches, OpType.Add).extrude(ROOF_T + 2).translate((0, 0, -1))
    # underneath: grooves that the gable tongues key into (the panel overhangs the front gable by 6 mm)
    for yc in (6 + T / 2, 6 + D - T / 2):
        p -= box(1.0, run - 1.0, yc - TONGUE_W / 2 - 0.15, yc + TONGUE_W / 2 + 0.15, -1, TONGUE_H + 0.2)
    # along the ridge edge: a bevel, so the two panels make a flat seat for the ridge cap
    sp, cp = math.sin(math.radians(PITCH)), math.cos(math.radians(PITCH))
    p -= prism_xz([(-1, (RIDGE_SEAT - sp) / cp), (4, (RIDGE_SEAT + 4 * sp) / cp), (4, 10), (-1, 10)], -1, L + 1)
    return p, S, L


FINIAL = 2.4                                    # half-height of the diamond finials


def make_ridge():
    """Ridge cap, flat side down, with a diamond finial at each gable. The diamonds' faces are all at 45 degrees
    to the ridge, so the roof prints standing on its front edge without supports."""
    L = D + 12
    r = (Manifold.cylinder(L, RIDGE_R, RIDGE_R, SEG).rotate((-90, 0, 0)) - box(-5, 5, -1, L + 1, -5, -1.0)).translate((0, 0, 1.0))
    zc = 1.0 + RIDGE_R                              # diamond centres on the top of the cap
    for y in (FINIAL, L - FINIAL):
        r += Manifold.batch_hull([box(-0.9, 0.9, y - 0.01, y + 0.01, zc - FINIAL, zc + FINIAL),
                                  box(-0.9, 0.9, y - FINIAL, y + FINIAL, zc - 0.01, zc + 0.01)])
    return r


def make_roof(panel, ridge):
    """The whole roof as one piece, in the house frame: both panels and the ridge cap, the cap sunk 0.1 mm into the
    panels' seat so they fuse. It rests on the gables, located by the tongues; nothing holds it down."""
    right = panel.translate((0, -6, 0)).rotate((0, PITCH, 0)).translate((0, 0, RIDGE))
    return right + right.mirror((1, 0, 0)) + ridge.translate((0, -6, RIDGE + RIDGE_SEAT - 0.1))


def roof_for_print(roof):
    """Stood on its front edge: the panels and cap run straight up, so nothing overhangs."""
    return to_bed(roof.rotate((-90, 0, 0)))


# ---------------------------------------------------------------- bar (in the bar frame: shaft at the origin, z=0 at the stage floor top)
PAD_R = 7.5
TURBINE_BASE_R, SOCKET_D, SOCKET_CLEAR = 5.5, 1.2, 0.2   # the turbine's base drops into a socket in the right-hand pad
TURBINE_FLAT = 4.3                              # the base and socket have a flat this far behind the centre, so the turbine can't turn
STATION_FOOT = (-8.4, 7.2, -4.6, 4.6)           # the power station's plinth in its own frame: u0, u1, v0, v1
STATION_RIM = 1.5                               # bar round the power station's pocket


def rounded_rect(u0, u1, v0, v1, z0, h, r):
    return Manifold.batch_hull([cyl(u, v, z0, h, r, 24) for u in (u0 + r, u1 - r) for v in (v0 + r, v1 - r)])


def make_bar():
    zb = BAR_Z
    bar = cyl(0, 0, zb, BAR_T, HUB_R + 3.5) + cyl(0, 0, zb - HUB_DROP, HUB_DROP + 0.01, HUB_R) + box(-ARM_R, ARM_R, -ARM_W / 2, ARM_W / 2, zb, zb + BAR_T)
    # each figure drops into a pocket in its end of the bar: a keyed round one for the turbine, an oblong one for the power station
    bar += cyl(ARM_R, 0, zb, BAR_T, PAD_R)
    u0, u1, v0, v1 = STATION_FOOT; e, c = STATION_RIM, SOCKET_CLEAR
    on_left = lambda m: m.rotate((0, 0, 90)).translate((-ARM_R, 0, 0))   # figure frame to the bar's left end, as in figure_on_bar
    bar += on_left(rounded_rect(u0 - e, u1 + e, v0 - e, v1 + e, zb, BAR_T, 2.0))
    bar -= on_left(rounded_rect(u0 - c, u1 + c, v0 - c, v1 + c, zb + BAR_T - SOCKET_D, SOCKET_D + 1, 0.6))
    bar -= cyl(ARM_R, 0, zb + BAR_T - SOCKET_D, SOCKET_D + 1, TURBINE_BASE_R + SOCKET_CLEAR) - box(ARM_R - 10, ARM_R - TURBINE_FLAT - SOCKET_CLEAR, -10, 10, zb, zb + BAR_T + 2)
    # shaft hole up the hub, the centre screw's hole, and a recess for its head so the servo's own short screw reaches
    top = zb - HUB_DROP + SHAFT_GRIP
    bar -= cyl(0, 0, zb - HUB_DROP - 1, SHAFT_GRIP + 1, SHAFT_R, 48) + cyl(0, 0, top - 0.5, BAR_T + 10, SCREW_R, 24) + cyl(0, 0, top + SCREW_LEFT, BAR_T + 10, SCREW_HEAD_R, 32)
    return bar


# ---------------------------------------------------------------- 3D figures
# Figure frame: u across (tangential), v outward along the arm (towards the door when the figure is out), w up from
# the top of the bar. Both sit in pockets in the ends of the bar, to be glued.
HUB_W, ROTOR_R, ROTOR_GAP = 14.0, 7.0, 0.4
ROTOR_HUB_L = 2.6                               # blade face to the back of the hub
NACELLE_R, NACELLE_FRONT = 1.6, 2.2


def turbine_body():
    """Wind turbine without its rotor: chamfered round base, tapered tower, nacelle with an axle hole. Prints upright."""
    t = Manifold.cylinder(1.6, TURBINE_BASE_R, 4.8, SEG) + Manifold.cylinder(HUB_W - 1.6, 1.5, 0.85, 40).translate((0, 0, 1.6))
    nac = Manifold.cylinder(NACELLE_FRONT + 2.8, NACELLE_R, NACELLE_R, 32).rotate((-90, 0, 0)).translate((0, -2.8, HUB_W))
    t += Manifold.batch_hull([nac, cyl(0, 0, HUB_W - NACELLE_R - 2.9, 0.1, 0.9, 24)])   # 45 degree underside to the tower
    t -= Manifold.cylinder(SPIGOT_L + 0.3, SPIGOT_R + SPIGOT_CLEAR, SPIGOT_R + SPIGOT_CLEAR, 24).rotate((-90, 0, 0)).translate((0, NACELLE_FRONT - SPIGOT_L - 0.2, HUB_W))   # socket for the rotor
    return t - box(-10, 10, -10, -TURBINE_FLAT, -1, 1.7)   # flat at the back of the base keys it into the socket


def turbine_rotor():
    """Three-bladed rotor, printed flat: blades on the bed, hub and spigot up. The spigot slots into the nacelle."""
    prof = [(1.0, 0.8), (3.0, 1.05), (5.0, 0.75), (ROTOR_R, 0.42)]   # (radius, half width); roots run into the hub
    blades = []
    for a in (90, 210, 330):
        c, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
        pts = [(r, w) for r, w in prof] + [(r, -w * 0.7) for r, w in reversed(prof)]
        blades.append(CrossSection([ccw([(x * c - y * s_, x * s_ + y * c) for x, y in pts])]))
    r = CrossSection.batch_boolean(blades, OpType.Add).extrude(0.9) + Manifold.cylinder(2.0, 2.0, 2.0, 40) + Manifold.cylinder(ROTOR_HUB_L - 2.0, 2.0, NACELLE_R, 40).translate((0, 0, 2.0))
    return r + Manifold.cylinder(SPIGOT_L, SPIGOT_R, SPIGOT_R, 24).translate((0, 0, ROTOR_HUB_L))


def rotor_in_figure(rotor):
    """Rotor from its print orientation into the figure frame: its axis along v, spigot back into the nacelle."""
    M = np.array([[1, 0, 0, 0], [0, 0, -1, NACELLE_FRONT + ROTOR_GAP + ROTOR_HUB_L], [0, 1, 0, HUB_W]], float)
    return rotor.transform(M)


def rotor_envelope():
    """The disc the rotor sweeps, for the clearance check, so it clears at any blade position."""
    return Manifold.cylinder(3.6, ROTOR_R + 0.3, ROTOR_R + 0.3, 48).rotate((-90, 0, 0)).translate((0, NACELLE_FRONT + ROTOR_GAP - 0.1, HUB_W))


def power_station():
    """Power station: turbine hall with a pitched roof, a cooling tower and a smoking chimney stack. Prints upright."""
    u0, u1, v0, v1 = STATION_FOOT
    ps = Manifold.batch_hull([box(u0, u1, v0, v1, 0, 1.0), box(u0 + 0.6, u1 - 0.6, v0 + 0.6, v1 - 0.6, 0, 1.6)])
    ps += box(-1.0, 6.5, -3.5, 2.5, 1.6, 8.5)                                        # turbine hall
    ps += prism_yz([(-3.8, 8.49), (2.8, 8.49), (-0.5, 10.6)], -1.3, 6.8)   # its roof, ridge along u
    for u in (0.4, 2.75, 5.1):                                                       # tall windows on the front
        ps -= box(u - 0.4, u + 0.4, 2.1, 3.0, 3.0, 7.0)
    w0, rw, c = 11.0, 2.5, 7.85                                                      # cooling tower: hyperbolic profile
    prof = [(rw * math.sqrt(1 + ((w - w0) / c) ** 2), w) for w in np.linspace(1.6, 15.0, 30)]
    tower = CrossSection([ccw([(0, 1.6)] + prof + [(0, 15.0)])]).revolve(48).translate((-3.6, -0.5, 0))
    ps += tower - cyl(-3.6, -0.5, 13.5, 3, 2.1, 40)
    ps += Manifold.cylinder(18.4, 1.25, 0.85, 32).translate((4.6, -1.8, 1.6)) + Manifold.cylinder(0.6, 0.85, 1.05, 32).translate((4.6, -1.8, 19.4))   # stack
    for (du, dw, r) in [(0.1, 21.0, 1.2), (1.0, 22.1, 1.5), (2.2, 23.0, 1.35)]:          # smoke, each puff no more than 45 degrees out from the last
        ps += Manifold.sphere(r, 28).translate((4.6 + du, -1.8, dw))
    return ps


def figure_on_bar(fig, side):
    """Stand a figure on an end pad of the bar: v along the arm, outward; a proper rotation, no mirroring."""
    s = 1 if side > 0 else -1
    return fig.rotate((0, 0, -90 * s)).translate((s * ARM_R, 0, BAR_Z + BAR_T - SOCKET_D))   # sat down in its pocket


def rotor_assembly(angle_deg, bar, right_fig, left_fig):
    """Bar and figures in the house frame for a servo angle (positive = anticlockwise from above)."""
    a = (bar + figure_on_bar(right_fig, +1) + figure_on_bar(left_fig, -1)).rotate((0, 0, angle_deg))
    return a.translate((0, PIVOT_Y, STAGE))


# ---------------------------------------------------------------- fit check of the whole assembly
def bought_parts():
    """Stand-ins for the parts that are not printed: the servo, the board and a USB-C plug."""
    x0, x1, y0, y1 = SERVO_BODY; tab_top = BASE_FLOOR + SERVO_TAB_TOP; shaft_top = tab_top + SHAFT_ABOVE_TAB
    servo = (box(x0, x1, PIVOT_Y + y0, PIVOT_Y + y1, BASE_FLOOR, tab_top + SERVO_ABOVE_TAB)
             + box(SERVO_TABS[0], SERVO_TABS[1], PIVOT_Y + y0, PIVOT_Y + y1, tab_top - SERVO_TAB_T, tab_top)
             + cyl(0, PIVOT_Y, tab_top + SERVO_ABOVE_TAB, shaft_top - tab_top - SERVO_ABOVE_TAB, SHAFT_R, 48))
    bc = BOARD_X0 + BOARD_W / 2
    board = box(bc - 9, bc + 9, D - T - 22.5, D - T, BOARD_Z, BOARD_Z + 1.0) + box(bc - 4.5, bc + 4.5, D - T - 7.3, D - T + 0.3, BOARD_Z + 1.0, BOARD_Z + 4.2)
    plug = box(bc - 6.0, bc + 6.0, D - T + 0.3, D + 20, USB_Z - 3.25, USB_Z + 3.25)
    return [('servo', servo), ('board', board), ('USB plug', plug)]


def fit_check(parts, touching):
    """parts: [(name, manifold in the house frame)]. Reports any pair that overlaps, and the gap at each
    joint that is meant to touch."""
    bad = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            v = (parts[i][1] ^ parts[j][1]).volume()
            if v > 0.05: bad.append((parts[i][0], parts[j][0], round(v, 2)))
    print('fit check: no two parts overlap' if not bad else f'fit check: OVERLAPS (mm3) {bad}')
    d = dict(parts)
    for a, b_, what in touching:
        print(f'  {what:44s} gap {d[a].min_gap(d[b_], 3.0):.2f} mm')
    return bad


# ---------------------------------------------------------------- export and checks
def write_stl(m, name):
    mesh = m.to_mesh(); v = np.asarray(mesh.vert_properties)[:, :3]; f = np.asarray(mesh.tri_verts)
    tri = v[f]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    with open(os.path.join(OUT, name), 'wb') as fh:
        fh.write(b'GridSeis weather house'.ljust(80, b' ')); fh.write(struct.pack('<I', len(f)))
        for nn, tt in zip(n.astype('<f4'), tri.astype('<f4')): fh.write(nn.tobytes() + tt.tobytes() + b'\x00\x00')
    bb = m.bounding_box(); print(f'  {name:22s} {len(f):6d} triangles  {bb[3]-bb[0]:6.1f} x {bb[4]-bb[1]:6.1f} x {bb[5]-bb[2]:6.1f} mm')


def to_bed(m):
    bb = m.bounding_box(); return m.translate((-(bb[0] + bb[3]) / 2, -(bb[1] + bb[4]) / 2, -bb[2]))


if __name__ == '__main__':
    base, house = make_base(), make_house()
    panel, S, L = make_roof_panel(); ridge = make_ridge(); bar = make_bar()
    t_body, t_rotor, station = turbine_body(), turbine_rotor(), power_station()
    turbine = t_body + rotor_in_figure(t_rotor)                 # as it looks
    turbine_env = t_body + rotor_envelope()                     # what the clearance check uses: the rotor at any angle

    # clearance: figures and bar against the house over the whole travel, in 1 degree steps
    walls = house - box(-W, W, -PORCH - 5, D + 5, FLOOR_Z - 1, STAGE + 0.01)
    for nm, part in [('base', base), ('house', house), ('walls', walls), ('bar', bar), ('turbine', t_body), ('rotor', t_rotor), ('station', station), ('roof', panel)]:
        assert part.volume() > 20, f'{nm} came out empty: check polygon orientation'
        print(f'  {nm:8s} volume {part.volume() / 1000:6.2f} cm3, genus {part.genus()}')
    worst = []
    for a in np.arange(-ANGLE_MAX, ANGLE_MAX + 0.1, 1.0):
        vol = (rotor_assembly(a, bar, turbine_env, station) ^ walls).volume(); worst.append((a, vol))
    hits = [(a, v) for a, v in worst if v > 0.01]
    print(f'clearance check: figures clear the house at every angle from {-ANGLE_MAX:g} to {ANGLE_MAX:g} degrees' if not hits else f'clearance check: COLLISION at {hits[:5]}')
    fig_top = max(figure_on_bar(f, sd).bounding_box()[5] for f, sd in [(turbine_env, 1), (station, -1)])
    print(f'figure tops {fig_top:.1f} mm above the stage floor, door arch tops at {DOOR_TOP:.1f}')
    assert fig_top < DOOR_TOP - 1, 'figures too tall for the doors: shrink the figures or raise DOOR_TOP'
    sx0, sx1, sy0, sy1 = SERVO_BODY
    assert PIVOT_Y + sy1 + CLEAR < D - T - BOARD_L or sx1 < BOARD_X0 - 1.6 or sx0 > BOARD_X0 + BOARD_W + 1.6, 'servo case hits the board'
    assert SERVO_TABS[1] + 0.5 < W / 2 - T and SERVO_TABS[0] - 0.5 > -W / 2 + T, 'servo tabs hit the side wall'
    assert BASE_FLOOR + SERVO_TAB_TOP + SERVO_ABOVE_TAB + 1 < FLOOR_Z, 'servo case hits the underside of the stage floor'
    assert BASE_FLOOR + SERVO_TAB_TOP + SERVO_ABOVE_TAB + 0.5 < STAGE + BAR_Z - HUB_DROP, 'bar hub hits the top of the servo case'
    for name, fig, side, ang in [('turbine (clean)', turbine_env, +1, -ANGLE_MAX), ('power station (dirty)', station, -1, ANGLE_MAX)]:
        bb = figure_on_bar(fig, side).rotate((0, 0, ang)).translate((0, PIVOT_Y, STAGE)).bounding_box()
        assert bb[4] < 0.5, f'{name} does not get fully out of its door'
        assert bb[1] > -PORCH, f'{name} goes past the edge of the porch'
        print(f'  {name} fully out: from y = {bb[4]:.1f} to {bb[1]:.1f} mm (front wall at 0, porch edge at {-PORCH:.0f}), x from {bb[0]:.1f} to {bb[3]:.1f}')
    roof = make_roof(panel, ridge)
    assert len(roof.decompose()) == 1, 'roof is not one piece: panels and ridge cap do not join'
    for dz in np.arange(0.1, 3.01, 0.1):            # lifting the roof straight off: the grooves slide off the tongues
        assert (roof.translate((0, 0, dz)) ^ house).volume() < 0.01, f'roof catches on the gables {dz:.1f} mm up'
    print('roof: one piece, lifts straight off the gables')
    ridge_a = ridge.translate((0, -6, RIDGE + RIDGE_SEAT - 0.1))
    placed = lambda m, sd: figure_on_bar(m, sd).translate((0, PIVOT_Y, STAGE))
    parts = [('base', base), ('house', house), ('roof', roof),
             ('bar', bar.translate((0, PIVOT_Y, STAGE))), ('turbine', placed(t_body, 1)), ('rotor', placed(rotor_in_figure(t_rotor), 1)),
             ('power station', placed(station, -1))] + bought_parts()
    bad = fit_check(parts, [('house', 'base', 'house floor on the base'), ('servo', 'base', 'servo on the base floor and posts'),
                            ('board', 'base', 'board on its support'), ('roof', 'house', 'roof on the gables'), ('bar', 'servo', 'bar on the shaft'),
                            ('turbine', 'bar', 'turbine in its socket'), ('rotor', 'turbine', 'rotor on the nacelle'),
                            ('power station', 'bar', 'power station on the bar'), ('servo', 'house', 'servo case below the stage floor'),
                            ('bar', 'house', 'bar hub in the floor hole')])
    assert not bad, 'parts overlap: see the fit check above'
    print('writing STL files to', OUT)
    for old in ('figure_chimney.stl', 'figure_turbine.stl', 'roof_panel.stl', 'ridge.stl'):
        if os.path.exists(os.path.join(OUT, old)): os.remove(os.path.join(OUT, old))
    write_stl(to_bed(base), 'base.stl'); write_stl(to_bed(house), 'house.stl'); write_stl(roof_for_print(roof), 'roof.stl')
    write_stl(to_bed(bar.mirror((0, 0, 1))), 'bar.stl')   # bar printed top face down: hub and shaft hole up
    write_stl(to_bed(t_body), 'turbine_tower.stl'); write_stl(to_bed(t_rotor), 'turbine_rotor.stl'); write_stl(to_bed(station), 'power_station.stl')

    print('roof bounding box', [round(v, 1) for v in roof.bounding_box()], 'ridge at', round(RIDGE, 1))
    import preview
    def figures_at(a):
        place = lambda m, sd: figure_on_bar(m, sd).rotate((0, 0, a)).translate((0, PIVOT_Y, STAGE))
        return [(bar.rotate((0, 0, a)).translate((0, PIVOT_Y, STAGE)), preview.TRIM), (place(turbine, 1), preview.TURB), (place(station, -1), preview.CHIM)]
    preview.render(base, house, roof, ridge_a, figures_at, ANGLE_MAX, IMG)
    preview.render_layout(OUT, IMG)
