#!/usr/bin/env python3
"""
gen_maps.py - Genera layouts aleatorios para recolectar datos de navegacion.

Por cada mapa produce:
  worlds/map_XXX.world     mundo de Gazebo (SDF): arena 10x10 m + obstaculos
  maps/map_XXX.pgm/.yaml   mapa de ocupacion para map_server de Nav2
  manifest.json            obstaculos, split (train/val/test) y pares (inicio, destino)

Convenio de ejes: la arena esta centrada en (0, 0) de Gazebo y el origen del
mapa Nav2 es (-5, -5), asi que pixel y mundo coinciden sin transformaciones.

Uso:
  pip install numpy scipy
  python3 gen_maps.py --n-maps 40 --episodes 20 --out dataset_maps
  # mundos especiales de test (mas densos / con pared):
  python3 gen_maps.py --n-maps 6 --obstacles 18 25 --prefix dense --split 0 0 1 --seed 100
  python3 gen_maps.py --n-maps 6 --wall --prefix wall --split 0 0 1 --seed 200
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

# ---------------- parametros geometricos ----------------
A = 10.0            # lado de la arena (m)
RES = 0.005          # resolucion del mapa (m/pixel)
N = int(round(A / RES))
WALL_T = 0.2        # grosor de las paredes del borde (m)
OBST_H = 0.5        # altura de los obstaculos (m); el LiDAR del TB3 esta a ~0.18 m
ROBOT_R = 0.22      # radio de seguridad del TurtleBot3 en Nav2 (m)
MIN_GAP = 0.8       # separacion minima entre obstaculos y paredes (m)
START_CLEAR = 0.45  # distancia minima a obstaculos de inicio/destino (m)
GOAL_DIST = (2.0, 8.0)

# Coordenadas del centro de cada pixel. Fila 0 = arriba (y maximo), como en un PGM.
xs = -A / 2 + (np.arange(N) + 0.5) * RES
ys = A / 2 - (np.arange(N) + 0.5) * RES
XX, YY = np.meshgrid(xs, ys)


# ---------------- geometria / rasterizado ----------------
def border_mask():
    lim = A / 2 - WALL_T
    return (np.abs(XX) > lim) | (np.abs(YY) > lim)


def raster(o):
    if o["type"] == "cyl":
        return (XX - o["x"]) ** 2 + (YY - o["y"]) ** 2 <= o["r"] ** 2
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    dx, dy = XX - o["x"], YY - o["y"]
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return (np.abs(lx) <= o["sx"] / 2) & (np.abs(ly) <= o["sy"] / 2)


def dist_to_occ(occ):
    """Distancia (m) de cada celda al obstaculo mas cercano."""
    return ndimage.distance_transform_edt(~occ) * RES


# ---------------- generacion del layout ----------------
def random_obstacle(rng):
    lim = A / 2 - 1.0
    x, y = (float(v) for v in rng.uniform(-lim, lim, 2))
    if rng.random() < 0.5:
        return dict(type="cyl", x=x, y=y, r=float(rng.uniform(0.15, 0.4)))
    return dict(type="box", x=x, y=y, yaw=float(rng.uniform(0, math.pi)),
                sx=float(rng.uniform(0.3, 1.0)), sy=float(rng.uniform(0.3, 1.0)))


def make_layout(rng, n_obs, wall):
    occ = border_mask()
    obstacles = []
    if wall:  # una pared larga en la zona central
        w = dict(type="box", x=float(rng.uniform(-1.5, 1.5)), y=float(rng.uniform(-1.5, 1.5)),
                 yaw=float(rng.uniform(0, math.pi)), sx=float(rng.uniform(3.0, 5.0)), sy=0.2)
        obstacles.append(w)
        occ |= raster(w)
    dist = dist_to_occ(occ)
    target = n_obs + len(obstacles)
    tries = 0
    while len(obstacles) < target and tries < 500:
        tries += 1
        o = random_obstacle(rng)
        m = raster(o)
        if not m.any() or (dist[m] < MIN_GAP).any():
            continue
        obstacles.append(o)
        occ |= m
        dist = dist_to_occ(occ)
    return obstacles, occ


def validate_and_sample(rng, occ, n_eps):
    """Comprueba que el espacio libre es conexo y muestrea pares (inicio, destino)."""
    dist = dist_to_occ(occ)
    nav = dist >= ROBOT_R + 0.05
    lab, k = ndimage.label(nav)
    if k == 0:
        return None
    sizes = ndimage.sum(nav, lab, range(1, k + 1))
    if sizes.max() < 0.95 * nav.sum():      # hay bolsas aisladas: layout rechazado
        return None
    main = lab == (1 + int(np.argmax(sizes)))
    cand = np.argwhere(main & (dist >= START_CLEAR))
    if len(cand) < 200:
        return None
    pairs = []
    for _ in range(20000):
        if len(pairs) >= n_eps:
            break
        a, b = cand[rng.integers(len(cand), size=2)]
        sx, sy = xs[a[1]], ys[a[0]]
        gx, gy = xs[b[1]], ys[b[0]]
        if GOAL_DIST[0] <= math.hypot(gx - sx, gy - sy) <= GOAL_DIST[1]:
            yaw = float(rng.uniform(-math.pi, math.pi))
            pairs.append({"start": [round(float(sx), 3), round(float(sy), 3), round(yaw, 3)],
                          "goal": [round(float(gx), 3), round(float(gy), 3)]})
    return pairs if len(pairs) == n_eps else None


# ---------------- salida: mapa Nav2 ----------------
def write_map(path_no_ext, occ):
    img = np.where(occ, 0, 254).astype(np.uint8)      # 0 = ocupado, 254 = libre
    with open(f"{path_no_ext}.pgm", "wb") as f:
        f.write(f"P5\n{N} {N}\n255\n".encode())
        f.write(img.tobytes())
    name = Path(path_no_ext).name
    with open(f"{path_no_ext}.yaml", "w") as f:
        f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {RES}\n"
                f"origin: [{-A / 2}, {-A / 2}, 0.0]\nnegate: 0\n"
                "occupied_thresh: 0.65\nfree_thresh: 0.196\n")


# ---------------- salida: mundo Gazebo (SDF) ----------------
def _model(name, x, y, yaw, geom, color):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {OBST_H / 2:.3f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="c"><geometry>{geom}</geometry></collision>
        <visual name="v"><geometry>{geom}</geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
      </link>
    </model>"""


def _box(sx, sy):
    return f"<box><size>{sx:.3f} {sy:.3f} {OBST_H}</size></box>"


def _cyl(r):
    return f"<cylinder><radius>{r:.3f}</radius><length>{OBST_H}</length></cylinder>"


GZ_PLUGINS = """
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>"""

CLASSIC_PLUGINS = """
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>10.0</update_rate>
    </plugin>"""


def write_world(path, obstacles, sim):
    parts = []
    wc = "0.35 0.35 0.4 1"
    oc = "0.85 0.45 0.1 1"
    e = A / 2 - WALL_T / 2
    for k, (x, y, sx, sy) in enumerate([(0, e, A, WALL_T), (0, -e, A, WALL_T),
                                        (e, 0, WALL_T, A), (-e, 0, WALL_T, A)]):
        parts.append(_model(f"wall_{k}", x, y, 0.0, _box(sx, sy), wc))
    for k, o in enumerate(obstacles):
        if o["type"] == "cyl":
            parts.append(_model(f"obs_{k:02d}", o["x"], o["y"], 0.0, _cyl(o["r"]), oc))
        else:
            parts.append(_model(f"obs_{k:02d}", o["x"], o["y"], o["yaw"], _box(o["sx"], o["sy"]), oc))
    plugins = GZ_PLUGINS if sim == "gz" else CLASSIC_PLUGINS
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">{plugins}
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <cast_shadows>false</cast_shadows>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>{''.join(parts)}
  </world>
</sdf>
"""
    Path(path).write_text(sdf)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-maps", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=20, help="pares inicio/destino por mapa")
    ap.add_argument("--obstacles", type=int, nargs=2, default=[8, 15], metavar=("MIN", "MAX"))
    ap.add_argument("--wall", action="store_true", help="anade una pared larga central")
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"), help="fracciones por mapa")
    ap.add_argument("--sim", choices=["classic", "gz"], default="classic",
                    help="classic = Gazebo Classic (Humble); gz = gz sim (Harmonic)")
    ap.add_argument("--prefix", default="map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dataset_maps")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "worlds").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(parents=True, exist_ok=True)

    n = args.n_maps
    n_tr, n_va = round(args.split[0] * n), round(args.split[1] * n)
    order = np.random.default_rng(args.seed).permutation(n)
    split_of = {}
    for rank, idx in enumerate(order):
        split_of[int(idx)] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")

    ext = "world" if args.sim == "classic" else "sdf"
    manifest = []
    for i in range(n):
        name = f"{args.prefix}_{i:03d}"
        for attempt in range(100):          # reintenta hasta tener un layout valido
            rng = np.random.default_rng([args.seed, i, attempt])
            n_obs = int(rng.integers(args.obstacles[0], args.obstacles[1] + 1))
            obstacles, occ = make_layout(rng, n_obs, args.wall)
            episodes = validate_and_sample(rng, occ, args.episodes)
            if episodes is not None:
                break
        else:
            raise RuntimeError(f"No se pudo generar un layout valido para {name}")

        write_map(str(out / "maps" / name), occ)
        write_world(out / "worlds" / f"{name}.{ext}", obstacles, args.sim)
        manifest.append({
            "name": name, "split": split_of[i],
            "world": f"worlds/{name}.{ext}", "map": f"maps/{name}.yaml",
            "obstacles": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}
                          for o in obstacles],
            "episodes": episodes,
        })
        print(f"{name}: {len(obstacles)} obstaculos, split={split_of[i]}, intentos={attempt + 1}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nListo: {n} mapas en {out}/ (train/val/test = "
          f"{sum(m['split'] == 'train' for m in manifest)}/"
          f"{sum(m['split'] == 'val' for m in manifest)}/"
          f"{sum(m['split'] == 'test' for m in manifest)})")


if __name__ == "__main__":
    main()
