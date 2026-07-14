import csv
import cv2
import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from pathlib import Path

ROOT = Path(__file__).parent.parent
CALIB_CSV = ROOT / "camera_calibration/camera_calibration.csv"

VIDEOS = {
    "out2": ROOT / "videos/out2.mp4",
    "out4": ROOT / "videos/out4.mp4",
    "out13": ROOT / "videos/out13.mp4",
}

CAL_FRAMES = [0, 350, 0]

CAL_POINTS = {
    "out2": [
        {"world": [14, 2.4, 0], "image": [1815, 2054]},
        {"world": [8.2, 2.4, 0], "image": [3124, 1363]},
        {"world": [8.2, -2.4, 0], "image": [1958, 993]},
        {"world": [14, -2.4, 0], "image": [611, 1449]},
        {"world": [5.67, -7.5, 0], "image": [1495, 597]},
        {"world": [8.2, 1.8, 0], "image": [2959, 1301]},
        {"world": [8.2, -1.8, 0], "image": [2109, 1033]},
        {"world": [0, -1.8, 0], "image": [3269, 621]},
        {"world": [12.8, 0.9, 3.95], "image": [1705, 172]},
        {"world": [12.8, -0.9, 3.95], "image": [1214, 82]},
        {"world": [12.8, 0.295, 3.50], "image": [1524, 349]},
        {"world": [12.8, -0.295, 3.50], "image": [1382, 316]},
    ],
    "out4": [
        {"world": [-14, 2.4, 0], "image": [1560, 2064]},
        {"world": [-8.2, 2.4, 0], "image": [356, 1292]},
        {"world": [-8.2, -2.4, 0], "image": [1522, 1086]},
        {"world": [-14, -2.4, 0], "image": [2790, 1642]},
        {"world": [-5.67, -7.5, 0], "image": [2009, 782]},
        {"world": [-8.2, 1.8, 0], "image": [524, 1254]},
        {"world": [-8.2, -1.8, 0], "image": [1372, 1106]},
        {"world": [0, -1.8, 0], "image": [300, 606]},
        {"world": [-12.8, -0.9, 3.95], "image": [2367, 242]},
        {"world": [-12.8, 0.9, 3.95], "image": [1882, 253]},
        {"world": [-12.8, -0.295, 3.50], "image": [2177, 443]},
        {"world": [-12.8, 0.295, 3.50], "image": [2035, 452]},
    ],
    "out13": [
        {"world": [-14, -7.5, 0], "image": [153, 1456]},
        {"world": [14, -7.5, 0], "image": [3549, 1505]},
        {"world": [14, 7.5, 0], "image": [2849, 1013]},
        {"world": [-14, 7.5, 0], "image": [859, 982]},
        {"world": [0, -7.5, 0], "image": [1830, 1610]},
        {"world": [0, 7.5, 0], "image": [1848, 982]},
        {"world": [0, 1.8, 0], "image": [1843, 1122]},
        {"world": [0, -1.8, 0], "image": [1840, 1249]},
        {"world": [-12.8, -0.9, 3.95], "image": [556, 812]},
        {"world": [-12.8, -0.295, 3.50], "image": [594, 849]},
        {"world": [12.8, -0.9, 3.95], "image": [3156, 850]},
        {"world": [12.8, -0.295, 3.50], "image": [3115, 887]},
    ],
}



def normalize_points(points):
    """Hartley normalization: centroid at origin, mean distance sqrt(dim).

    points: (N, dim) array (dim=2 for image points, dim=3 for world points).
    Returns (normalized_points, T) where T is the (dim+1, dim+1) homogeneous
    transform such that normalized = (T @ [points | 1].T).T[:, :dim].
    """
    points = np.asarray(points, dtype=float)
    dim = points.shape[1]

    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_dist = np.mean(np.linalg.norm(centered, axis=1))
    scale = np.sqrt(dim) / mean_dist

    T = np.eye(dim + 1)
    T[:dim, :dim] *= scale
    T[:dim, dim] = -scale * centroid

    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    normalized = (T @ homogeneous.T).T[:, :dim]
    return normalized, T


def normalize_cal_points(cal_points):
    """Normalize world/image coords stored in a CAL_POINTS-style dict.

    Returns a dict keyed like cal_points, each value a dict with
    world/image normalized points plus their T_world/T_image transforms.
    """
    result = {}
    for key, entries in cal_points.items():
        world_pts = np.array([e["world"] for e in entries], dtype=float)
        image_pts = np.array([e["image"] for e in entries], dtype=float)

        world_norm, T_world = normalize_points(world_pts)
        image_norm, T_image = normalize_points(image_pts)

        result[key] = {
            "world": world_norm,
            "image": image_norm,
            "T_world": T_world,
            "T_image": T_image,
        }
    return result


def build_A(Xn, xn):
    Xn = np.asarray(Xn, float); xn = np.asarray(xn, float)
    N = len(Xn)
    Xt = np.hstack([Xn, np.ones((N, 1))])  # (N,4)
    u = xn[:, 0:1]; v = xn[:, 1:2]          # (N,1)
    Z = np.zeros((N, 4))
    A_u = np.hstack([Xt, Z, -u * Xt])       # (N,12) equazioni u
    A_v = np.hstack([Z, Xt, -v * Xt])       # (N,12) equazioni v
    # interleave così le righe di uno stesso punto restano adiacenti
    A = np.empty((2 * N, 12))
    A[0::2] = A_u; A[1::2] = A_v
    return A


def build_p(A):
    """
    Risolve A @ p = 0 per il vettore p (12,) e lo re-impacchetta in P (3x4).
    p = vettore singolare destro associato al minimo valore singolare di A,
        cioè l'ultima riga di Vt.  ||p|| = 1 (fissato dalla SVD).

    Ritorna:
      P    : (3,4) matrice di proiezione (nelle coordinate NORMALIZZATE)
      p    : (12,) il vettore soluzione
      cond : rapporto S[-2]/S[-1] (margine di unicità della soluzione nulla;
             alto = buono, vicino a 1 = configurazione degenere)
    """
    A = np.asarray(A, float)
    if A.shape[0] < 11:
        raise ValueError(f"Servono >=6 punti (>=12 righe): A ha {A.shape[0]} righe.")

    U, S, Vt = np.linalg.svd(A)
    p = Vt[-1]                    # (12,) minimo valore singolare
    P = p.reshape(3, 4)          # re-impacchetta PER RIGHE (coerente con build_A)

    cond = S[-2] / S[-1] if S[-1] > 0 else np.inf
    return P, p, cond


def denormalize_P(P_norm, T_world, T_image):
    """Riporta P dalle coordinate normalizzate a quelle originali.

    P_norm agisce su punti normalizzati: x_n ~ P_norm @ X_n, con
    x_n = T_image @ x e X_n = T_world @ X, quindi
    T_image @ x ~ P_norm @ T_world @ X  =>  x ~ T_image^-1 @ P_norm @ T_world @ X
    """
    P = np.linalg.inv(T_image) @ P_norm @ T_world
    P /= np.linalg.norm(P[2, :3])  # fissa la scala: ||riga3[:3]|| = 1
    return P


def decompose_P(P):
    """
    Decompone P (3x4) denormalizzata in K, R, t con P = K [R | t].
    Ritorna:
      K : (3,3) intrinseci (triangolare superiore, K[2,2]=1, diagonale > 0)
      R : (3,3) rotazione mondo->camera (ortogonale, det=+1)
      t : (3,)  traslazione
      C : (3,)  centro camera nel mondo (per la verifica di sanità)
    """
    P = np.asarray(P, float)
    M = P[:, :3]       # = K R
    p4 = P[:, 3]        # = K t

    # RQ: M = K R  (K triangolare superiore, R ortogonale)
    K, R = rq(M)

    # fix segni: diagonale di K positiva (D@D = I -> K R invariato)
    s = np.sign(np.diag(K)); s[s == 0] = 1
    D = np.diag(s)
    K = K @ D
    R = D @ R

    # normalizza scala: K[2,2] -> 1 (scala ANCHE p4 con lo stesso fattore)
    scale = K[2, 2]
    K = K / scale
    p4 = p4 / scale

    # t = inv(K) p4
    t = np.linalg.inv(K) @ p4

    # rotazione propria: det(R) = +1
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # centro camera nel mondo: C = -inv(M) p4_originale
    C = -np.linalg.inv(M) @ P[:, 3]

    return K, R, t, C


def pack(K, R, t, dist):
    """dist = [k1, k2, p1, p2]"""
    rvec = Rotation.from_matrix(R).as_rotvec()
    return np.concatenate([
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]],  # fx, fy, cx, cy, skew
        rvec, t, dist
    ])


def unpack(p):
    fx, fy, cx, cy, s = p[0:5]
    K = np.array([[fx, s, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    R = Rotation.from_rotvec(p[5:8]).as_matrix()
    t = p[8:11]
    dist = p[11:15]  # k1, k2, p1, p2
    return K, R, t, dist


def project(K, R, t, dist, X):
    """X: (N,3) mondo -> x: (N,2) pixel, con distorsione"""
    Xc = (R @ X.T + t[:, None]).T          # coord camera
    z = Xc[:, 2]
    xn = Xc[:, 0] / z                       # normalizzate
    yn = Xc[:, 1] / z
    k1, k2, p1, p2 = dist
    r2 = xn**2 + yn**2
    radial = 1 + k1*r2 + k2*r2**2
    xd = xn*radial + 2*p1*xn*yn + p2*(r2 + 2*xn**2)
    yd = yn*radial + p1*(r2 + 2*yn**2) + 2*p2*xn*yn
    u = K[0, 0]*xd + K[0, 1]*yd + K[0, 2]
    v = K[1, 1]*yd + K[1, 2]
    return np.stack([u, v], axis=1)


def reprojection_error(params, X, x_obs):
    K, R, t, dist = unpack(params)
    x_pred = project(K, R, t, dist, X)
    return (x_pred - x_obs).ravel()          # residui 2N


def refine_camera(K_init, R_init, t_init, X, x_obs,
                   dist_init=np.zeros(4), fix_principal_point=False):
    params0 = pack(K_init, R_init, t_init, dist_init)

    if fix_principal_point:
        cx0, cy0 = params0[2], params0[3]
        free = np.ones(len(params0), dtype=bool)
        free[2] = free[3] = False

        def resid(pf):
            p = params0.copy()
            p[free] = pf
            p[2], p[3] = cx0, cy0
            return reprojection_error(p, X, x_obs)

        res = least_squares(resid, params0[free], loss="huber", method="trf")
        params0[free] = res.x
        result_params = params0
    else:
        res = least_squares(reprojection_error, params0,
                             args=(X, x_obs), loss="huber", method="trf")
        result_params = res.x

    K, R, t, dist = unpack(result_params)
    rms = np.sqrt(np.mean(reprojection_error(result_params, X, x_obs)**2))
    return K, R, t, dist, rms


def main():
    normalized = normalize_cal_points(CAL_POINTS)

    cameras = {}
    points = {}
    for key, data in normalized.items():
        A = build_A(data["world"], data["image"])
        P_norm, p, cond = build_p(A)
        P = denormalize_P(P_norm, data["T_world"], data["T_image"])
        K, R, t, C = decompose_P(P)
        print(key, "cond:", cond)
        print("P:\n", P)
        print("K:\n", K)
        print("R:\n", R)
        print("t:", t)
        print("C:", C)

        cameras[key] = (K, R, t)
        entries = CAL_POINTS[key]
        points[key] = (
            np.array([e["world"] for e in entries], dtype=float),
            np.array([e["image"] for e in entries], dtype=float),
        )

    results = {}
    for name, (K, R, t) in cameras.items():
        X, x_obs = points[name]
        K, R, t, dist, rms = refine_camera(K, R, t, X, x_obs)
        print(f"{name}: RMS reproj = {rms:.3f} px, dist = {dist}")

        C = -R.T @ t
        P = K @ np.hstack([R, t[:, None]])
        results[name] = {"K": K, "R": R, "t": t, "C": C, "dist": dist, "P": P, "rms": rms}

    export_calibration_csv(results, CALIB_CSV)


def export_calibration_csv(results, path):
    """Scrive su CSV, una riga per camera, tutte le matrici utili per il
    tracking 3d: K, R, t, C (centro camera), dist, P (K [R|t]), rms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["camera"]
    fieldnames += [f"K{i}{j}" for i in range(3) for j in range(3)]
    fieldnames += [f"R{i}{j}" for i in range(3) for j in range(3)]
    fieldnames += [f"t{i}" for i in range(3)]
    fieldnames += [f"C{i}" for i in range(3)]
    fieldnames += ["k1", "k2", "p1", "p2"]
    fieldnames += [f"P{i}{j}" for i in range(3) for j in range(4)]
    fieldnames += ["rms"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, r in results.items():
            row = {"camera": name, "rms": r["rms"]}
            for i in range(3):
                for j in range(3):
                    row[f"K{i}{j}"] = r["K"][i, j]
                    row[f"R{i}{j}"] = r["R"][i, j]
            for i in range(3):
                row[f"t{i}"] = r["t"][i]
                row[f"C{i}"] = r["C"][i]
            for i in range(3):
                for j in range(4):
                    row[f"P{i}{j}"] = r["P"][i, j]
            row["k1"], row["k2"], row["p1"], row["p2"] = r["dist"]
            writer.writerow(row)


if __name__ == "__main__":
    main()