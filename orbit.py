#!/usr/bin/env python3
"""orbit.py — гладкая суставная (Joint) интерполяция через записанные точки."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

AXES = {
    "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "+y": (0, 1, 0), "-y": (0, -1, 0),
    "+z": (0, 0, 1), "-z": (0, 0, -1),
}
WORLD_UP = np.array([0.0, 0.0, 1.0])
EULER = "xyz"


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("нулевой вектор")
    return v / n


def pose_rot(tcp_deg):
    return Rot.from_euler(EULER, tcp_deg[3:6], degrees=True)


def estimate_target(positions, axes):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(positions, axes):
        d = unit(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ p
    cond = np.linalg.cond(A)
    x = np.linalg.solve(A, b)
    dists = [
        np.linalg.norm((np.eye(3) - np.outer(unit(d), unit(d))) @ (x - p))
        for p, d in zip(positions, axes)
    ]
    return x, dists, cond


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def build_path(pA, pB, T, RA, RB, step_m, step_deg):
    """Безопасная интерполяция: XYZ линейно, ориентация по SLERP."""
    length = np.linalg.norm(pB - pA)
    rot_angle_deg = math.degrees((RA.inv() * RB).magnitude())
    n = max(2, math.ceil(max(length / step_m, rot_angle_deg / step_deg)))

    key_rots = Rot.from_quat(np.vstack([RA.as_quat(), RB.as_quat()]))
    slerp = Slerp([0.0, 1.0], key_rots)

    out = []
    for k in range(n + 1):
        t = k / n
        pos = (1.0 - t) * pA + t * pB
        R = slerp([t])[0]
        out.append((pos, R))

    return out


def build_full_path(points, T, rotations, step_m, step_deg):
    """P1 -> P2 -> P3 -> ...; XYZ линейно, ориентация по SLERP."""
    full = []
    labels = []

    for i in range(len(points) - 1):
        pA = np.asarray(points[i]["tcp"][:3], float)
        pB = np.asarray(points[i + 1]["tcp"][:3], float)

        part = build_path(
            pA, pB, T,
            rotations[i], rotations[i + 1],
            step_m, step_deg,
        )

        if i > 0:
            part = part[1:]

        full.extend(part)
        name = f"{points[i].get('name','?')}->{points[i + 1].get('name','?')}"
        labels.extend([name] * len(part))

    return full, labels


def to_tcp_poses(path):
    pos = np.array([p for p, _ in path])
    eul = np.array([R.as_euler(EULER) for _, R in path])
    eul = np.unwrap(eul, axis=0)
    return np.hstack([pos, eul])


def api_pose(pose):
    """RC10 API pose с углами Эйлера, нормализованными к [-pi, pi]."""
    x = np.asarray(pose, float).copy()
    x[3:6] = (x[3:6] + math.pi) % (2 * math.pi) - math.pi
    return x


def aim_error_deg(poses, T, cam_axis):
    a = unit(cam_axis)
    errs = []
    for x in poses:
        aim = Rot.from_euler(EULER, x[3:6]).apply(a)
        errs.append(math.degrees(math.acos(float(np.clip(aim @ unit(T - x[:3]), -1, 1)))))
    return max(errs)


def select_points(steps, src=None, dst=None):
    points = [
        s for s in steps
        if s.get("type") in ("joint", "linear") and "tcp" in s
    ]

    if not points:
        raise SystemExit("В JSON нет joint/linear точек")

    names = [p.get("name") for p in points]

    i0 = 0 if src is None else names.index(src)
    i1 = len(points) - 1 if dst is None else names.index(dst)

    if i0 > i1:
        raise SystemExit("--from должен идти раньше --to")

    return points[i0:i1 + 1]


def get_motion_interface(robot):
    """Обеспечивает обратную совместимость между версиями 1.2.x и 1.4.x."""
    return robot.motion if hasattr(robot, "motion") else robot


def print_diag(robot, prefix, i=None, pose=None, label=None):
    motion = get_motion_interface(robot)
    rt = robot._rtd_receiver.rt_data
    extra = "" if i is None else f" pose={i} {label or ''}"
    print(f"[DIAG] {prefix}{extra}")
    print(
        f"[DIAG] mode={motion.mode.get()} warning={motion.mode.check_warning_status()} "
        f"buffer={rt.buff_fill}/{rt.buff_sz} wp_cntr={rt.wp_cntr} "
        f"safety={rt.safety} state_flags={rt.state_flags}"
    )
    print("[DIAG] actual TCP xyz:", np.round(np.asarray(rt.act_tcp_x[:3], float), 4))
    if pose is not None:
        p_api = api_pose(pose)
        print("[DIAG] target TCP xyz:", np.round(np.asarray(pose[:3], float), 4))
        print("[DIAG] target RPY raw deg:", np.round(np.degrees(np.asarray(pose[3:6], float)), 2))
        print("[DIAG] target RPY API deg:", np.round(np.degrees(p_api[3:6]), 2))


def _ik_once(robot, pose, seed):
    """Одиночный расчёт оптимальной обратной кинематики."""
    motion = get_motion_interface(robot)
    try:
        sol = motion.kinematics.get_inverse(
            tcp_pose=list(api_pose(pose)),
            angle_pose=list(np.asarray(seed, float)),
            orientation_units="rad",
            get_all=False,
        )
        return sol, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def ik_precompute_joint_path(robot, poses, labels):
    """
    Рассчитывает траекторию в углах суставов (Joint space) для каждой точки.
    Гарантирует гладкость ветви решений и отсутствие резких прыжков моторов.
    """
    motion = get_motion_interface(robot)
    q = np.asarray(robot._rtd_receiver.rt_data.act_q, float)
    joint_path = []
    print(f"[IK] Расчёт Joint-траектории для {len(poses)} точек пути...")

    for i, pose in enumerate(poses):
        pose_api = api_pose(pose)
        try:
            sol = motion.kinematics.get_inverse(
                tcp_pose=list(pose_api),
                angle_pose=list(q),  # Прошлые углы передаются для выбора ближайшего непрерывного решения
                orientation_units="rad",
                get_all=False,
            )
        except Exception as exc:
            print(f"[IK] Ошибка вычисления на точке {i}/{len(poses)-1} ({labels[i]}): {exc}")
            print_diag(robot, "IK exception", i, pose, labels[i])
            return None

        if sol is None:
            print(f"[IK] Точка недостижима (IK вернул None) {i}/{len(poses)-1} ({labels[i]})")
            print_diag(robot, "IK returned None", i, pose, labels[i])
            return None

        q_next = np.asarray(sol, float)

        # Проверка отсутствия неестественных скачков осей (> 30 градусов за один шаг)
        if len(joint_path) > 0:
            jump_deg = np.max(np.abs(np.degrees(q_next - q)))
            if jump_deg > 30.0:
                print(f"[IK] ВНИМАНИЕ: обнаружен скачок кинематической ветки на {jump_deg:.1f}° в точке {i}")

        q = q_next
        joint_path.append(q)

        if (i + 1) % 25 == 0 or i == len(poses) - 1:
            print(f"[IK] {i+1}/{len(poses)} точек рассчитано ({labels[i]})")

    print("[IK] Все точки успешно преобразованы в непрерывные углы суставов!")
    return joint_path


def add_joint_checked(robot, joint_path, labels, i, speed, accel, blend):
    """Отправка суставной точки в контроллер."""
    motion = get_motion_interface(robot)
    try:
        motion.joint.add_new_waypoint(
            angle_pose=list(joint_path[i]),
            speed=speed,       # Скорость суставов (рад/с)
            accel=accel,       # Ускорение суставов (рад/с^2)
            blend=blend,       # Радиус сглаживания в метрах
            units="rad",
        )
    except Exception as exc:
        print(f"[STREAM] Ошибка отправки joint-точки {i}/{len(joint_path)-1} ({labels[i]}): {exc}")
        print_diag(robot, "waypoint add failure", i, label=labels[i])
        raise


def stream_joint_path(robot, joint_path, labels, speed, accel, blend):
    """
    Отправляет предварительно рассчитанную непрерывную траекторию через скользящее окно.
    Увеличенное окно и ненулевой blend исключают остановки на точках.
    """
    motion = get_motion_interface(robot)
    rt = robot._rtd_receiver.rt_data
    window = 15  # Буферное окно планировщика контроллера

    motion.mode.set("hold")
    i = 0

    # Предзаполнение буфера ядра контроллера
    while i < len(joint_path) and i < window:
        add_joint_checked(robot, joint_path, labels, i, speed, accel, blend)
        i += 1

    motion.mode.set("move")
    print(f"[EXEC] Движение запущено (всего точек: {len(joint_path)}, окно очереди: {window})")

    while i < len(joint_path):
        mode = motion.mode.get()

        if mode == "pause":
            print_diag(robot, "controller PAUSE", i, label=labels[i])
            raise RuntimeError(f"Контроллер перешёл в pause перед точкой {i}/{len(joint_path)-1}")

        if mode == "hold" and rt.buff_fill > 0:
            motion.mode.set("move")

        # Пополняем буфер при освобождении слотов
        if rt.buff_fill < window:
            add_joint_checked(robot, joint_path, labels, i, speed, accel, blend)
            i += 1
            if i % 25 == 0 or i == len(joint_path):
                print(f"[EXEC] Отправлено {i}/{len(joint_path)}, буфер: {rt.buff_fill}, участок: {labels[i-1]}")
        else:
            time.sleep(0.005)

    motion.wait_waypoint_completion(0)
    print("[EXEC] Траектория полностью выполнена!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path, help="Путь к JSON-файлу с точками")
    ap.add_argument("--from", dest="src", help="Начальная точка")
    ap.add_argument("--to", dest="dst", help="Конечная точка")
    ap.add_argument("--target", type=float, nargs=3, help="Координаты центра фокуса X Y Z")
    ap.add_argument("--cam-axis", choices=AXES, default="+z")
    ap.add_argument("--cam-up", choices=AXES)
    ap.add_argument("--ip", default="10.10.10.10")
    ap.add_argument("--speed", type=float, default=0.3, help="Суставная скорость (рад/с)")
    ap.add_argument("--accel", type=float, default=0.6, help="Суставное ускорение (рад/с^2)")
    ap.add_argument("--scale", type=float, default=0.4, help="Глобальный множитель скорости (0.0 - 1.0)")
    ap.add_argument("--step", type=float, default=0.03, help="Линейный шаг дискретизации (м)")
    ap.add_argument("--ang-step", type=float, default=3.0, help="Угловой шаг ориентации (град)")
    # Blend > 0 исключает микроостановки; должен составлять ~30-45% от шага сетки
    ap.add_argument("--blend", type=float, default=0.012, help="Радиус сглаживания траектории (м)")
    ap.add_argument("--dry-run", action="store_true", help="Только расчёт траектории без подключения к роботу")
    ap.add_argument("--yes", action="store_true", help="Не запрашивать подтверждение старта")
    args = ap.parse_args()

    # Защита: blend не может превышать половину шага между точками
    if args.blend >= (args.step / 2.0):
        adjusted_blend = args.step * 0.4
        print(f"[ВНИМАНИЕ] Blend ({args.blend} м) >= step/2 ({args.step/2} м). Автокоррекция blend: {adjusted_blend:.4f} м")
        args.blend = adjusted_blend

    cam_axis = np.asarray(AXES[args.cam_axis], float)
    if args.cam_up:
        cam_up = np.asarray(AXES[args.cam_up], float)
    else:
        cam_up = next(
            np.asarray(v, float)
            for v in AXES.values()
            if abs(np.dot(v, cam_axis)) < 0.5
        )

    steps = json.loads(args.file.read_text(encoding="utf-8"))["steps"]
    points = select_points(steps, args.src, args.dst)

    if len(points) < 2:
        raise SystemExit("Нужно минимум две точки")

    tcp = [np.asarray(p["tcp"], float) for p in points]
    P = np.array([x[:3] for x in tcp])
    Rrec = [pose_rot(x) for x in tcp]
    axes = [R.apply(cam_axis) for R in Rrec]

    if args.target:
        T = np.asarray(args.target, float)
        dists = [
            np.linalg.norm((np.eye(3) - np.outer(unit(a), unit(a))) @ (T - p))
            for p, a in zip(P, axes)
        ]
        cond = 0.0
    else:
        T, dists, cond = estimate_target(P, axes)

    print("Целевой объект (фокус камеры):", np.round(T, 3), "м")
    print("Маршрут точек:", " -> ".join(str(p.get("name", "?")) for p in points))

    for p, pos, ax, dist in zip(points, P, axes, dists):
        dev = math.degrees(math.acos(float(np.clip(unit(ax) @ unit(T - pos), -1, 1))))
        print(f"{p.get('name','?')}: расстояние от оптической оси {dist*100:.1f} см, отклонение {dev:.1f}°")

    path, labels = build_full_path(
        points, T, Rrec,
        args.step, args.ang_step,
    )

    poses = to_tcp_poses(path)
    seg = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    jump = np.degrees(np.abs(np.diff(poses[:, 3:6], axis=0))).max()
    rots = Rot.from_euler(EULER, poses[:, 3:6])
    real_jump = np.degrees((rots[:-1].inv() * rots[1:]).magnitude()).max()

    print(
        f"Геометрия: {len(poses)} точек, "
        f"шаг <= {seg.max()*100:.1f} см, "
        f"длина {seg.sum()*100:.0f} см, "
        f"макс. угловой шаг {real_jump:.1f}°"
    )
    print(f"Макс. ошибка прицела камеры к центру: {aim_error_deg(poses, T, cam_axis):.3f}°")

    if args.dry_run:
        print("[DRY-RUN] Расчёт геометрии завершён.")
        return

    from rc10_api.rc_api import RobotApi

    print(f"Подключение к роботу {args.ip}...")
    robot = RobotApi(ip=args.ip, read_only=False, timeout=5)
    motion = get_motion_interface(robot)

    try:
        motion.scale_setup.set(velocity=args.scale, acceleration=args.scale)
        robot.controller_state.set("run", await_sec=120)
        motion.mode.set("hold")
        print_diag(robot, "Контроллер готов к работе")

        # 1. Расчёт углов сочленений для всех точек маршрута
        joint_path = ik_precompute_joint_path(robot, poses, labels)
        if joint_path is None:
            print("ОШИБКА: Траектория содержит недостижимые участки. Робот не двигался.")
            return

        # 2. Плавный выход в первую точку
        first = points[0]
        cur = motion.joint.get_actual_position(units="deg")
        far = max(abs(a - b) for a, b in zip(cur, first["joints_deg"]))

        print(f"Выход на старт к {first.get('name','?')} (макс. отклонение звеньев {far:.1f}°)")

        if not args.yes and input("Траектория свободна? Запустить? [y/N] ").strip().lower() != "y":
            return

        motion.joint.add_new_waypoint(
            angle_pose=first["joints_deg"],
            speed=20,
            accel=30,
            units="deg",
        )
        motion.mode.set("move")
        motion.wait_waypoint_completion(0)

        # 3. Выполнение непрерывной суставной траектории
        stream_joint_path(
            robot=robot,
            joint_path=joint_path,
            labels=labels,
            speed=args.speed,
            accel=args.accel,
            blend=args.blend,
        )
        print("Готово. Съёмка завершена.")

    except KeyboardInterrupt:
        print("\nПрервано оператором (Ctrl+C).")

    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise

    finally:
        try:
            motion.mode.set("hold")
        except Exception:
            pass
        robot._shutdown_api_sockets()


if __name__ == "__main__":
    main()