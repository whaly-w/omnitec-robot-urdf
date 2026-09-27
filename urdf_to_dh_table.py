#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
urdf_to_dh_table.py
-------------------
Convert a URDF or XACRO robot description file into a
Denavit-Hartenberg (DH) parameter table and print it to stdout.

DH convention used (Standard / Craig):
  T_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)

Columns:
  a      - link length      (distance along x_{i-1} between z axes)
  alpha  - link twist       (angle  about x_{i-1} from z_{i-1} to z_i)
  d      - link offset      (distance along z_{i-1} from x_{i-1} to x_i)
  theta  - joint angle      (angle  about z_{i-1} from x_{i-1} to x_i)
           (zero-pose value; this is the joint variable for revolute joints)

Usage:
  python urdf_to_dh_table.py path/to/robot.urdf [--unit deg|rad]
  python urdf_to_dh_table.py path/to/robot.xacro [--unit rad]
"""

import argparse
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np

# Ensure stdout can handle Unicode on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==============================================================================
# Math helpers
# ==============================================================================

def _parse_vec(text, default=(0.0, 0.0, 0.0)):
    """Parse a whitespace-separated string of floats into a list."""
    if text is None:
        return list(default)
    return [float(x) for x in text.strip().split()]


def _rpy_to_rot(rpy):
    """Intrinsic RPY (roll-pitch-yaw) to 3x3 rotation matrix (Rz*Ry*Rx)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,             cp*sr,             cp*cr ],
    ], dtype=float)


def _make_T(xyz, rpy):
    """Build a 4x4 homogeneous transform from xyz translation and rpy angles."""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rot(rpy)
    T[:3, 3]  = xyz
    return T


def _unit(v):
    """Return unit vector; returns input unchanged if norm is negligible."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _signed_angle(v1, v2, axis):
    """
    Signed angle from v1 to v2 measured about *axis* (right-hand rule).
    v1 and v2 need not be unit vectors.
    """
    v1   = _unit(np.asarray(v1,   dtype=float))
    v2   = _unit(np.asarray(v2,   dtype=float))
    axis = _unit(np.asarray(axis, dtype=float))
    cross = np.cross(v1, v2)
    sin_a = float(np.dot(cross, axis))
    cos_a = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.atan2(sin_a, cos_a)


# ==============================================================================
# URDF / XACRO loading
# ==============================================================================

def _load_urdf_root(path: str) -> ET.Element:
    """
    Parse URDF or XACRO file and return the XML root element.
    XACRO files are processed with the `xacro` CLI tool first.
    """
    if path.lower().endswith(".xacro"):
        try:
            result = subprocess.run(
                ["xacro", path],
                capture_output=True, text=True, check=True
            )
            return ET.fromstring(result.stdout)
        except FileNotFoundError:
            sys.exit(
                "ERROR: The 'xacro' tool was not found on PATH.\n"
                "Install it with:  pip install xacro\n"
                "or (ROS):         sudo apt install ros-<distro>-xacro"
            )
        except subprocess.CalledProcessError as exc:
            sys.exit(f"ERROR: xacro failed:\n{exc.stderr}")
    else:
        try:
            return ET.parse(path).getroot()
        except ET.ParseError as exc:
            sys.exit(f"ERROR: failed to parse URDF file:\n{exc}")


# ==============================================================================
# Kinematic chain extraction
# ==============================================================================

def _parse_joints(root: ET.Element) -> dict:
    """Return {joint_name: joint_dict} for every joint in the URDF."""
    joints = {}
    for j in root.findall("joint"):
        name  = j.get("name")
        jtype = j.get("type", "fixed")
        pe    = j.find("parent")
        ce    = j.find("child")
        if pe is None or ce is None:
            continue

        orig = j.find("origin")
        xyz = _parse_vec(orig.get("xyz") if orig is not None else None)
        rpy = _parse_vec(orig.get("rpy") if orig is not None else None)

        ax_e = j.find("axis")
        axis = _parse_vec(
            ax_e.get("xyz") if ax_e is not None else None,
            default=(0.0, 0.0, 1.0),
        )

        joints[name] = dict(
            type=jtype, parent=pe.get("link"), child=ce.get("link"),
            xyz=xyz, rpy=rpy, axis=axis,
        )
    return joints


def _find_base_link(root: ET.Element, joints: dict) -> str:
    """Return the root link (not a child of any joint)."""
    all_links   = {lk.get("name") for lk in root.findall("link")}
    child_links = {jd["child"] for jd in joints.values()}
    roots = all_links - child_links
    if not roots:
        sys.exit("ERROR: Could not find a root link in the URDF.")
    return roots.pop()


def _longest_chain(joints: dict, base_link: str) -> list:
    """
    Return the longest path of joint names from base_link to a leaf.
    Fixed joints are included so that transforms accumulate correctly.
    Active joints are filtered later when computing DH rows.
    """
    parent_to_joints: dict = {}
    for jn, jd in joints.items():
        parent_to_joints.setdefault(jd["parent"], []).append(jn)

    @lru_cache(maxsize=None)
    def _walk(link: str) -> list:
        if link not in parent_to_joints:
            return []
        best: list = []
        for jn in parent_to_joints[link]:
            candidate = [jn] + _walk(joints[jn]["child"])
            if len(candidate) > len(best):
                best = candidate
        return best

    return _walk(base_link)


# ==============================================================================
# DH parameter extraction
# ==============================================================================

_ACTIVE = {"revolute", "continuous", "prismatic"}


def _dh_params(p_prev, z_prev, x_prev, p_curr, z_curr):
    """
    Compute (a, alpha, d, theta) in metres / radians given two consecutive
    joint frames expressed in the base frame.
    """
    z_prev = _unit(np.asarray(z_prev, float))
    z_curr = _unit(np.asarray(z_curr, float))
    x_prev = _unit(np.asarray(x_prev, float))
    p_prev = np.asarray(p_prev, float)
    p_curr = np.asarray(p_curr, float)

    # Common normal direction
    x_n      = np.cross(z_prev, z_curr)
    x_n_norm = float(np.linalg.norm(x_n))
    parallel  = x_n_norm < 1e-6

    if parallel:
        # Axes are parallel: common normal = perpendicular component of dp
        dp      = p_curr - p_prev
        d       = float(np.dot(dp, z_prev))
        dp_perp = dp - d * z_prev
        pn      = float(np.linalg.norm(dp_perp))
        x_n     = dp_perp / pn if pn > 1e-6 else x_prev
        a       = pn
        alpha   = 0.0
    else:
        x_n   = x_n / x_n_norm
        alpha = _signed_angle(z_prev, z_curr, x_n)

        # Closest points on the two z-axis lines
        dp    = p_curr - p_prev
        b     = float(np.dot(z_prev, z_curr))
        denom = 1.0 - b * b   # > 0 since lines not parallel

        t = float((np.dot(dp, z_prev) - b * np.dot(dp, z_curr)) / denom)
        s = float((b * np.dot(dp, z_prev) - np.dot(dp, z_curr)) / denom)

        q_prev = p_prev + t * z_prev
        q_curr = p_curr + s * z_curr

        a = float(np.dot(q_curr - q_prev, x_n))
        d = t   # distance along z_prev

    theta = _signed_angle(x_prev, x_n, z_prev)
    return a, alpha, d, theta


# ==============================================================================
# Main conversion
# ==============================================================================

def urdf_to_dh(path: str, unit: str = "deg") -> None:
    """Load URDF/XACRO, compute DH parameters, and print the table."""
    root   = _load_urdf_root(path)
    joints = _parse_joints(root)
    base   = _find_base_link(root, joints)
    chain  = _longest_chain(joints, base)

    if not chain:
        sys.exit("ERROR: No joints found in the kinematic chain.")

    # Propagate transforms and collect DH rows
    rows = []
    T_cum = np.eye(4)

    # Previous active-joint frame (starts at base = identity)
    pz = np.array([0., 0., 1.])   # z-axis
    px = np.array([1., 0., 0.])   # x-axis
    pp = np.zeros(3)               # origin

    for jname in chain:
        jd    = joints[jname]
        T_cum = T_cum @ _make_T(jd["xyz"], jd["rpy"])

        if jd["type"] in _ACTIVE:
            z_b = T_cum[:3, :3] @ np.array(jd["axis"], float)
            x_b = T_cum[:3, 0].copy()
            p_b = T_cum[:3, 3].copy()

            a, alpha, d, theta = _dh_params(pp, pz, px, p_b, z_b)

            rows.append(dict(
                name=jname, jtype=jd["type"],
                a=a, alpha=alpha, d=d, theta=theta,
            ))
            pz, px, pp = z_b, x_b, p_b

    if not rows:
        sys.exit("ERROR: No revolute/prismatic joints found in the chain.")

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    to_disp  = math.degrees if unit == "deg" else (lambda x: x)
    u_lbl    = "deg" if unit == "deg" else "rad"
    robot_nm = root.get("name", path)

    # Column widths
    w_name  = max((len(r["name"]) for r in rows), default=10) + 6
    w_name  = max(w_name, 14)

    HDR_A     = f"a (m)"
    HDR_ALPHA = f"alpha ({u_lbl})"
    HDR_D     = f"d (m)"
    HDR_THETA = f"theta ({u_lbl})"

    w_a     = max(len(HDR_A),     10)
    w_alpha = max(len(HDR_ALPHA), 14)
    w_d     = max(len(HDR_D),     10)
    w_theta = max(len(HDR_THETA), 14)

    SEP = "-" * (4 + w_name + 4 + w_a + 4 + w_alpha + 4 + w_d + 4 + w_theta + 2)

    def fmt_row(idx, name, a, alpha, d, theta, jtype):
        jvar = "theta" if jtype in ("revolute", "continuous") else "d"
        label = f"{idx}: {name} ({jvar})"
        return (
            f"| {label:<{w_name}}\t"
            f"| {a:>{w_a}.5f}\t"
            f"| {to_disp(alpha):>{w_alpha}.4f}\t"
            f"| {d:>{w_d}.5f}\t"
            f"| {to_disp(theta):>{w_theta}.4f} |"
        )

    def fmt_header():
        return (
            f"| {'Joint':<{w_name}}\t"
            f"| {HDR_A:>{w_a}}\t"
            f"| {HDR_ALPHA:>{w_alpha}}\t"
            f"| {HDR_D:>{w_d}}\t"
            f"| {HDR_THETA:>{w_theta}} |"
        )

    print(f"\nDH Parameter Table  --  {robot_nm}")
    print(f"Angle unit : {u_lbl}    |    Zero-pose (all joints = 0)")
    print(SEP)
    print(fmt_header())
    print(SEP)
    for i, r in enumerate(rows, 1):
        print(fmt_row(i, r["name"], r["a"], r["alpha"], r["d"], r["theta"], r["jtype"]))
    print(SEP)
    print()


# ==============================================================================
# CLI entry point
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="urdf_to_dh_table",
        description=(
            "Convert a URDF or XACRO robot description file to a\n"
            "Denavit-Hartenberg (DH) parameter table printed to stdout.\n\n"
            "Columns:\n"
            "  a      - link length  (distance along x_{i-1} between z axes)\n"
            "  alpha  - link twist   (angle about x_{i-1} from z_{i-1} to z_i)\n"
            "  d      - link offset  (distance along z_{i-1} from x_{i-1} to x_i)\n"
            "  theta  - joint angle  (zero-pose value; variable for revolute joints)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "urdf_path",
        metavar="URDF_PATH",
        help="Path to the .urdf or .xacro file.",
    )
    p.add_argument(
        "--unit",
        choices=["deg", "rad"],
        default="deg",
        help="Angular unit for alpha and theta columns (default: deg).",
    )
    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()
    urdf_to_dh(args.urdf_path, unit=args.unit)


if __name__ == "__main__":
    main()
