"""
Expert Policy: Pure Pursuit + Stanley Controller

This module provides a baseline expert policy for car racing using a combination
of Pure Pursuit and Stanley control methods.

Notes:
- Parameters (lookahead distances, gains, etc.) can be tuned to change driving behavior
- The noise model used during data collection lives in env/collect_data.py, not here
"""

import math

import numpy as np

from grand_prix.env.car_racing import TRACK_WIDTH


class PurePursuitStanley:
    def __init__(
        self,
        L=2.5,
        Ld_min=8.0,
        Ld_max=30.0,
        k_v=1.2,
        kappa_Ld_scale=4.0,
        k_stanley=1.6,
        delta_max=0.6,
        v_min=2.0,
        v_max=50.0,
        v_straight=20.0,
        v_curve_gain=6.0,
        a_y_max=9.0,
        straight_kappa=2e-3,
        straight_boost=6.0,
        kp_throttle=2.0,
        ki_throttle=0.15,
        throttle_band=1.0,
        road_margin=0.8,
        steer_smooth=0.3,
    ):
        self.L = L
        self.Ld_min, self.Ld_max, self.k_v, self.kappa_Ld_scale = (
            Ld_min,
            Ld_max,
            k_v,
            kappa_Ld_scale,
        )
        self.k_stanley, self.delta_max = k_stanley, delta_max
        self.v_min, self.v_max = v_min, v_max
        self.v_straight, self.v_curve_gain = v_straight, v_curve_gain
        self.a_y_max = a_y_max
        self.straight_kappa, self.straight_boost = straight_kappa, straight_boost
        self.kp_th, self.ki_th = kp_throttle, ki_throttle
        self.band = throttle_band
        self.road_margin = road_margin
        self.steer_smooth = float(np.clip(steer_smooth, 0.0, 0.95))
        self.th_i = 0.0
        self.delta_prev = 0.0

    @staticmethod
    def _wrap_pi(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _point_at_arclength_from_obs(self, obs, s_query):
        """Interpolate point on the *sampled* centerline using obs['path'] tables."""
        cl = obs["track"]["centerline"]
        mask = obs["track"]["mask"].astype(bool)
        n = int(mask.sum())
        if n < 2:
            return float(obs["car"]["position"][0]), float(obs["car"]["position"][1])

        pts = cl[:n]  # [N,2]
        seg_len = obs["path"]["seg_len"][:n]  # [N]
        s_arr = obs["path"]["cl_s"][: n + 1]  # [N+1]
        Ltot = float(obs["path"]["length"][0])

        s = float(s_query) % max(Ltot, 1e-6)
        i = int(np.searchsorted(s_arr, s, side="right") - 1)
        i = max(0, min(i, n - 1))
        i_next = (i + 1) % n
        s0 = float(s_arr[i])
        seg_i = float(seg_len[i])
        tau = 0.0 if seg_i < 1e-9 else (s - s0) / seg_i
        p = pts[i] + tau * (pts[i_next] - pts[i])
        return float(p[0]), float(p[1])

    def _v_ref_from_curvature(self, kappa):
        v1 = self.v_straight / (1.0 + self.v_curve_gain * abs(kappa))
        if abs(kappa) > 1e-6:
            v2 = math.sqrt(max(0.0, self.a_y_max / max(1e-6, abs(kappa))))
        else:
            v2 = self.v_max
        return min(v1, v2)

    def control(self, obs):
        """
        Args:
          obs: the environment observation dict (with 'path' features added).
        Returns:
          np.array([steer, gas, brake], dtype=np.float32)
        """
        ey = float(obs["path"]["ey"][0])
        epsi = float(obs["path"]["epsi"][0])
        s = float(obs["path"]["s"][0])
        kappa = float(obs["path"]["kappa"][0])
        v = float(obs["car"]["speed"][0])

        px, py = float(obs["car"]["position"][0]), float(obs["car"]["position"][1])
        yaw = float(obs["car"]["heading"][0])

        # --- adaptive lookahead ---
        Ld = self.Ld_min + self.k_v * v
        Ld = Ld / (1.0 + self.kappa_Ld_scale * abs(kappa))
        Ld = float(np.clip(Ld, self.Ld_min, self.Ld_max))

        # --- target point at s + Ld, interpolated from sampled centerline ---
        tx, ty = self._point_at_arclength_from_obs(obs, s + Ld)

        # --- transform target into vehicle frame ---
        dx, dy = tx - px, ty - py
        cy, sy = math.cos(yaw), math.sin(yaw)
        x_r = cy * dx + sy * dy
        y_r = -sy * dx + cy * dy

        # --- steering: pure pursuit + Stanley blend ---
        curv_pp = 2.0 * y_r / max(1e-6, Ld * Ld)
        delta_pp = math.atan(self.L * curv_pp)
        delta_st = epsi + math.atan2(self.k_stanley * ey, max(0.1, v))

        half_w = TRACK_WIDTH - self.road_margin
        w = float(np.clip(abs(ey) / max(1e-3, half_w), 0.0, 1.0))
        delta = (1.0 - w) * delta_pp + w * delta_st
        if abs(ey) > half_w:
            delta = delta_st

        delta = float(np.clip(delta, -self.delta_max, self.delta_max))
        if self.steer_smooth > 0:
            delta = (
                self.steer_smooth * self.delta_prev + (1.0 - self.steer_smooth) * delta
            )
        self.delta_prev = delta

        # --- speed plan & throttle/brake ---
        v_ref = self._v_ref_from_curvature(kappa)
        if abs(kappa) < 2e-3 and abs(ey) < 0.5 and abs(epsi) < 0.06:
            v_ref = min(v_ref + self.straight_boost, self.v_max)
        v_ref = float(np.clip(v_ref, self.v_min, self.v_max))

        dv = v_ref - v
        if dv > self.band:
            gas, brake = 1.0, 0.0
            self.th_i = 0.0
        elif dv < -self.band:
            gas, brake = 0.0, float(np.clip((-dv) / (self.band * 3.0), 0.0, 1.0))
            self.th_i = 0.0
        else:
            self.th_i = float(np.clip(self.th_i + self.ki_th * dv, -1.0, 1.0))
            a_cmd = self.kp_th * dv + self.th_i
            if a_cmd >= 0:
                gas, brake = float(np.clip(a_cmd / 2.0, 0.0, 1.0)), 0.0
            else:
                gas, brake = 0.0, float(np.clip(-a_cmd / 3.0, 0.0, 1.0))

        if abs(ey) > half_w * 1.2:
            brake = max(brake, 0.2)

        steer = float(np.clip(delta / self.delta_max, -1.0, 1.0))
        return np.array([steer, gas, brake], dtype=np.float32)