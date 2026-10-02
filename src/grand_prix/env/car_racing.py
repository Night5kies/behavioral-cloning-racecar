import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D import b2ContactListener, b2Contact, b2FixtureDef, b2PolygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 2.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 1.5  # Camera zoom

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)

MAX_TRACK_POINTS = 1024
RAY_COUNT = 32
RAY_MAX_DISTANCE = PLAYFIELD * 3.0


class FrictionDetector(b2ContactListener):
    def __init__(self, env, lap_complete_percent):
        super().__init__()
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact: b2Contact):
        self.on_contact(contact, begin=True)

    def EndContact(self, contact: b2Contact):
        self.on_contact(contact, begin=False)

    def on_contact(self, contact: b2Contact, begin: bool):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return

        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    """
    A top-down racing environment (modified to include path-relative features
    directly in the observation so controllers can be fully observation-based).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
        camera_center: str | None = None,
        include_pixels: bool = False,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
            camera_center,
            include_pixels,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: pygame.Surface | None = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Car | None = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        self.last_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # steer, gas, brake
        self.ray_angles = np.linspace(-math.pi, math.pi, RAY_COUNT, endpoint=False).astype(
            np.float32
        )

        self.camera_center = camera_center or "follow_car"
        assert self.camera_center in ("follow_car", "track_centroid"), "camera_center must be either 'follow_car' or 'track_centroid'"
        self.include_pixels = include_pixels

        # Action space
        if self.continuous:
            # steer, gas, brake
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )
        else:
            # do nothing, right, left, gas, brake
            self.action_space = spaces.Discrete(5)

        # Observation space
        obs_dict = {
            "track": spaces.Dict(
                {
                    "centerline": spaces.Box(
                        low=-PLAYFIELD, high=PLAYFIELD, shape=(MAX_TRACK_POINTS, 2), dtype=np.float32
                    ),
                    "mask": spaces.MultiBinary(MAX_TRACK_POINTS),
                    "centroid": spaces.Box(
                        low=-PLAYFIELD, high=PLAYFIELD, shape=(2,), dtype=np.float32
                    ),
                    "aabb": spaces.Box(
                        low=-PLAYFIELD, high=PLAYFIELD, shape=(4,), dtype=np.float32
                    ),
                }
            ),
            "rays": spaces.Box(
                low=0.0, high=RAY_MAX_DISTANCE, shape=(RAY_COUNT,), dtype=np.float32
            ),
            "car": spaces.Dict(
                {
                    "position": spaces.Box(
                        low=-PLAYFIELD, high=PLAYFIELD, shape=(2,), dtype=np.float32
                    ),
                    "heading": spaces.Box(
                        low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32
                    ),
                    "linear_velocity": spaces.Box(
                        low=-1e3, high=1e3, shape=(2,), dtype=np.float32
                    ),
                    "speed": spaces.Box(low=0.0, high=1e3, shape=(1,), dtype=np.float32),
                    "angular_velocity": spaces.Box(
                        low=-1e3, high=1e3, shape=(1,), dtype=np.float32
                    ),
                    "steering_angle": spaces.Box(
                        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            ),
            "controls": spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([+1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            # --- NEW: path-relative features in obs ---
            "path": spaces.Dict(
                {
                    "ey": spaces.Box(-PLAYFIELD, PLAYFIELD, shape=(1,), dtype=np.float32),
                    "epsi": spaces.Box(-math.pi, math.pi, shape=(1,), dtype=np.float32),
                    "s": spaces.Box(0.0, PLAYFIELD * 100.0, shape=(1,), dtype=np.float32),
                    "kappa": spaces.Box(-1e2, 1e2, shape=(1,), dtype=np.float32),
                    "idx": spaces.Box(low=0, high=MAX_TRACK_POINTS - 1, shape=(1,), dtype=np.int32),
                    "t": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                    "ref_xy": spaces.Box(-PLAYFIELD, PLAYFIELD, shape=(2,), dtype=np.float32),
                    "ref_yaw": spaces.Box(-math.pi, math.pi, shape=(1,), dtype=np.float32),
                    "v_long": spaces.Box(-1e3, 1e3, shape=(1,), dtype=np.float32),
                    "v_lat": spaces.Box(-1e3, 1e3, shape=(1,), dtype=np.float32),
                    "length": spaces.Box(0.0, PLAYFIELD * 100.0, shape=(1,), dtype=np.float32),
                    # sampled-centerline derived arrays; aligned with track.centerline/mask
                    "cl_s": spaces.Box(
                        0.0, PLAYFIELD * 100.0, shape=(MAX_TRACK_POINTS + 1,), dtype=np.float32
                    ),
                    "seg_len": spaces.Box(
                        0.0, PLAYFIELD * 10.0, shape=(MAX_TRACK_POINTS,), dtype=np.float32
                    ),
                    "cl_yaw": spaces.Box(
                        -math.pi, math.pi, shape=(MAX_TRACK_POINTS,), dtype=np.float32
                    ),
                }
            ),
        }
        if self.include_pixels:
            obs_dict["pixels"] = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )

        self.observation_space = spaces.Dict(obs_dict)

        self.render_mode = render_mode
        if self.render_mode == "state_pixels" and not self.include_pixels:
            gym.logger.warn(
                "include_pixels=False but render_mode='state_pixels'; switching render_mode to 'human'."
            )
            self.render_mode = "human"

    # ---------------------------------------------------------------------
    # Lifecycle helpers
    # ---------------------------------------------------------------------
    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert self.domain_randomize, "domain_randomize must be True to use this function."
        if randomize:
            self.road_color = self.np_random.uniform(0, 210, size=3)
            self.bg_color = self.np_random.uniform(0, 210, size=3)
            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    # ---------------------------------------------------------------------
    # Track generation
    # ---------------------------------------------------------------------
    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        self.road_poly = []
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track

        if len(self.track) > 0:
            arr = np.asarray(self.track, dtype=np.float32)
            self.track_centroid = (
                float(np.mean(arr[:, 2])),
                float(np.mean(arr[:, 3])),
            )
        else:
            self.track_centroid = (0.0, 0.0)

        xs, ys = [], []
        for poly, _ in self.road_poly:
            for (x, y) in poly:
                xs.append(x)
                ys.append(y)

        if xs and ys:
            self._track_aabb = (min(xs), max(xs), min(ys), max(ys))
        else:
            self._track_aabb = (-PLAYFIELD, -PLAYFIELD, PLAYFIELD, PLAYFIELD)

        # compute outer boundary segments for raycasts
        def q(p):
            return (round(p[0], 3), round(p[1], 3))

        edge_map = {}
        for poly, _ in self.road_poly:
            pts = [tuple(poly[i]) for i in range(4)]
            edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]
            for (a, b) in edges:
                ka, kb = q(a), q(b)
                key = (ka, kb) if ka <= kb else (kb, ka)
                edge_map.setdefault(key, []).append((a, b))

        self._boundary_segments = []
        for _, segs in edge_map.items():
            if len(segs) == 1:
                self._boundary_segments.append(segs[0])

        return True

    # ---------------------------------------------------------------------
    # Reset / Step
    # ---------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(  # type: ignore[unknown-assignment]
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround  # type: ignore[unknown-access]
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.last_action[:] = 0.0

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict) and "randomize" in options:
                randomize = options["randomize"]
            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()

        return self._make_obs(), {}

    # ---------------------------------------------------------------------
    # Observation construction
    # ---------------------------------------------------------------------
    def _compute_track_repr(self):
        """Centerline sample (fixed size) + mask, centroid, aabb."""
        n = len(self.track)
        centerline = np.zeros((MAX_TRACK_POINTS, 2), dtype=np.float32)
        mask = np.zeros((MAX_TRACK_POINTS,), dtype=np.uint8)
        if n > 0:
            idx = np.linspace(0, n - 1, min(n, MAX_TRACK_POINTS)).astype(int)
            pts = np.stack(
                [np.array([self.track[j][2], self.track[j][3]], dtype=np.float32) for j in idx]
            )
            centerline[: len(pts)] = pts
            mask[: len(pts)] = 1
        centroid = np.array(self.track_centroid, dtype=np.float32)
        aabb = np.array(self._track_aabb, dtype=np.float32)
        return centerline, mask, centroid, aabb

    def _derive_from_centerline(self, cl_xy_valid: np.ndarray):
        """
        Given valid Nx2 sampled centerline (loop), return:
        seg_len [N], cl_s [N+1], cl_yaw [N], tan [N,2], nrm [N,2]
        """
        N = int(cl_xy_valid.shape[0])
        pts = cl_xy_valid
        pts_next = np.roll(pts, -1, axis=0)
        seg_vec = pts_next - pts  # [N,2]
        seg_len = np.linalg.norm(seg_vec, axis=1)  # [N]
        seg_len = np.maximum(seg_len, 1e-6).astype(np.float32)
        tan = (seg_vec / seg_len[:, None]).astype(np.float32)
        nrm = np.stack((-tan[:, 1], tan[:, 0]), axis=1)  # left normal
        cl_yaw = np.arctan2(tan[:, 1], tan[:, 0]).astype(np.float32)
        cl_s = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)  # [N+1]
        return seg_len, cl_s, cl_yaw, tan, nrm

    def _closest_on_polyline_loop(self, p: np.ndarray, pts: np.ndarray):
        """
        Project point p onto each segment of loop pts; return (i, t, ref_xy).
        i: segment index, t in [0,1] along segment i, ref_xy: closest point.
        """
        pts_next = np.roll(pts, -1, axis=0)
        d = pts_next - pts
        d2 = (d**2).sum(axis=1)
        t = np.clip(((p - pts) * d).sum(axis=1) / np.maximum(d2, 1e-12), 0.0, 1.0)  # [N]
        proj = pts + (d * t[:, None])
        err = p - proj
        dist2 = (err**2).sum(axis=1)
        i = int(dist2.argmin())
        ti = float(t[i])
        ref_xy = (float(proj[i, 0]), float(proj[i, 1]))
        return i, ti, ref_xy

    def _make_obs(self):
        # track rep
        assert self.car is not None, "car must be initialized to make observation"
        centerline, mask, centroid, aabb = self._compute_track_repr()

        # car kinematics
        x, y = float(self.car.hull.position[0]), float(self.car.hull.position[1])
        vx, vy = float(self.car.hull.linearVelocity[0]), float(self.car.hull.linearVelocity[1])
        speed = math.sqrt(vx * vx + vy * vy)
        heading = float(self.car.hull.angle)
        ang_vel = float(self.car.hull.angularVelocity)
        steer_angle = float(self.car.wheels[0].joint.angle) if self.car.wheels else 0.0

        # rays
        rays = self._raycast_distances((x, y), heading).astype(np.float32)

        # ---- path-relative features computed from the *sampled* centerline ----
        n_valid = int(mask.astype(np.uint8).sum())
        path = {
            "ey": np.array([0.0], dtype=np.float32),
            "epsi": np.array([0.0], dtype=np.float32),
            "s": np.array([0.0], dtype=np.float32),
            "kappa": np.array([0.0], dtype=np.float32),
            "idx": np.array([0], dtype=np.int32),
            "t": np.array([0.0], dtype=np.float32),
            "ref_xy": np.zeros((2,), dtype=np.float32),
            "ref_yaw": np.array([0.0], dtype=np.float32),
            "v_long": np.array([0.0], dtype=np.float32),
            "v_lat": np.array([0.0], dtype=np.float32),
            "length": np.array([0.0], dtype=np.float32),
            "cl_s": np.zeros((MAX_TRACK_POINTS + 1,), dtype=np.float32),
            "seg_len": np.zeros((MAX_TRACK_POINTS,), dtype=np.float32),
            "cl_yaw": np.zeros((MAX_TRACK_POINTS,), dtype=np.float32),
        }

        if n_valid >= 2:
            cl_xy_valid = centerline[:n_valid]
            seg_len, cl_s, cl_yaw, tan, nrm = self._derive_from_centerline(cl_xy_valid)
            Ltot = float(cl_s[-1])

            p = np.array([x, y], dtype=np.float32)
            i, ti, ref_xy = self._closest_on_polyline_loop(p, cl_xy_valid)
            ref_yaw = float(cl_yaw[i])

            # signed cross-track error (+ left of tangent)
            ey = float(np.dot(p - np.array(ref_xy, dtype=np.float32), nrm[i]))
            epsi = self._wrap_to_pi(heading - ref_yaw)

            # along-track arclength s (wrap)
            s = float(cl_s[i] + ti * float(seg_len[i]))
            if s >= Ltot:
                s -= Ltot

            # curvature at i via central difference on yaw over arc length
            i_prev = (i - 1) % n_valid
            i_next = (i + 1) % n_valid
            dtheta = self._wrap_to_pi(float(cl_yaw[i_next] - cl_yaw[i_prev]))
            dl = 0.5 * (float(seg_len[i_prev]) + float(seg_len[i])) + 1e-6
            kappa = dtheta / dl

            vxy = np.array([vx, vy], dtype=np.float32)
            v_long = float(np.dot(vxy, tan[i]))
            v_lat = float(np.dot(vxy, nrm[i]))

            # pack scalars
            path.update(
                {
                    "ey": np.array([ey], dtype=np.float32),
                    "epsi": np.array([epsi], dtype=np.float32),
                    "s": np.array([s], dtype=np.float32),
                    "kappa": np.array([float(kappa)], dtype=np.float32),
                    "idx": np.array([i], dtype=np.int32),
                    "t": np.array([ti], dtype=np.float32),
                    "ref_xy": np.array(ref_xy, dtype=np.float32),
                    "ref_yaw": np.array([ref_yaw], dtype=np.float32),
                    "v_long": np.array([v_long], dtype=np.float32),
                    "v_lat": np.array([v_lat], dtype=np.float32),
                    "length": np.array([Ltot], dtype=np.float32),
                }
            )
            # padded arrays
            path["cl_s"][: n_valid + 1] = cl_s
            path["seg_len"][:n_valid] = seg_len
            path["cl_yaw"][:n_valid] = cl_yaw

        obs = {
            "track": {
                "centerline": centerline,
                "mask": mask,
                "centroid": np.array(centroid, dtype=np.float32),
                "aabb": np.array(aabb, dtype=np.float32),
            },
            "rays": rays,
            "car": {
                "position": np.array([x, y], dtype=np.float32),
                "heading": np.array([heading], dtype=np.float32),
                "linear_velocity": np.array([vx, vy], dtype=np.float32),
                "speed": np.array([speed], dtype=np.float32),
                "angular_velocity": np.array([ang_vel], dtype=np.float32),
                "steering_angle": np.array([steer_angle], dtype=np.float32),
            },
            "controls": self.last_action.copy(),
            "path": path,
        }

        if self.include_pixels:
            obs["pixels"] = self._render("state_pixels")  # same pixels as before

        return obs

    # ---------------------------------------------------------------------
    # Rendering helpers
    # ---------------------------------------------------------------------
    def _world_to_screen_pt(self, pt, zoom, translation, angle):
        """pt=(x,y) in world; return (X,Y) in screen pixels (pre-flip)."""
        v = pygame.math.Vector2(pt).rotate_rad(angle)
        return (v[0] * zoom + translation[0], v[1] * zoom + translation[1])

    def _render_centerline(self, zoom, translation, angle, color=(255, 255, 0)):
        """Draw the full centerline as an anti-aliased loop."""
        # Build cached arrays if needed
        self._ensure_centerline_cache()
        if not hasattr(self, "_cl_xy") or self._cl_xy is None or len(self._cl_xy) < 2:
            return
        pts = self._cl_xy  # shape [N,2] in world coords
        step = max(1, len(pts) // 600)
        scr_pts = [
            self._world_to_screen_pt((float(p[0]), float(p[1])), zoom, translation, angle)
            for p in pts[::step]
        ]
        if len(scr_pts) >= 2:
            pygame.draw.aalines(self.surf, color, True, scr_pts)

    def step(self, action: np.ndarray | int | None):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)  # type: ignore[attr-defined]
                self.last_action = np.array([action[0], action[1], action[2]], dtype=np.float32)
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        obs = self._make_obs()

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Termination due to finishing lap
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                step_reward = -100

        if self.render_mode == "human":
            self.render()

        return obs, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        if self.camera_center == "follow_car":
            scroll_x = -(self.car.hull.position[0]) * zoom
            scroll_y = -(self.car.hull.position[1]) * zoom
        else:
            angle = 0.0
            zoom = self._fit_zoom_for_angle(angle, mode)
            scroll_x = -self.track_centroid[0] * zoom
            scroll_y = -self.track_centroid[1] * zoom

        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        vw, vh, info_bar_px = self._view_dims_px(mode)
        base_cx = WINDOW_W / 2.0
        base_cy = info_bar_px / 2.0 + vh / 2.0
        trans = (base_cx + trans[0], base_cy + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        self._render_centerline(zoom, trans, angle)

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, int(WINDOW_H - WINDOW_H * 2.5 / 40.0))
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    # ---------------------------------------------------------------------
    # Geometry & indicators
    # ---------------------------------------------------------------------
    def _segment_ray_intersect(self, o, d, p, q):
        """Return t >= 0 where ray o + t d hits segment [p,q], or None if no hit."""
        # 2D cross
        def cross(a, b):
            return a[0] * b[1] - a[1] * b[0]

        r = (q[0] - p[0], q[1] - p[1])
        denom = cross(d, r)
        if abs(denom) < 1e-9:
            return None  # parallel
        op = (p[0] - o[0], p[1] - o[1])
        t = cross(op, r) / denom  # along ray
        u = cross(op, d) / denom  # along segment
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def _raycast_distances(self, origin, heading):
        """Ray distances to the OUTER boundary only (not internal tile edges)."""
        if not getattr(self, "_boundary_segments", None):
            return np.full((RAY_COUNT,), RAY_MAX_DISTANCE, dtype=np.float32)
        out = np.full((RAY_COUNT,), RAY_MAX_DISTANCE, dtype=np.float32)
        for i, rel in enumerate(self.ray_angles):
            ang = heading + rel
            d = (math.cos(ang), math.sin(ang))
            best = RAY_MAX_DISTANCE
            for (p, q) in self._boundary_segments:
                t = self._segment_ray_intersect(origin, d, p, q)
                if t is not None:
                    dist = float(t)
                    if dist < best:
                        best = dist
            out[i] = best
        return out

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity, horiz_ind(30, -0.8 * self.car.hull.angularVelocity), (255, 0, 0)
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))

    def _view_dims_px(self, mode: str) -> tuple[float, float, float]:
        """Returns usable (width_px, height_px, info_bar_px) for the drawing area."""
        info_bar_px = 0.0
        if mode == "human":
            # Indicators occupy ~5 * (H/40) = H/8 at the bottom
            info_bar_px = WINDOW_H / 8.0
        return float(WINDOW_W), float(WINDOW_H - info_bar_px), info_bar_px

    def _fit_zoom_for_angle(self, angle: float, mode: str) -> float:
        """Zoom (px per world unit) to fit the whole track for a given rotation angle."""
        assert self._track_aabb is not None
        min_x, max_x, min_y, max_y = self._track_aabb
        w = (max_x - min_x) * 1.1
        h = (max_y - min_y) * 1.1
        vw, vh, _ = self._view_dims_px(mode)

        ca, sa = math.cos(angle), math.sin(angle)
        need_w = abs(w * ca) + abs(h * sa)
        need_h = abs(w * sa) + abs(h * ca)

        if need_w <= 1e-6 or need_h <= 1e-6:
            return ZOOM * SCALE  # fallback to default
        return min(vw / need_w, vh / need_h)

    # ---------------------------------------------------------------------
    # Centerline caches (full-resolution, not the sampled obs)
    # ---------------------------------------------------------------------
    def _wrap_to_pi(self, a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _ensure_centerline_cache(self) -> None:
        """Build and cache centerline arrays derived from self.track (full resolution)."""
        if (
            not hasattr(self, "_cl_xy")
            or self._cl_xy is None
            or (hasattr(self, "track") and len(self._cl_xy) != len(self.track))
        ):
            if not hasattr(self, "track") or len(self.track) == 0:
                self._cl_xy = np.zeros((0, 2), dtype=np.float32)
                self._cl_tan = np.zeros((0, 2), dtype=np.float32)
                self._cl_nrm = np.zeros((0, 2), dtype=np.float32)
                self._cl_yaw = np.zeros((0,), dtype=np.float32)
                self._cl_seg_len = np.zeros((0,), dtype=np.float32)
                self._cl_s = np.zeros((1,), dtype=np.float32)
                self._cl_len_total = 0.0
                return

            pts = np.array([[t[2], t[3]] for t in self.track], dtype=np.float32)
            self._cl_xy = pts  # [N,2]
            pts_next = np.roll(pts, -1, axis=0)  # wrap to make a loop
            seg_vec = pts_next - pts  # [N,2]
            seg_len = np.linalg.norm(seg_vec, axis=1)  # [N]
            seg_len = np.maximum(seg_len, 1e-6)  # avoid div-by-zero
            self._cl_tan = seg_vec / seg_len[:, None]  # unit tangent per segment [N,2]
            self._cl_nrm = np.stack((-self._cl_tan[:, 1], self._cl_tan[:, 0]), axis=1)  # left normal [N,2]
            self._cl_yaw = np.arctan2(self._cl_tan[:, 1], self._cl_tan[:, 0]).astype(np.float32)  # [N]
            self._cl_seg_len = seg_len.astype(np.float32)  # [N]
            self._cl_s = np.concatenate(([0.0], np.cumsum(self._cl_seg_len))).astype(np.float32)  # [N+1]
            self._cl_len_total = float(self._cl_s[-1])

    # ---------------------------------------------------------------------
    # Optional: original pose error helper (kept for backward-compat/testing)
    # ---------------------------------------------------------------------
    def _pose_errors(self) -> dict:
        """
        Compute pose errors w.r.t. the full centerline (not the sampled one used in obs).
        """
        assert self.car is not None
        self._ensure_centerline_cache()
        if len(self._cl_xy) == 0:
            # degenerate
            return dict(
                ey=0.0,
                epsi=0.0,
                s=0.0,
                idx=0,
                t=0.0,
                ref_xy=(0.0, 0.0),
                ref_yaw=0.0,
                kappa=0.0,
                v_long=0.0,
                v_lat=0.0,
                L=0.0,
            )

        p = np.array(self.car.hull.position, dtype=np.float32)  # [2]
        yaw = float(self.car.hull.angle)

        # Project onto each segment; pick closest point on the polyline loop.
        pts = self._cl_xy  # [N,2]
        pts_next = np.roll(pts, -1, axis=0)  # [N,2]
        d = pts_next - pts  # [N,2]
        d2 = (d**2).sum(axis=1)  # [N]
        # param t along each segment, clamped to [0,1]
        t = np.clip(((p - pts) * d).sum(axis=1) / np.maximum(d2, 1e-12), 0.0, 1.0)  # [N]
        proj = pts + (d * t[:, None])  # [N,2] closest points per-seg
        err = p - proj  # [N,2]
        dist2 = (err**2).sum(axis=1)  # [N]
        i = int(dist2.argmin())  # best segment index
        ti = float(t[i])
        Pi = pts[i]
        di = d[i]
        seg_len_i = float(self._cl_seg_len[i])
        ref_xy = (float(Pi[0] + ti * di[0]), float(Pi[1] + ti * di[1]))

        # Path frame at segment i
        Ti = self._cl_tan[i]  # unit tangent
        Ni = self._cl_nrm[i]  # left normal
        ref_yaw = float(self._cl_yaw[i])

        # Signed cross-track error: projection of (p - ref_xy) onto left normal
        ey = float(np.dot(p - np.array(ref_xy, dtype=np.float32), Ni))

        # Heading error
        epsi = self._wrap_to_pi(yaw - ref_yaw)

        # Along-track arclength s
        s = float(self._cl_s[i] + ti * seg_len_i)
        if s >= self._cl_len_total:
            s -= self._cl_len_total

        # Approx curvature using central difference of yaw over arc length
        i_prev = (i - 1) % len(pts)
        i_next = (i + 1) % len(pts)
        dtheta = self._wrap_to_pi(float(self._cl_yaw[i_next] - self._cl_yaw[i_prev]))
        dl = 0.5 * (float(self._cl_seg_len[i_prev]) + float(self._cl_seg_len[i])) + 1e-6
        kappa = dtheta / dl

        # Velocity components in path frame (useful for MPC cost/dynamics)
        vxy = np.array(self.car.hull.linearVelocity, dtype=np.float32)
        v_long = float(np.dot(vxy, Ti))
        v_lat = float(np.dot(vxy, Ni))

        return {
            "ey": ey,
            "epsi": epsi,
            "s": s,
            "idx": i,
            "t": ti,
            "ref_xy": ref_xy,
            "ref_yaw": ref_yaw,
            "kappa": float(kappa),
            "v_long": v_long,
            "v_lat": v_lat,
            "L": float(self._cl_len_total),
        }

    # ---------------------------------------------------------------------
    # Close
    # ---------------------------------------------------------------------
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    a[0] = -1.0
                if event.key == pygame.K_d:
                    a[0] = +1.0
                if event.key == pygame.K_w:
                    a[1] = +1.0
                if event.key == pygame.K_s:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    a[0] = 0
                if event.key == pygame.K_d:
                    a[0] = 0
                if event.key == pygame.K_w:
                    a[1] = 0
                if event.key == pygame.K_s:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human", camera_center="track_centroid")

    quit = False
    while not quit:
        env.reset()
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            if terminated or truncated or restart or quit:
                break
    env.close()