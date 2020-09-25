import random
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Dict, List, Set, Tuple, TypeVar

import cv2


@dataclass
class GameDescription:
    half_width: int  # 5


PL_SELF = "self"
PL_NATURE = "nature"
PL_OTHER = "nature"

# what choices

X = TypeVar("X")


@dataclass
class DecPose:
    x: Decimal
    y: Decimal
    theta: Decimal
    vx: Decimal


@dataclass
class CarProperties:
    width: Decimal
    length: Decimal
    max_vx: Decimal
    max_accel: Decimal
    max_brake: Decimal
    max_side_by_vx: List[Decimal]
    field_of_view: Decimal


@dataclass
class CarState:
    pose: DecPose  # with respect to the lane
    turn_signal: int  # -1 left, 0 none, +1 right
    front_light: bool
    brake_light: bool


#  v += gas - brake * 2
#
#  v = 0
#  v = 1
#  v = 2
#      move: +1, 0, -1
#  v = 3
#      move: +2, +1, 0, -1, -2


@dataclass
class CarCommands:
    brake: int  # between 0 and 1
    gas: int  # between 0 and 1
    lateral: int

    turn_signal: int
    front_light: bool


@dataclass
class GameState:
    pass
    # your_pose: CarState
    # their_pose: Set[CarState]


import numpy as np


@dataclass
class WorldMap:
    obstacles: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    horizontal_roads: List[int]
    vertical_roads: List[int]


def sample_pose(wm: WorldMap) -> Tuple[int, int]:
    n, q = wm.obstacles.shape[0:2]
    B = 50
    while True:
        i = random.randint(B, n - B - 1)
        j = random.randint(B, q - B - 1)
        if not wm.obstacles[i, j]:
            return i, j


@dataclass
class CarObservations:
    local_map: np.ndarray
    local_obs: np.ndarray
    visible: np.ndarray
    relative: np.ndarray

    def __post_init__(self):
        s1 = self.local_map.shape
        s2 = self.local_obs.shape
        s3 = self.visible.shape
        s4 = self.relative.shape[:2]
        assert s1 == s2 == s3 == s4, (s1, s2, s3, s4)


@dataclass
class Car:
    cp: CarProperties
    cs: CarState
    policy: "Policy"
    obs_history: List[CarObservations]


class Policy:
    def get_action(self, obs_history: List[CarObservations], cp: CarProperties) -> CarCommands:
        pass


class SimplePolicy(Policy):
    def get_action(self, obs_history: List[CarObservations], cp: CarProperties) -> CarCommands:
        lateral = random.randint(-1, +1)
        gas = random.random() > 0.4
        brake = random.random() > 0.8
        turn_signal = random.randint(-1, +1)
        c = CarCommands(brake=brake, gas=gas, lateral=lateral, turn_signal=turn_signal, front_light=False)
        return c


class AvoidPolicy(Policy):
    def get_action(self, obs_history: List[CarObservations], cp: CarProperties) -> CarCommands:
        last = obs_history[-1]
        dx = last.relative[:, :, 0]
        dy = last.relative[:, :, 1]
        # d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)

        close1 = np.rad2deg(np.abs(theta)) < 50

        close2 = np.logical_and(dx > 2, dx < 20)
        close = np.logical_and(close1, close2)
        if np.any(last.local_obs[close]):
            gas = 0
            brake = 1
        else:

            gas = 1
            brake = 0
        lateral = 0
        # lateral = random.randint(-1, +1)
        # gas = random.random() > 0.4
        # brake = random.random() > 0.8
        turn_signal = random.randint(-1, +1)
        c = CarCommands(brake=brake, gas=gas, lateral=lateral, turn_signal=turn_signal, front_light=False)
        return c


def sample_car(wm: WorldMap) -> Car:
    i, j = sample_pose(wm)
    if wm.horizontal[i, j] > 0:
        theta = Decimal(np.pi / 2)
    else:
        theta = Decimal(0)
    pick = np.random.rand() > 0.5
    if pick:
        theta = theta + Decimal(np.pi)

    pose = DecPose(x=Decimal(i), y=Decimal(j), theta=Decimal(theta), vx=Decimal(1))
    field_of_view = np.random.randint(10, 50)
    cs = CarState(pose=pose, turn_signal=0, front_light=False, brake_light=False)
    cp = CarProperties(
        width=Decimal(2),
        length=Decimal(3),
        max_vx=Decimal(3),
        max_accel=Decimal(1),
        max_brake=Decimal(2),
        field_of_view=Decimal(field_of_view),
        max_side_by_vx=[Decimal(0), Decimal(0), Decimal(1), Decimal(2)],
    )

    if random.random() > 0.4:
        policy = AvoidPolicy()
    else:
        policy = SimplePolicy()
    car = Car(cp, cs, policy, [])
    return car


def sample_cars(wm: WorldMap, n: int) -> Dict[str, Car]:
    cars: Dict[str, Car] = {}
    for a in range(n):
        name = str(a)
        cars[name] = sample_car(wm)
    return cars


def zoom_d(a, z: int):
    return np.kron(a, np.ones((z, z), "uint8"))


def get_background_image(wm: WorldMap):
    h, w = wm.obstacles.shape
    rgb = np.zeros((h, w, 3), "uint8")
    rgb[wm.obstacles] = 255
    return rgb


def get_car_fov(theta: Decimal) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    fovX = -3, +10
    fovY = -2, +6

    fovX = -20, +30
    fovY = -15, +15
    return get_area(theta, fovX, fovY)


def get_car_footprint(theta: Decimal) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    fovX = -1, +2
    fovY = -1, +1
    return get_area(theta, fovX, fovY)


def get_area(theta: Decimal, fovX, fovY) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    nx, ny = int(np.cos(float(theta))), int(np.sin(float(theta)))

    def swap(x) -> Tuple[int, int]:
        return -x[1], -x[0]

    if (nx, ny) == (1, 0):
        return fovX, fovY
    if (nx, ny) == (-1, 0):
        return swap(fovX), fovY
    if (nx, ny) == (0, 1):
        return fovY, fovX
    if (nx, ny) == (0, -1):
        return fovY, swap(fovX)
    assert False, (theta, nx, ny)


def compute_observations(wm: WorldMap, cars: Dict[str, Car]):
    h, w = wm.obstacles.shape
    moving_obstacles = np.zeros((h, w), int)
    shape = wm.obstacles.shape

    for k, car in cars.items():
        cs: CarState = car.cs
        i = int(cs.pose.x)
        j = int(cs.pose.y)
        fpx, fpy = get_car_footprint(car.cs.pose.theta)
        i1 = i + fpx[0]
        i2 = i + fpx[1]
        j1 = j + fpy[0]
        j2 = j + fpy[1]
        i1, i2, j1, j2 = clip(shape, i1, i2, j1, j2)

        assert 0 <= j1 <= j2
        assert 0 <= i1 <= i2
        moving_obstacles[i1:i2, j1:j2] = 1

    fpx, fpy = get_car_fov(Decimal(0))
    the_is = list(range(fpx[0], fpx[1]))
    the_js = list(range(fpy[0], fpy[1]))
    s2 = (len(the_is), len(the_js))
    relative_i = np.zeros(s2, int)
    relative_j = np.zeros(s2, int)
    for m, a in enumerate(the_is):
        relative_i[m, :] = a
    for m, b in enumerate(the_js):
        relative_j[:, m] = b
    ds2 = relative_i * relative_i + relative_j * relative_j

    relative = np.dstack((relative_i, relative_j))

    res = {}
    for k, car in cars.items():
        cs: CarState = car.cs
        i = int(cs.pose.x)
        j = int(cs.pose.y)
        fpx, fpy = get_car_fov(car.cs.pose.theta)

        i1 = i + fpx[0]
        i2 = i + fpx[1]
        j1 = j + fpy[0]
        j2 = j + fpy[1]
        i1, i2, j1, j2 = clip(shape, i1, i2, j1, j2)
        assert 0 <= j1 <= j2
        assert 0 <= i1 <= i2
        sense_obstacles = wm.obstacles[i1:i2, j1:j2].copy()
        sense_cars = moving_obstacles[i1:i2, j1:j2].copy()

        # print(ds.flatten())
        outside = ds2 > car.cp.field_of_view * car.cp.field_of_view
        visible = np.logical_not(outside)

        sense_obstacles = rotate(sense_obstacles, car.cs.pose.theta)
        sense_cars = rotate(sense_cars, car.cs.pose.theta)

        sense_obstacles[outside] = 0
        sense_cars[outside] = 0

        obs = CarObservations(sense_obstacles, sense_cars, visible, relative)
        res[k] = obs
    return res


def rotate(x, theta):
    nx, ny = int(np.cos(float(theta))), int(np.sin(float(theta)))
    if (nx, ny) == (1, 0):
        return x
    if (nx, ny) == (-1, 0):
        return np.rot90(np.rot90(x))
    if (nx, ny) == (0, -1):
        return np.rot90(x)
    if (nx, ny) == (0, 1):
        return np.rot90(np.rot90(np.rot90(x)))
    assert False, theta


def clip(shape, i1, i2, j1, j2):
    i1 = max(0, i1)
    i2 = min(shape[0] - 1, i2)
    j1 = max(0, j1)
    j2 = min(shape[1] - 1, j2)
    return i1, i2, j1, j2


def draw_cars(background, rgb, cars: Dict[str, Car], erase=False):

    shape = background.shape[:2]
    for k, car in cars.items():
        cs: CarState = car.cs
        nx = int(np.cos(float(car.cs.pose.theta)))
        ny = int(np.sin(float(car.cs.pose.theta)))

        i = int(cs.pose.x)
        j = int(cs.pose.y)
        fpx, fpy = get_car_footprint(car.cs.pose.theta)
        i1 = i + fpx[0]
        i2 = i + fpx[1]
        j1 = j + fpy[0]
        j2 = j + fpy[1]
        i1, i2, j1, j2 = clip(shape, i1, i2, j1, j2)
        assert 0 <= j1 <= j2
        assert 0 <= i1 <= i2

        colors = {
            (0, 1): (255, 0, 0),
            (0, -1): (0, 255, 0),
            (1, 0): (0, 0, 255),
            (-1, 0): (120, 0, 120),
        }
        color = colors[(nx, ny)]
        if erase:
            rgb[i1:i2, j1:j2, :] = background[i1:i2, j1:j2, :]
        else:
            rgb[i1:i2, j1:j2, :] = color

    return rgb


def compute_collisions(wm: WorldMap, cars: Dict[str, Car]) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    c = np.zeros(wm.obstacles.shape, dtype=int)
    c.fill(-1)
    names = list(cars)
    collisions = set()
    outside = set()
    for m, k in enumerate(names):
        car = cars[k]

        cs: CarState = car.cs
        i = int(cs.pose.x)
        j = int(cs.pose.y)
        w = 1  # int(car.cp.width / 2)
        l = 1  # int(car.cp.length / 2)

        obstacles = wm.obstacles[i - w : i + w + 1, j - l : j + l + 1]
        if np.any(obstacles):
            outside.add(names[m])

        ground = c[i - w : i + w + 1, j - l : j + l + 1]
        if len(ground.flatten()) == 0:
            continue

        already = np.max(ground.flatten())
        if already >= 0:
            collisions.add((names[already], names[m]))

        c[i - w : i + w, j - l : j + l] = m

    return collisions, outside


@dataclass
class MapDistribution:
    n: int
    min_d: int
    max_d: int
    width_road: int


def create_map(md: MapDistribution) -> WorldMap:
    n = md.n
    min_d = md.min_d
    max_d = md.max_d
    width_road = md.width_road
    x = np.zeros((n, n), bool)
    vertical = np.zeros((n, n), int)
    vertical.fill(-1)
    horizontal = np.zeros((n, n), int)
    horizontal.fill(-1)
    x.fill(1)

    i_s = get_roads(n, min_d, max_d, width_road)
    j_s = get_roads(n, min_d, max_d, width_road)
    w = width_road

    for i in i_s:
        x[i - w : i + w, :] = False
        horizontal[i - w : i + w, :] = i
    for j in j_s:
        x[:, j - w : j + w] = False
        vertical[:, j - w : j + w] = j

    return WorldMap(
        obstacles=x, horizontal=horizontal, vertical=vertical, horizontal_roads=i_s, vertical_roads=j_s
    )


def get_roads(n: int, min_d: int, max_d: int, width: int) -> List[int]:
    i = width
    res = []

    def sample() -> int:
        dx = random.randint(min_d, max_d)
        return dx

    while True:
        dx = sample()
        i += dx

        if i >= n - width:
            break
        res.append(i)
        i += width
    return res


def reset_coords(wm: WorldMap, cp1: DecPose, cp2: DecPose) -> DecPose:
    x = cp2.x
    y = cp2.y
    vx = cp2.vx
    theta = cp2.theta
    B = 35
    H, W = wm.obstacles.shape
    if x < B:
        x = H - B
    if y < B:
        y = W - B

    if x > H - B:
        x = B

    if y > W - B:
        y = B
    return DecPose(x=x, y=y, vx=vx, theta=theta)


def update_pose(
    wm: WorldMap, cp: DecPose, cprop: CarProperties, commands: CarCommands, dt: Decimal
) -> DecPose:
    assert dt == 1
    nx = Decimal(np.cos(float(cp.theta)))
    ny = Decimal(np.sin(float(cp.theta)))

    assert 0 <= commands.brake <= 1
    assert 0 <= commands.gas <= 1
    vx = cp.vx + cprop.max_accel * commands.gas - cprop.max_brake * commands.brake
    vx = max(0, vx)
    vx = min(cprop.max_vx, vx)
    max_side = cprop.max_side_by_vx[int(cp.vx)]
    side = max(min(commands.lateral, max_side), -max_side)

    dx = cp.vx * nx - side * ny
    dy = cp.vx * ny + side * nx

    x = cp.x + dx
    y = cp.y + dy
    theta = cp.theta
    cp2 = DecPose(x=x, y=y, vx=vx, theta=theta)
    return reset_coords(wm, cp, cp2)


def update(wm: WorldMap, cs: CarState, cprop: CarProperties, commands: CarCommands, dt: Decimal) -> CarState:
    pose2 = update_pose(wm, cs.pose, cprop, commands, dt)
    # print(cs.pose, pose2)
    return replace(cs, pose=pose2)


carnames = 0


def update_cars(wm: WorldMap, cars: Dict[str, Car], dt: Decimal, min_cars: int):
    collisions, outside = compute_collisions(wm, cars)
    for k1, k2 in collisions:
        cars.pop(k1, None)
        cars.pop(k2, None)
    for k in outside:
        cars.pop(k, None)

    obs_per_car = compute_observations(wm, cars)
    for k, v in cars.items():
        obs = obs_per_car[k]
        v.obs_history.append(obs)
        commands = v.policy.get_action(v.obs_history, v.cp)
        cs = update(wm, v.cs, v.cp, commands, dt)
        v.cs = cs

    toadd = max(0, min_cars - len(cars))
    global carnames
    for i in range(toadd):
        car = sample_car(wm)
        name = str("new%s" % carnames)
        carnames += 1
        cars[name] = car


def pretty_obs(obs: CarObservations):
    a = obs.local_map.astype("uint8") * 255
    b = obs.local_obs.astype("uint8") * 255
    c = np.logical_not(obs.visible).astype("uint8") * 128

    # rgb = np.zeros((a.shape[0], a.shape[1], 3), 'uint8')
    # rgb[obs.local_map, 0] = 255
    # rgb[obs.local_obs, 1] = 255
    # rgb[obs.local_obs, 1] = 255
    # close = obs.relative[:, :, 0] > 0
    # a =b =c =close.astype('uint8')* 128
    bgr = np.dstack((a, b, c))

    bgr = np.rot90(np.rot90(bgr))
    return bgr
    # return np.vstack((a, b))


def go():
    window_name = "wn"
    md = MapDistribution(n=500, min_d=50, max_d=80, width_road=10)
    wm = create_map(md)
    N = 200
    N_min = 100
    cars = sample_cars(wm, n=N)

    it = 0
    bg = get_background_image(wm)
    frame = bg.copy()
    while True:
        it += 1
        update_cars(wm, cars, dt=Decimal(1), min_cars=N_min)

        if it % 1 == 0:

            ncars = 10
            pretties = []
            for k in list(cars)[:ncars]:
                if not cars[k].obs_history:
                    continue
                pretty = pretty_obs(cars[k].obs_history[-1])
                pretties.append(np.ones((pretty.shape[0], 5, 3), "uint8") * 255)
                pretties.append(pretty)

            rgb = np.hstack(tuple(pretties))
            cv2.imshow("one", rgb)

            draw_cars(bg, frame, cars, erase=False)

            cv2.imshow(window_name, frame)
            draw_cars(bg, frame, cars, erase=True)
            cv2.waitKey(1)

            print(it)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    go()
