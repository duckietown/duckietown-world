from duckietown_world.world_duckietown.other_objects import SIGNS_ALIASES

ORIENTATIONS: list = ['S', 'E', 'N', 'W']

__all__ = ['get_degree_for_orientation', 'get_canonical_sign_name']


def get_degree_for_orientation(orientation: str) -> int:
    index = ORIENTATIONS.index(orientation)
    degrees: dict = {
        'S': 90, #270 + 180,
        'E': 0, #180 + 180,
        'N': 270, #90 + 180,
        'W': 180, #0 + 180
    }
    return degrees[orientation]  # index * 90 + 180#+ 180#) % 360


def get_orientation_for_degree(degree: int) -> str:
    for orientation in ORIENTATIONS:
        deg = get_degree_for_orientation(orientation)
        if deg % 360 == degree % 360:
            return orientation


def get_canonical_sign_name(name_of_sign: str) -> str:
    if name_of_sign in SIGNS_ALIASES:
        return SIGNS_ALIASES[name_of_sign]
    return name_of_sign


if __name__ == '__main__':
    print(get_canonical_sign_name("stop"))
    print(get_orientation_for_degree(90))
    print(get_degree_for_orientation('W'))
    print(get_orientation_for_degree(get_degree_for_orientation('W')) == "W")
    print(get_orientation_for_degree(get_degree_for_orientation('W') + 90) == "S")
