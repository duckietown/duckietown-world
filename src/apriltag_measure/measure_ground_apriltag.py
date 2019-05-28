import os
import argparse
from pathlib import Path
import os
import yaml
import pprint


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def error(string: str):
    print(bcolors.FAIL +
          "--------------------------------------------------------------------------------------" + bcolors.ENDC)
    print("\n\n" + bcolors.FAIL + string + bcolors.ENDC + "\n\n")

    print(bcolors.FAIL +
          "--------------------------------------------------------------------------------------" + bcolors.ENDC)


def warning(string: str):
    print("\n\n" + bcolors.WARNING + string + bcolors.ENDC + "\n\n")


def header(string: str):
    print(bcolors.OKGREEN + string + bcolors.ENDC)


def separater(string: str = None):
    if(string == None):
        warning(
            "--------------------------------------------------------------------------------------")
    else:
        warning(
            "--------------------------------------------------------------------------------------\n%s" % string)


def write_yaml(fn, data):
    with open(fn, 'w') as f:
        yaml.dump(data, f)


def load_yaml_file(fn: str):
    with open(fn) as f:
        data = f.read()
    return yaml.load(data, Loader=yaml.SafeLoader)


def get_ground_tag_dict(map_yaml):
    apriltag_dict = dict()
    for name, at_dict in map_yaml["objects"].items():
        if at_dict["kind"] == "floor_tag":
            tag_id = int(at_dict["tag"]["~TagInstance"]["tag_id"])
            apriltag_dict[tag_id] = name

    return apriltag_dict


def get_at_dict(at_number):
    warning("TODO : this part needs to ask for coordinates!")
    return {}


def update_apriltags(ground_tag_dict, map_yaml):
    result_map_yaml = map_yaml
    while True:
        header("Measure a new april tag. Enter -1 to exit ")
        at_number = input("April tag id : ")
        try:
            at_number = int(at_number)
        except Exception as e:
            error("Error : %s " % e)
            continue
        if at_number < 0:
            break
        if at_number < 300 or at_number >= 400:
            warning(
                "This april tag number (%i) is not in the 300-399 range" % at_number)
            continue_anyway = input(
                "Are you sure you want to proceed? (y/n) : ")
            if continue_anyway != "y":
                continue

        if at_number in ground_tag_dict:
            print("The april tag %i is already present" % at_number)
            modify = input("Do you want to modify its coordinates ? (y/n) : ")
            if modify == "y":
                print("modifying april tag %i" % at_number)
                at_dict = get_at_dict(at_number)
                new_name = "tag%i" % at_number
                result_map_yaml["objects"].pop(ground_tag_dict[at_number])
                result_map_yaml["objects"][new_name] = at_dict
                ground_tag_dict[at_number] = new_name

            else:
                print("not modifying april tag %i" % at_number)
                continue
        else:
            print("Adding new april tag %i" % at_number)
            at_dict = get_at_dict(at_number)
            new_name = "tag%i" % at_number
            result_map_yaml["objects"][new_name] = at_dict
            ground_tag_dict[at_number] = new_name
    return result_map_yaml


MAP_PATH = "../duckietown_world/data/gd1/maps"


def main():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('map_name', type=str,
                        help='name of the map to create or change')

    args = parser.parse_args()

    map_name = args.map_name

    if not map_name.lower().endswith(('.yaml', '.yml')):
        map_name = map_name.split(".")[0] + ".yaml"

    header("--------------\n")
    header("WELCOME FRIEND")
    header("\n--------------")

    abs_path = os.path.abspath(MAP_PATH)

    full_path = os.path.join(abs_path, map_name)

    new_file = False
    modify_file = False

    map_yaml = {}

    if not os.path.isfile(full_path):
        if not os.path.isdir(abs_path):
            error("the directory %s does not exist" % abs_path)
        else:
            warning("file %s does not exist,\nbut directory %s is there." %
                    (full_path, abs_path))
            new_file_asked = input(
                "Do you want to create file %s ? (y/n) : " % map_name)
            if(new_file_asked == "y"):
                new_file = False
            else:
                error(
                    "Program is closing as map file doesnt exist and is not to be created")
                exit()
    else:
        header("File %s is already present." % map_name)
        continue_file = input(
            "Would you like to load and modify april tag positions for %s ? (y/n) : " % map_name)
        if(continue_file == "y"):
            modify_file = True
        else:
            error(
                "Program is closing as map file exists and is not to be modified")
            exit()

    if new_file and modify_file:
        error("Can not at the same time create and modify a file")
        exit()
    elif modify_file:
        map_yaml = load_yaml_file(full_path)
        # pprint.pprint(map_yaml)
        ground_tag_dict = get_ground_tag_dict(map_yaml)
    else:
        ground_tag_dict = dict()

    result_map_yaml = update_apriltags(ground_tag_dict, map_yaml)
    pprint.pprint(result_map_yaml["objects"])
    warning("Not saving anything yet . Work in progress")


if __name__ == "__main__":
    main()

# 'A10': {'kind': 'floor_tag',
#         'pose': {'~SE2Transform': {'p': [5.0040000000000004,
#                                          -0.12]}},
#         'tag': {'~TagInstance': {'family': '36h11',
#                                  'size': 0.08,
#                                  'tag_id': 332}}}
