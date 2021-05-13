import numpy as np
import pandas as pd

import argparse
import os
import pprint
import csv
import yaml


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def error(string: str):
    print(
        bcolors.FAIL
        + "--------------------------------------------------------------------------------------"
        + bcolors.ENDC
    )
    print("\n\n" + bcolors.FAIL + string + bcolors.ENDC + "\n\n")

    print(
        bcolors.FAIL
        + "--------------------------------------------------------------------------------------"
        + bcolors.ENDC
    )


def warning(string: str):
    print("\n\n" + bcolors.WARNING + string + bcolors.ENDC + "\n\n")


def header(string: str):
    print(bcolors.OKGREEN + string + bcolors.ENDC)


def separater(string: str = None):
    if string is None:
        warning("--------------------------------------------------------------------------------------")
    else:
        warning(
            "--------------------------------------------------------------------------------------\n%s"
            % string
        )


def write_yaml(fn, data):
    with open(fn, "w") as f:
        yaml.dump(data, f)


def load_yaml_file(fn: str):
    with open(fn) as f:
        data = f.read()
    return yaml.load(data, Loader=yaml.SafeLoader)


def create_empty_map_file():
    empty_map = {"objects": {}, "tiles": {}, "tile_size": 0.585, "version": 2}
    return empty_map


def input_int(string):
    int_input = input(string)
    try:
        int_input = int(int_input)
    except Exception as e:
        error("Error : %s" % e)
        return None
    return int_input


def input_float(string):
    float_input = input(string)
    try:
        float_input = float(float_input)
    except Exception as e:
        error("Error : %s" % e)
        return None
    return float_input


class Apriltag:
    def __init__(self, tag_id, x, y, angle):
        self.x = x
        self.tag_id = tag_id
        self.y = y
        self.angle = angle

    def to_dict(self):
        return {
            "kind": "floor_tag",
            "pose": {"~SE2Transform": {"p": [self.x, self.y], "theta_deg": self.angle}},
            "tag": {"~TagInstance": {"family": "36h11", "size": 0.065, "tag_id": self.tag_id}},
        }


class Apriltag_measurer:
    def __init__(self, full_path, new_file, modify_file):
        self.name = "Mouahaha"
        if new_file and modify_file:
            error("Can not at the same time create and modify a file")
            exit()
        elif modify_file:
            self.map_yaml = load_yaml_file(full_path)
            if "objects" not in self.map_yaml:
                self.map_yaml["objects"] = dict()
            self.ground_tag_dict = self.get_ground_tag_dict()
        else:
            self.map_yaml = create_empty_map_file()
            self.ground_tag_dict = dict()

        self.x_offset = 0.012
        self.y_offset = 0.012
        self.tile_size = float(self.map_yaml["tile_size"])

    def get_ground_tag_dict(self):
        apriltag_dict = dict()
        for name, at_dict in self.map_yaml["objects"].items():
            if at_dict["kind"] == "floor_tag":
                tag_id = int(at_dict["tag"]["~TagInstance"]["tag_id"])
                apriltag_dict[tag_id] = name

        return apriltag_dict

    def get_at_dict_csv(self, Id, x_coord, y_coord, x_measure, y_measure, angle_measure):
        complete_x = self.tile_size * x_coord + x_measure + self.x_offset
        complete_y = self.tile_size * y_coord + y_measure + self.y_offset
        apriltag = Apriltag(Id, complete_x, complete_y, angle_measure)
        return apriltag.to_dict()

    def get_at_dict(self, at_number):
        x_count = input_int("give x coordinate of tile (first tile at origin is (0,0)) : ")
        if x_count is not None:
            y_count = input_int("give y coordinate of tile (first tile at origin is (0,0)) : ")
            if y_count is not None:
                x_measure = input_float("give x measure in meters from interior side of tile : ")
                if x_measure is not None:
                    y_measure = input_float("give y measure in meters from interior side of tile : ")
                    if y_measure is not None:
                        angle_measure = input_int(
                            "give angle of apriltag in degrees (turned as readable from origin is 0) : "
                        )
                        if angle_measure is not None:
                            complete_x = self.tile_size * x_count + x_measure + self.x_offset
                            complete_y = self.tile_size * y_count + y_measure + self.y_offset
                            apriltag = Apriltag(at_number, complete_x, complete_y, angle_measure)
                            return apriltag.to_dict()
        return {}

    def read_at_data_from_csv(self, filepath,map_yaml):
        while True:
            try:
                data = pd.read_csv(filepath)
            except:
                print("\nwrong file path or format!")
                filepath=input("Please provide the absolute path to the csv file: ")
                continue
            break
        Id = data[['Id']].to_numpy()
        x_coord = data[['x_coord']].to_numpy()
        y_coord = data[['y_coord']].to_numpy()
        x_measure = data[['x_measure']].to_numpy()
        y_measure = data[['y_measure']].to_numpy()
        angle_measure = data[['angle']].to_numpy()
        for i in range(len(Id)):
            at_dict_csv = self.get_at_dict_csv(int(Id[i]), float(x_coord[i]), float(y_coord[i]), float(x_measure[i]), float(y_measure[i]), int(angle_measure[i]))
            new_name = "tag%i" % int(Id[i])
            if int(Id[i]) in self.ground_tag_dict: 
                map_yaml["objects"].pop(self.ground_tag_dict[int(Id[i])])
            map_yaml["objects"][new_name] = at_dict_csv
            self.ground_tag_dict[int(Id[i])] = new_name   
        
        set_map_diff_csv = set(self.ground_tag_dict.keys()) - set(Id.flatten())
        if len(set_map_diff_csv) >0: 
            print("These april tags are present:")
            for tag_id in sorted(set_map_diff_csv):
                print(tag_id)
            modify = input("Do you want to keep them? (y/n) : ")
            if modify == "n":
                for tag_id in set_map_diff_csv:
                    map_yaml["objects"].pop(self.ground_tag_dict[tag_id])
        
        return map_yaml

    def update_apriltags(self):
        result_map_yaml = self.map_yaml
        choice = int(input("Read at from file (0) or enter manually (1): "))
        if choice == 0:
            self.read_at_data_from_csv(
                filepath=input("Please provide the absolute path to the csv file: "),
                map_yaml=result_map_yaml,
            )
        else:
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
                    warning("This april tag number (%i) is not in the 300-399 range" % at_number)
                    continue_anyway = input("Are you sure you want to proceed? (y/n) : ")
                    if continue_anyway != "y":
                        continue

                if at_number in self.ground_tag_dict:
                    print("The april tag %i is already present" % at_number)
                    modify = input("Do you want to modify its coordinates ? (y/n) : ")
                    if modify == "y":
                        at_dict = self.get_at_dict(at_number)
                        new_name = "tag%i" % at_number
                        result_map_yaml["objects"].pop(self.ground_tag_dict[at_number])
                        result_map_yaml["objects"][new_name] = at_dict
                        self.ground_tag_dict[at_number] = new_name

                    else:
                        print("not modifying april tag %i" % at_number)
                        continue
                else:
                    print("Adding new april tag %i" % at_number)
                    at_dict = self.get_at_dict(at_number)
                    new_name = "tag%i" % at_number
                    result_map_yaml["objects"][new_name] = at_dict
                    self.ground_tag_dict[at_number] = new_name
        return result_map_yaml


MAP_PATH = "src/duckietown_world/data/gd1/maps"

def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("map_name", type=str, help="name of the map to create or change")

    args = parser.parse_args()

    map_name = args.map_name

    if not map_name.lower().endswith((".yaml", ".yml")):
        map_name = map_name.split(".")[0] + ".yaml"

    header("--------------\n")
    header("WELCOME FRIEND")
    header("\n--------------")

    abs_path = os.path.abspath(MAP_PATH)

    full_path = os.path.join(abs_path, map_name)

    new_file = False
    modify_file = False

    if not os.path.isfile(full_path):
        if not os.path.isdir(abs_path):
            error("the directory %s does not exist" % abs_path)
        else:
            warning("file %s does not exist,\nbut directory %s is there." % (full_path, abs_path))
            new_file_asked = input("Do you want to create file %s ? (y/n) : " % map_name)
            if new_file_asked == "y":
                new_file = False
            else:
                error("Program is closing as map file doesnt exist and is not to be created")
                exit()
    else:
        header("File %s is already present." % map_name)
        continue_file = input(
            "Would you like to load and modify april tag positions for %s ? (y/n) : " % map_name
        )
        if continue_file == "y":
            modify_file = True
        else:
            error("Program is closing as map file exists and is not to be modified")
            exit()

    apriltag_measurer = Apriltag_measurer(full_path, new_file, modify_file)

    result_map_yaml = apriltag_measurer.update_apriltags()
    pprint.pprint(result_map_yaml["objects"])

    separater()
    header("Inputing of april tag position done")
    separater()
    while True:
        overwrite = input("Are you sure you want to overwrite %s ? (y/n) : " % map_name)
        if overwrite == "y":
            write_yaml(full_path, result_map_yaml)
            separater()
            header("File %s saved at %s. Exiting now." % (map_name, full_path))
            separater()
            exit()
        elif overwrite == "n":
            error("Did not overwrite %s. Exiting now." % map_name)
            exit()
        else:
            warning("Please enter 'y' or 'n'")


if __name__ == "__main__":
    main()
