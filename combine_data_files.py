import json
import argparse

parser = argparse.ArgumentParser()

# -db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument('-i', '--input_dirs',
                    help='list of input dirs: /path/to/inputs/of/strat1,/path/to/inputs/of/strat2', type=str)
parser.add_argument('-o', '--output_dir', help='output_dir: /path/to/output/dir', type=str)
args = parser.parse_args()
input_dirs = [item for item in args.input_dirs.split(',')]
output_dir = args.output_dir

cos_acc_path = "/cos_acc.json"
inner_sim_path = "/inner_sim.json"
meta_sim_path = "/meta_sim.json"
file_paths = [cos_acc_path, inner_sim_path, meta_sim_path]


def load_json(filename):
    f = open(filename)
    return json.load(f)


def store_json(data, filename):
    f = open(filename, 'w', encoding='utf-8')
    json.dump(data, f, indent=4)


for file_path in file_paths:
    combined_version = {}
    for input_dir in input_dirs:
        single_version = load_json(input_dir + file_path)
        for key, data in single_version.items():
            combined_version[key] = data

    store_json(combined_version,output_dir + file_path)

