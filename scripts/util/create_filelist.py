"""
Create filelist from a directory.
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict
import random
import math


def remove_extension(filename):
    return os.path.splitext(filename)[0]


parser = argparse.ArgumentParser(description="Create filelist from a directory.")

parser.add_argument(
    "--root_dir",
    type=str,
    help="Root directory",
    required=True,
)

parser.add_argument(
    "--dest_file",
    type=str,
    help="Destination folder",
    default=None,
    required=False,
)

parser.add_argument("--ext", type=str, help="the file extension", default=".mp4")

parser.add_argument("--additional_filter", nargs="+", help="more filters for video", default=[])

parser.add_argument("--filter_type", type=str, help="the kind of filter", default="ends")

parser.add_argument("--train_val_test", nargs="+", type=float, default=None)

parser.add_argument("--by_id", action="store_true", default=False)

parser.add_argument("--id_type", type=str, default="split")

parser.add_argument("--filter_folders", nargs="+", default=[])

parser.add_argument("--only_folders", nargs="+", default=[])
parser.add_argument("--contains", nargs="+", default=[])

parser.add_argument("--start_with", nargs="+", default=[])

args = parser.parse_args()

root_dir = args.root_dir

Path(args.dest_file).parent.mkdir(parents=True, exist_ok=True)

files_dict = defaultdict(list)

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if len(args.filter_folders) > 0:
            if any([folder in root for folder in args.filter_folders]):
                continue

        # Skip if start_with is specified and file doesn't start with any of the specified prefixes
        if len(args.start_with) > 0:
            if not any(file.lower().startswith(prefix) for prefix in args.start_with):
                continue

        # Skip if only_folders is specified and none of the folders are in the path
        if len(args.only_folders) > 0:
            if not any([folder in root for folder in args.only_folders]):
                continue

        complete_path = os.path.join(root, file)
        valid_ext = False
        if args.ext == ".mp4":
            valid_ext = file.endswith(".mp4") or file.lower().endswith(".mov")
        else:
            valid_ext = file.endswith(args.ext)

        if valid_ext and os.stat(complete_path).st_size > 0:
            if args.filter_type == "ends" and len(args.additional_filter) > 0:
                if file.split(".")[0].endswith(tuple(args.additional_filter)):
                    continue

            # Check if file contains any of the required keywords
            if len(args.contains) > 0:
                if not any(keyword in file for keyword in args.contains):
                    continue

            if args.id_type == "split":
                files_dict[file.split("_")[0].split("-")[0]].append(complete_path)
            elif args.id_type == "prev_folder":
                files_dict[complete_path.split("/")[-2]].append(complete_path)
            else:
                raise ValueError("id_type must be split or prev_folder")
            # f.write(os.path.join(root, file) + "\n")

if args.train_val_test is None:
    with open(args.dest_file, "w") as f:
        for file_list in files_dict.values():
            for file in file_list:
                f.write(file + "\n")
else:
    if len(args.train_val_test) != 3:
        raise ValueError("train_val_test must have 3 values")
    assert sum(args.train_val_test) == 1.0
    if args.by_id:
        ids = list(files_dict.keys())
        random.shuffle(ids)
        train_len = int(len(ids) * args.train_val_test[0])
        val_len = math.ceil(len(ids) * args.train_val_test[1])
        test_len = math.ceil(len(ids) * args.train_val_test[2])
        print(f"total len: {len(ids)}, train_len: {train_len}, val_len: {val_len}, test_len: {test_len}")
        train_ids = ids[:train_len]
        test_ids = ids[train_len : train_len + test_len]
        val_ids = ids[train_len + test_len :]
        with open(remove_extension(args.dest_file) + "_train.txt", "w") as f:
            for id in train_ids:
                for file in files_dict[id]:
                    f.write(file + "\n")
        with open(remove_extension(args.dest_file) + "_val.txt", "w") as f:
            for id in val_ids:
                for file in files_dict[id]:
                    f.write(file + "\n")
        with open(remove_extension(args.dest_file) + "_test.txt", "w") as f:
            for id in test_ids:
                for file in files_dict[id]:
                    f.write(file + "\n")
    else:
        all_files = [file for file_list in files_dict.values() for file in file_list]
        random.shuffle(all_files)
        train_len = int(len(all_files) * args.train_val_test[0])
        val_len = int(len(all_files) * args.train_val_test[1])
        test_len = int(len(all_files) * args.train_val_test[2])
        train_files = all_files[:train_len]
        test_files = all_files[train_len : train_len + test_len]
        val_files = all_files[train_len + test_len :]
        with open(remove_extension(args.dest_file) + "_train.txt", "w") as f:
            for file in train_files:
                f.write(file + "\n")
        with open(remove_extension(args.dest_file) + "_val.txt", "w") as f:
            for file in val_files:
                f.write(file + "\n")
        with open(remove_extension(args.dest_file) + "_test.txt", "w") as f:
            for file in test_files:
                f.write(file + "\n")
