import json
import os
import re
from collections import defaultdict

def load_json(json_file_path):
    with open(json_file_path) as file:
        return json.load(file)


def get_files_from_json(data):
    files_in_json = set()
    for entry in data:
        files_in_json.add(entry["img"])
        files_in_json.add(entry["explication"])
        files_in_json.add(entry["activations"])
        files_in_json.add(entry["imgpertubated"])
        files_in_json.add(entry["explicationpertubated"])
        files_in_json.add(entry["activationspertubated"])
    return files_in_json


def get_files_in_directory(directory_path):
    
    excluded_files = [ 'expl.json','readme.md']
    
    files_in_directory = set()
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file not in excluded_files:
                files_in_directory.add(
                    "result_dataset/" + os.path.relpath(os.path.join(root, file), start=directory_path)
                )
    return files_in_directory


def compare_files(files_in_json, files_in_directory):
    missing_in_json = files_in_directory - files_in_json
    missing_in_directory = files_in_json - files_in_directory

    return missing_in_json, missing_in_directory


def extract_ids1(file_path):
    match = re.search(r"_([0-9]+)_([0-9]+)_", file_path)
    if match:
        id_img = match.group(1)
        id_xai = match.group(2)
        return id_img, id_xai
    return None, None


def extract_ids2(file_path):
    match = re.search(r"_([0-9]+)_([0-9]+)_([0-9]+)_", file_path)
    if match:
        id_img = match.group(1)
        id_perturb = match.group(2)
        id_xai = match.group(3)
        return id_img, id_perturb, id_xai
    return None, None, None


def find_matching_file(directory_path, id_img, id_xai, dataset_name, same_or_diff):
    for root, _, files in os.walk(directory_path + "/" + same_or_diff):
        for file in files:
            id_img_file, _, id_xai_file = extract_ids2(file)
            if (
                id_img_file == id_img
                and id_xai_file == id_xai
                and dataset_name == file.split("_")[0]
            ):
                return os.path.relpath(os.path.join(root, file), start=directory_path)

    return None


def update_json_entries(data, missing_files):
    new_data = []
    for entry in data:
        for key in ["imgpertubated"]:
            file_path = entry[key]
            if file_path in missing_files:
                dataset_name = entry["explication"].split("_")[1].split("/")[-1]
                same_or_diff = entry["imgpertubated"].split("/")[1]
                id_img, _, id_xai = extract_ids2(file_path)
                if id_img and id_xai:
                    new_file = find_matching_file(
                        "result_dataset", id_img, id_xai, dataset_name, same_or_diff
                    )
                    if new_file:
                        new_file = new_file.split("/")[-1]
                        print(
                            "Updating entry:",
                            entry["imgpertubated"],
                            entry["explicationpertubated"],
                            entry["activationspertubated"],
                            entry["predictedclasspertubated"],
                        )
                        entry["imgpertubated"] = "result_dataset/" + same_or_diff + "/" + new_file
                        entry["explicationpertubated"] = (
                            "result_dataset/"
                            + same_or_diff.replace("imgs_perturbated", "expl_pertubated")
                            + "/"
                            + new_file
                        )
                        if int(id_xai) > 38:
                            entry["activationspertubated"] = (
                                "result_dataset/"
                                + same_or_diff.replace(
                                    "imgs_perturbated", "activations_perturbated"
                                )
                                + "/"
                                + new_file.replace("png", "json")
                            )
                        else:
                            entry["activationspertubated"] = (
                                "result_dataset/"
                                + same_or_diff.replace(
                                    "imgs_perturbated", "activations_perturbated"
                                )
                                + "/"
                                + new_file
                            )
                        entry["predictedclasspertubated"] = new_file.split(".")[0].split("_")[-1]
                        print(
                            "Updated entry:",
                            entry["imgpertubated"],
                            entry["explicationpertubated"],
                            entry["activationspertubated"],
                            entry["predictedclasspertubated"],
                        )
        new_data.append(entry)
    return new_data

def update_json_entries2(data, missing_files):
    new_data = []
    for entry in data:
        for key in ["explicationpertubated"]:
            file_path = entry[key]
            _, _, id_xai = extract_ids2(file_path)
            if file_path in missing_files and id_xai == "39":
                print("oui", file_path, id_xai)
                print(
                    "Updating entry:",
                    entry["imgpertubated"],
                    entry["explicationpertubated"],
                    entry["activationspertubated"],
                    entry["predictedclasspertubated"],
                )
                entry["activationspertubated"] = entry["activationspertubated"].replace(
                    ".png", ".json"
                )
                print(
                    "Updated entry:",
                    entry["imgpertubated"],
                    entry["explicationpertubated"],
                    entry["activationspertubated"],
                    entry["predictedclasspertubated"],
                )
        new_data.append(entry)
    return new_data


def update_json_entries3(data, missing_files):
    for file in missing_files:
        _, _, id_xai = extract_ids2(file)
        if (int(id_xai) < 39) and ("activations_pertubated" in file) and (file.endswith(".png")):
            for entry in data:
                if entry["activationspertubated"] == file:
                    updated_entry = entry
                    updated_entry["activationspertubated"] = entry["activationspertubated"].replace(
                        ".png", ".npy"
                    )
                    data.remove(entry)
                    data.append(updated_entry)
                    print(
                        "Updated entry:",
                        updated_entry["activationspertubated"],
                        entry["activationspertubated"],
                    )
    return data

def remove_entry_by_path(json_data, path):
    """
    Remove an entry from a list of dictionaries based on a given path.

    Parameters:
    ----------
    json_data : list
        The list of dictionaries loaded from a JSON file.
    path : str
        The specific path to match for removal.

    Returns:
    -------
    list
        The updated list with the matching entry removed.
    """
    updated_data = [entry for entry in json_data if path not in entry.values()]
    return updated_data

def find_duplicates(json_data):
    """
    Find and return duplicate entries in a list of dictionaries.

    Parameters:
    ----------
    json_data : list
        The list of dictionaries to check for duplicates.

    Returns:
    -------
    list
        A list of duplicate dictionaries.
    """
    # Create a dictionary to store the occurrences of each entry
    entries_count = defaultdict(int)

    # Convert dictionaries to frozensets to make them hashable and count them
    for entry in json_data:
        # Convert each dictionary to a frozenset of its items, which is hashable
        entry_tuple = frozenset(entry.items())
        entries_count[entry_tuple] += 1

    # Identify duplicates: entries with a count greater than 1
    duplicates = [dict(entry) for entry, count in entries_count.items() if count > 1]

    return duplicates


def main(json_file_path, directory_path):
    data = load_json(json_file_path)
    files_in_json = get_files_from_json(data)
    files_in_directory = get_files_in_directory(directory_path)

    missing_in_json, missing_in_directory = compare_files(files_in_json, files_in_directory)

    print('dup',find_duplicates(data),len(find_duplicates(data)))
    
    print(f"Files in directory but not in JSON: {len(missing_in_json)}")
    for file in missing_in_json:
        print(f"- {file}")

    print(f"\nFiles in JSON but not in directory: {len(missing_in_directory)}")
    for file in missing_in_directory:
        print(f"- {file}")
        '''data = remove_entry_by_path(data, file)'''

    # Update JSON entries for missing perturbated files
    '''updated_data = update_json_entries(data, missing_in_directory)'''
    
    '''# Save updated JSON
    with open('dataset_zip/result_dataset/expl.json', 'w') as file:
        json.dump(data, file, indent=4)'''

if __name__ == "__main__":
    json_file_path = "dataset_zip/result_dataset/expl.json"
    directory_path = "dataset_zip/result_dataset"
    main(json_file_path, directory_path)
