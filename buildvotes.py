import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load the expl_combined_new.json file
file_path = "result_dataset/expl_vote.json"

with Path(file_path).open() as f:
    data = json.load(f)

# Provided fake_vote dictionary
fake_vote = {
    "51": {
        "0": 3,
        "1": 2,
        "2": 2,
        "3": 1,
        "4": 3,
        "5": 3,
        "6": 4,
        "7": 3,
        "8": 4,
        "9": 4,
        "10": 3,
        "11": 1,
        "12": 3,
        "13": 4,
        "14": 3,
        "15": 4,
        "16": 4,
        "17": 5,
        "18": 4,
        "19": 4,
        "20": 1,
        "21": 3,
        "22": 2,
        "23": 1,
        "24": 3,
        "25": 2,
        "26": 5,
        "27": 3,
        "28": 1,
        "29": 1,
        "30": 4,
        "31": 3,
        "32": 1,
        "33": 3,
        "34": 2,
        "35": 2,
        "36": 3,
        "37": 2,
        "38": 4,
        "46": 4,
        "47": 3,
        "39": 4,
        "40": 1,
        "41": 2,
        "42": 2,
        "43": 2,
        "44": 2,
        "45": 1,
        "48": 2,
        "49": 3,
    },
    "52": {
        "0": 4,
        "1": 1,
        "2": 2,
        "3": 1,
        "4": 2,
        "5": 1,
        "6": 4,
        "7": 3,
        "8": 4,
        "9": 4,
        "10": 2,
        "11": 1,
        "12": 4,
        "13": 5,
        "14": 4,
        "15": 4,
        "16": 2,
        "17": 5,
        "18": 4,
        "19": 2,
        "20": 1,
        "21": 4,
        "22": 2,
        "23": 1,
        "24": 3,
        "25": 2,
        "26": 4,
        "27": 4,
        "28": 2,
        "29": 3,
        "30": 4,
        "31": 4,
        "32": 1,
        "33": 3,
        "34": 2,
        "35": 3,
        "36": 4,
        "37": 4,
        "38": 4,
        "46": 5,
        "47": 2,
        "39": 1,
        "40": 4,
        "41": 5,
        "42": 3,
        "43": 1,
        "44": 2,
        "45": 1,
        "48": 1,
        "49": 3,
    },
    "53": {
        "0": 4,
        "1": 1,
        "2": 2,
        "3": 2,
        "4": 1,
        "5": 3,
        "6": 3,
        "7": 2,
        "8": 4,
        "9": 5,
        "10": 1,
        "11": 1,
        "12": 5,
        "13": 4,
        "14": 5,
        "15": 4,
        "16": 1,
        "17": 4,
        "18": 5,
        "19": 1,
        "20": 1,
        "21": 4,
        "22": 1,
        "23": 2,
        "24": 4,
        "25": 1,
        "26": 5,
        "27": 5,
        "28": 1,
        "29": 2,
        "30": 4,
        "31": 5,
        "32": 1,
        "33": 4,
        "34": 2,
        "35": 1,
        "36": 3,
        "37": 3,
        "38": 3,
        "46": 5,
        "47": 4,
        "39": 2,
        "40": 1,
        "41": 5,
        "42": 5,
        "43": 1,
        "44": 1,
        "45": 1,
        "48": 4,
        "49": 2,
    },
}

# Calculate the average of the dictionaries in fake_vote
fake_vote_mean = {}
for key in fake_vote["51"]:
    fake_vote_mean[key] = (fake_vote["51"][key] + fake_vote["52"][key] + fake_vote["53"][key]) / 3

sigma_noise = 0

filtered_data = []
# Process the JSON data
for item in data:
    explanation_path = item["explication"]
    mu, sigma = 0, sigma_noise  # mean and standard deviation
    s = np.random.normal(mu, sigma, 1)
    vote = np.clip(fake_vote_mean[str(explanation_path.split("_")[3])] + s, 1, 5)[0]
    dataset = explanation_path.split("/")[-1].split("_")[0]
    vote = np.round(vote, 2)
    filtered_data.append(
        {
            "ID": item["ID"],
            "img": item["img"],
            "GTclass": item["GTclass"],
            "explication": item["explication"],
            "activations": item["activations"],
            "predictedclass": item["predictedclass"],
            "imgpertubatedsame": item["imgpertubatedsame"],
            "explicationpertubatedsame": item["explicationpertubatedsame"],
            "activationspertubatedsame": item["activationspertubatedsame"],
            "imgpertubateddiff": item["imgpertubateddiff"],
            "explicationpertubateddiff": item["explicationpertubateddiff"],
            "activationspertubateddiff": item["activationspertubateddiff"],
            "predictedclasspertubated": item["predictedclasspertubated"],
            "target_mark": vote,
        }
    )

print("CHECK", len(data), len(filtered_data))

# Save the final filtered data to a new JSON file
filtered_file_path = f"result_dataset/expl_vote_.json"
with Path(filtered_file_path).open('w') as f:
    json.dump(filtered_data, f, indent=4)

'''datasets = {}
for item in filtered_data:
    dataset = item["explication"].split("/")[-1].split("_")[0]
    if dataset not in datasets:
        datasets[dataset] = []
    datasets[dataset].append(item["target_mark"])

# Plotting histograms
for dataset, scores in datasets.items():
    plt.figure()
    plt.hist(scores, bins=[1, 2, 3, 4, 5, 6], edgecolor="black", align="left", rwidth=0.8)
    plt.xticks([1, 2, 3, 4, 5])
    plt.title(f"Histogram of Scores for {dataset}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(1, 6))
    plt.savefig(f"{dataset}.png")'''
