import os
import pickle
from dotenv import load_dotenv
import pandas as pd

def save_to_file(data, filename):
    with open(filename, 'wb') as output_file:
        pickle.dump(data, output_file, pickle.HIGHEST_PROTOCOL)


def load_from_file(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)


def convert_to_correct_type(var_name):
    load_dotenv()
    parameters = os.getenv(var_name, "")
    if not parameters:
        return None

    if parameters.isdigit():
        return int(parameters)
    try:
        float_list = [float(i) for i in parameters.split(",")]
        return float_list
    except ValueError:
        pass
    return parameters.split(",")

def getCiteseerData():
    data = pd.read_csv("datasets/citeseer.content", sep='\t', header=None, dtype=str)

    words = data[data.columns[1:-1]]
    category_list = words.to_numpy()
    ferr = data[[data.columns[-1]]].copy()
    replace_dict = {
        'Agents': 0,
        'AI': 1,
        'DB': 2,
        'IR': 3,
        'ML': 4,
        'HCI': 5
    }
    ferr.rename(columns={ferr.columns[0]: "original"}, inplace=True)

    label = ferr["original"].map(replace_dict)
    ferr["label"] = label
    return category_list.astype(float),ferr["label"].to_numpy()