from pathlib import Path
import os
import json
from glob import glob
import datetime

import pandas as pd

from module.DataTable import DataTable


def dump_summary_json(summary_dict, output_path):
    """
    
    Stores dictionary in a JSON file. If path doesn't exist, it will be created.
    The full summary path will be added to the dictionary before saving.
    
    Args:
        summary_dict (dict[str, Any]): Dictionary to be dumped to json.
        output_path (str): Output path.
    """
    
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=False)
    
    summary_path = os.path.join(output_path, summary_dict["training_output_path"].split("/")[-1] + ".summary")
    summary_dict["summary_path"] = summary_path

    with open(summary_path, "w+") as f:
        json.dump(summary_dict, f)


def get_summaries_from_path(path):
    """
    Builds a data table containing all summaries from given path.
    
    Args:
        path (str): Summaries path (can contain wild cards)

    Returns:
        (DataTable): Data table with all summaries from specified path
    """
    
    files = glob(path + "/*.summary")
    
    data = []
    for f in files:
        with open(f) as to_read:
            d = json.load(to_read)
            d['time'] = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            data.append(d)
    
    if len(data) == 0:
        print("WARNING - no summary files found!!")
        return None
    
    return DataTable(pd.DataFrame(data))


def get_last_summary_file_version(summary_path, filename):
    """
    Finds version number of the most recent summary in given path matching given file name
    Args:
        summary_path (str): Path to summaries directory (can contain wildcards)
        filename (str): Base name of summary files (without extension and version, can contain wildcards)

    Returns:
        (int): latest summary version number
    """
    
    summary_search_path = summary_path + filename + "_v*"
    summary_files = glob(summary_search_path)
    
    existing_ids = [get_version(s) for s in summary_files]
    assert len(existing_ids) == len(set(existing_ids)), "no duplicate ids"
    
    return len(existing_ids) - 1


def get_version(summary_path):
    """ Finds summary version number.
    
    Args:
        summary_path (str): Full path to summary

    Returns:
        (int): Summary version number
    """
    
    return int(os.path.basename(summary_path).rstrip('.summary').split('_')[-1].lstrip('v'))
