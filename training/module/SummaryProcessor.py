from module.DataTable import DataTable

import os
import json
import glob
import datetime
import pandas as pd

from collections import OrderedDict
from pathlib import Path


def dump_summary_json(*dicts, output_path):
    summary_dict = OrderedDict()
    
    for d in dicts:
        summary_dict.update(d)
    
    assert 'training_output_path' in summary_dict, 'NEED to include a filename arg, so we can save the dict!'
    
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=False)
    
    fpath = os.path.join(output_path, summary_dict['training_output_path'].split("/")[-1] + '.summary')
    
    print("summary path: ", fpath)
    
    if os.path.exists(fpath):
        newpath = fpath
        
        while os.path.exists(newpath):
            newpath = fpath.replace(".summary", "_1.summary")
        
        # just a check
        assert not os.path.exists(newpath)
        fpath = newpath
    
    summary_dict['summary_path'] = fpath

    
    
    with open(fpath, "w+") as f:
        json.dump(summary_dict, f)
    
    return summary_dict


def get_summaries_from_path(path):
    files = glob.glob(path + "/*.summary")
    
    data = []
    for f in files:
        with open(f) as to_read:
            d = json.load(to_read)
            d['time'] = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            data.append(d)
    
    if len(data)==0:
        print("WARNING - no summary files found!!")
        return None
    
    return DataTable(pd.DataFrame(data), name='summary')


def get_summary_from_path(path):
    
    with open(path) as to_read:
        data = json.load(to_read)
        data['time'] = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    
    return DataTable(pd.DataFrame([data]), name='summary')

def get_last_summary_file_version(summary_path, filename):
    summary_search_path = summary_path + filename + "_v*"
    summary_files = glob.glob(summary_search_path)
    
    existing_ids = []
    
    for file in summary_files:
        version_number = os.path.basename(file).rstrip('.summary').split('_')[-1].lstrip('v')
        
        existing_ids.append(int(version_number))
    
    assert len(existing_ids) == len(set(existing_ids)), "no duplicate ids"
    id_set = set(existing_ids)
    version = 0
    while version in id_set:
        version += 1
    
    return version - 1


def get_version(summary_path):
    return int(os.path.basename(summary_path).rstrip('.summary').split('_')[-1].lstrip('v'))


def get_latest_summary_file_path(summaries_path, file_name_base, version=None):
    if version is None:
        version = get_last_summary_file_version(summaries_path, file_name_base)

    input_summary_path = summaries_path+"/{}_v{}.summary".format(file_name_base, version)
    return input_summary_path
