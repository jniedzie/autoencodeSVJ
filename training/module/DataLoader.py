from module.DataTable import DataTable

from collections import OrderedDict as odict
import h5py
import numpy as np
import os, glob, subprocess, chardet


class DataLoader:
    """
    data loader/handler/merger for h5 files with the general format of this repository
    """
    
    def __init__(self, name=""):
        self.name = name
        self.samples = odict()
        self.sample_keys = None
        self.data = odict()
        self.labels = odict()

        self.include_hlf = None
        self.include_eflow = None
        self.include_constituents = None
        self.hlf_to_drop = None
        self.efp_to_drop = None
        self.constituents_to_drop = None
        self.max_jets = None

    def set_params(self, include_hlf, include_eflow, include_constituents, hlf_to_drop, efp_to_drop, constituents_to_drop, max_jets):
        self.include_hlf = include_hlf
        self.include_eflow = include_eflow
        self.include_constituents = include_constituents
        self.hlf_to_drop = hlf_to_drop
        self.efp_to_drop = efp_to_drop
        self.constituents_to_drop = constituents_to_drop
        self.max_jets = int(max_jets)
        
    def load_all_data(self, globstring, name):
    
        """returns...
            - data: full data matrix wrt variables
            - jets: list of data matricies, in order of jet order (leading, subleading, etc.)
            - event: event-specific variable data matrix, information on MET and MT etc.
            - flavors: matrix of jet flavors to (later) split your data with
        """
    
        files = self.__glob_in_repo(globstring)
    
        if len(files) == 0:
            print("\n\nERROR -- no files found in ", globstring, "\n\n")
            raise AttributeError
    
        to_include = []
        if self.include_hlf:
            to_include.append("jet_features")
    
        if self.include_eflow:
            to_include.append("jet_eflow_variables")

        if self.include_constituents:
            to_include.append("jet_constituents")

        if not (self.include_hlf or self.include_eflow):
            raise AttributeError
    
        data_loader = DataLoader(name)
        data_loader.set_params(include_hlf=self.include_hlf,
                               include_eflow=self.include_eflow,
                               include_constituents=self.include_constituents,
                               hlf_to_drop=self.hlf_to_drop,
                               efp_to_drop=self.efp_to_drop,
                               constituents_to_drop=self.constituents_to_drop,
                               max_jets=self.max_jets)
        
        for f in files:
            data_loader.add_sample(f)
    
        train_modify = None
    
        # if self.include_hlf and self.include_eflow:
        train_modify = lambda *args, **kwargs: self.all_modify(*args, **kwargs)
        # elif self.include_hlf:
        #     train_modify = lambda *args, **kwargs: self.hlf_modify(*args, **kwargs)
        # else:
        #     train_modify = self.eflow_modify
        #
        event = data_loader.make_table('event_features', name + ' event features')

        newNames = dict()

        for column in event.df.columns:
            if type(column) is bytes:
                encoding = chardet.detect(column)["encoding"]
                newNames[column] = column.decode(encoding)

        event.df.rename(columns=newNames, inplace=True)
        event.headers = list(event.df.columns)
        
        data = train_modify(data_loader.make_tables(to_include, name, 'stack'))
        flavors = data_loader.make_table('jet_features', name + ' jet flavor', 'stack').cfilter("Flavor")

        print("Getting rid of empty jets...", end="")
        data.drop(data[data.Eta == 0].index, inplace=True)
        print("done.")

        return data, event, flavors

    def __glob_in_repo(self, globstring):
        info = {}
        info['head'] = subprocess.Popen("git rev-parse --show-toplevel".split(), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE).communicate()[0].decode("utf-8").strip('\n')
        info['name'] = subprocess.Popen("git config --get remote.origin.url".split(), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE).communicate()[0].decode("utf-8").strip('\n')
    
        repo_head = info['head']
        files = glob.glob(os.path.abspath(globstring))
    
        if len(files) == 0:
            files = glob.glob(os.path.join(repo_head, globstring))
    
        return files

    def all_modify(self, tables):
        if not isinstance(tables, list) or isinstance(tables, tuple):
            tables = [tables]
        for i, table in enumerate(tables):
            tables[i].cdrop(['0'] + self.hlf_to_drop + self.efp_to_drop + self.constituents_to_drop, inplace=True)
        
            newNames = dict()
        
            for column in table.df.columns:
                if type(column) is str:
                    if column.isdigit():
                        newNames[column] = "eflow %s" % (column)
                    else:
                        newNames[column] = column
                else:
                    encoding = chardet.detect(column)["encoding"]
                    if column.isdigit():
                        newNames[column] = "eflow %s" % (column.decode(encoding))
                    elif type(column) is bytes:
                        newNames[column] = column.decode(encoding)
        
            tables[i].df.rename(columns=newNames, inplace=True)
            tables[i].headers = list(tables[i].df.columns)
        if len(tables) == 1:
            return tables[0]
        return tables

    def hlf_modify(self, tables):
        if not isinstance(tables, list) or isinstance(tables, tuple):
            tables = [tables]
        for i, table in enumerate(tables):
            tables[i].cdrop(self.hlf_to_drop, inplace=True)
        if len(tables) == 1:
            return tables[0]
        return tables

    def eflow_modify(self, tables):
        if not isinstance(tables, list) or isinstance(tables, tuple):
            tables = [tables]
        for i, table in enumerate(tables):
            tables[i].cdrop(['0'] + self.efp_to_drop, inplace=True)
            tables[i].df.rename(columns=dict([(c, "eflow {}".format(c)) for c in tables[i].df.columns if c.isdigit()]),
                                inplace=True)
            tables[i].headers = list(tables[i].df.columns)
        if len(tables) == 1:
            return tables[0]
        return tables
    
    def add_sample(self, sample_path):
        filepath = os.path.abspath(sample_path)
        
        assert os.path.exists(filepath)
        
        if filepath not in self.samples:
            with h5py.File(filepath, mode="r") as f:
                
                print("Adding sample at path '{}'".format(filepath))
                self.samples[filepath] = f
                
                keys = set(f.keys())
                
                if self.sample_keys is None:
                    self.sample_keys = keys
                else:
                    if keys != self.sample_keys:
                        raise AttributeError
                
                self._update_data(f, keys)
    
    def make_table(self, key, name=None, third_dim_handle="stack"):
        """ stack, combine, or split """
        assert third_dim_handle in ['stack', 'combine', 'split']
        assert key in self.sample_keys
        
        data = self.data[key]
        labels = self.labels[key]
        name = name or self.name
        
        if len(data.shape) == 1:
            return DataTable(np.expand_dims(data, 1), headers=labels, name=name)
        elif len(data.shape) == 2:
            return DataTable(data, headers=labels, name=name)
        elif len(data.shape) == 3:
            ret = DataTable(
                np.vstack(data),
                headers=labels,
                name=name
            )
            # isa jet behavior
            if third_dim_handle == 'stack':
                # stack behavior
                return ret
            elif third_dim_handle == 'split':
                if key.startswith("jet"):
                    prefix = "jet"
                else:
                    prefix = "var"
                
                return [
                    DataTable(
                        ret.iloc[i::data.shape[1]],
                        name="{} {} {}".format(ret.name, prefix, i)
                    ) for i in range(data.shape[1])
                ]

            else:
                prefix = 'jet' if key.startswith('jet') else 'var'
                return DataTable(
                    self.stack_data(data, axis=1),
                    headers=self.stack_labels(labels, data.shape[1], prefix),
                    name=name,
                )
                # combine behavior
        elif len(data.shape) == 4:

            constituents_labels = []
            for i_constituent in range(len(data[0][0])):
                for label in labels:
                    constituents_labels.append(np.bytes_("constituent_")+label+np.bytes_("_"+str(i_constituent)))


            stacked_data = np.vstack(data)
            stacked_data = stacked_data.transpose((1, 2, 0)) # [eta_1, phi_1, pt_1, y_1, E_1, eta_2, phi_2, pt_2, y_2, E_2, ...]
            # stacked_data = stacked_data.transpose((2, 1, 0))  # [eta_1, eta_2,..., phi_1, phi_2, ..., pt_1, pt_2, ... , y_1, y_2, ..., E_1, E_2, ...]
            stacked_data = np.vstack(stacked_data)
            stacked_data = stacked_data.transpose()

            return DataTable(stacked_data, headers=constituents_labels, name=name )

        else:
            raise AttributeError
    
    def make_tables(self, keylist, name, third_dim_handle="stack"):
        tables = []
        for k in keylist:
            tables.append(self.make_table(k, None, third_dim_handle))
        assert len(tables) > 0
        ret, tables = tables[0], tables[1:]
        for table in tables:
            if third_dim_handle == "split":
                for i, (r, t) in enumerate(zip(ret, table)):
                    ret[i] = r.cmerge(t, name + str(i))
            else:
                ret = ret.cmerge(table, name)
        return ret
    
    def stack_data(self, data, axis=1):
        return np.hstack(np.asarray(np.split(data, data.shape[axis], axis=axis)).squeeze())
    
    def stack_labels(self, labels, n, prefix):
        new = []
        for j in range(n):
            for l in labels:
                new.append("{}{}_{}".format(prefix, j, l))
        return np.asarray(new)
       
    def _update_data(self, sample_file, keys_to_add):
        for key in keys_to_add:
            assert 'data' in sample_file[key]
            assert 'labels' in sample_file[key]
            
            if key not in self.labels:
                self.labels[key] = np.asarray(sample_file[key]['labels'])
            else:
                assert (self.labels[key] == np.asarray(sample_file[key]['labels'])).all()

            sample_data = sample_file[key]['data']

            if key == "jet_features" or key == "jet_eflow_variables":
                sample_data = sample_data[:, 0:self.max_jets, :]

            if key == "jet_constituents":
                sample_data = sample_data[:, 0:self.max_jets, :, :]

            if key not in self.data:
                self.data[key] = np.asarray(sample_data)
            else:
                self.data[key] = np.concatenate([self.data[key], sample_data])
            
                

