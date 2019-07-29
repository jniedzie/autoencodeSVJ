import utils
import trainer
import numpy as np
import tensorflow as tf
import os
import models

class ae_evaluation:
    
    def __init__(
        self,
        name,
        qcd_path=None,
        signal_path=None,
        custom_objects={},
    ):
        self.name = utils.summary_by_name(name)
        self.d = utils.load_summary(self.name)

        if qcd_path is None:
            if 'qcd_path' in self.d:
                qcd_path = self.d['qcd_path']
            else:
                raise AttributeError("No QCD path found; please specify!")

        if signal_path is None:
            if 'signal_path' in self.d:
                signal_path = self.d['signal_path']
            else:
                raise AttributeError("No signal path found; please specify!")
        
        self.qcd_path = qcd_path
        self.signal_path = signal_path
                
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
    #     signal_path = "data/signal/base_{}/*.h5".format(eflow_base)
    #     qcd_path = "data/background/base_{}/*.h5".format(eflow_base)

        (self.signal,
         self.signal_jets,
         self.signal_event,
         self.signal_flavor) = utils.load_all_data(
            self.signal_path,
            "signal", include_hlf=self.hlf, include_eflow=self.eflow
        )

        (self.qcd,
         self.qcd_jets,
         self.qcd_event,
         self.qcd_flavor) = utils.load_all_data(
            self.qcd_path, 
            "qcd background", include_hlf=self.hlf, include_eflow=self.eflow
        )

        self.seed = self.d['seed']

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.target_dim = self.d['target_dim']
        self.input_dim = len(self.signal.columns)
        self.test_split = self.d['test_split']
        self.val_split = self.d['val_split']
        self.filename = self.d['filename']

        self.norm_args = {
            "norm_type": str(self.d["norm_type"])
        }
        
        
        self.all_train, self.test = self.qcd.train_test_split(self.test_split, self.seed)
        self.train, self.val = self.all_train.train_test_split(self.val_split, self.seed)

        self.train_norm = self.train.norm(out_name="qcd train norm", **self.norm_args)
        self.val_norm = self.train.norm(self.val, out_name="qcd val norm", **self.norm_args)

        self.test_norm = self.test.norm(out_name="qcd test norm", **self.norm_args)
        self.signal_norm = self.signal.norm(out_name="signal norm", **self.norm_args)

        self.train.name = "qcd training data"
        self.test.name = "qcd test data"
        self.val.name = "qcd validation data"
        
        self.custom_objects = custom_objects
        
        self.instance = trainer.trainer(self.filename)
        self.ae = self.instance.load_model(custom_objects=self.custom_objects)
        
        [self.qcd_err, self.signal_err], [self.qcd_recon, self.signal_recon] = utils.get_recon_errors([self.test_norm, self.signal_norm], self.ae)

        self.qcd_reps = utils.data_table(self.ae.layers[1].predict(self.test_norm.data), name='background reps')
        self.signal_reps = utils.data_table(self.ae.layers[1].predict(self.signal_norm.data), name='signal reps')
        
    def node_reps(
        self,
        show_plot=True,
        alpha=1,
        normed=1,
        figname='node reps',
        figsize=10,
        figloc='upper right',
        cols=4,
        *args,
        **kwargs
    ):
        if show_plot:
            self.qcd_reps.plot(
                self.signal_reps, alpha=alpha,
                normed=normed, figname=figname, figsize=figsize,
                figloc=figloc, cols=cols, *args, **kwargs
            )
            return 
        return {'qcd': self.qcd_reps, 'signal': self.signal_reps}
        
    def metrics(
        self,
        show_plot=True,
        *args,
        **kwargs
    ):
        if show_plot:
            self.instance.plot_metrics(*args, **kwargs)
            return
        return self.instance.metrics
    
    def error(
        self,
        show_plot=True,
        figsize=15, normed='n', 
        figname='error for eflow variables', 
        yscale='linear', rng=((0, 0.08), (0, 0.3)), 
        figloc="upper right", *args, **kwargs
    ):
        if show_plot:
            self.qcd_err.plot(
                self.signal_err, figsize=figsize, normed=normed, 
                figname=figname, 
                yscale=yscale, rng=rng, 
                figloc=figloc, *args, **kwargs
            )
            return
        return {'qcd': self.qcd_err, 'signal': self.signal_err}
    
    def roc(
        self,
        show_plot=True,
        metrics=['mae', 'mse'],
        figsize=8,
        figloc=(0.3, 0.2),
        *args,
        **kwargs
    ):
        
        if show_plot:

            utils.roc_auc_plot(
                self.qcd_err, self.signal_err,
                metrics=metrics, figsize=figsize,
                figloc=figloc
            )
            
            return

        roc_dict = utils.roc_auc_dict(
            self.qcd_err, self.signal_err,
            metrics=metrics
        ).values()[0]

        result_args = dict([(r + '_auc', roc_dict[r]['auc']) for r in roc_dict])
        
        return result_args

eflow_base_lookup = {
    12: 3,
    13: 3,
    35: 4, 
    36: 4, 
}

def ae_train(
    signal_path,
    qcd_path,
    target_dim,
    hlf=True,
    eflow=True,
    version=None,
    seed=40,
    test_split=0.15, 
    val_split=0.15,
    norm_args={
        "norm_type": "StandardScaler"
    },
    train_me=True,
    batch_size=64,
    loss='mse',
    optimizer='adam',
    epochs=100,
    learning_rate=0.0005,
    custom_objects={},
    interm_architecture=(30,30),
):

    """Training function for basic autoencoder (inputs == outputs). 
    Will create and save a summary file for this training run, with relevant
    training details etc.

    Not super flexible, but gives a good idea of how good your standard AE is.
    """
    # set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # get all our data
    (signal,
     signal_jets,
     signal_event,
     signal_flavor) = utils.load_all_data(
        signal_path,
        "signal", include_hlf=hlf, include_eflow=eflow
    )

    (qcd,
     qcd_jets,
     qcd_event,
     qcd_flavor) = utils.load_all_data(
        qcd_path, 
        "qcd background", include_hlf=hlf, include_eflow=eflow
    )

    if eflow:
        qcd_eflow = len(filter(lambda x: "eflow" in x, qcd.columns))
        signal_eflow = len(filter(lambda x: "eflow" in x, signal.columns))

        assert qcd_eflow == signal_eflow, 'signal and qcd eflow basis must be the same!!'
        eflow_base = eflow_base_lookup[qcd_eflow]
    else:
        eflow_base = 0

    filename = "{}{}{}_".format('hlf_' if hlf else '', 'eflow{}_'.format(eflow_base) if eflow else '', target_dim)
    
    if version is None:
        existing_ids = map(lambda x: int(os.path.basename(x).rstrip('.summary').split('_')[-1].lstrip('v')), utils.summary_match(filename + "v*"))
        assert len(existing_ids) == len(set(existing_ids)), "no duplicate ids"
        id_set = set(existing_ids)
        this_num = 0
        while this_num in id_set:
            this_num += 1
        
        version = this_num

    filename += "v{}".format(version)

    assert len(utils.summary_match(filename)) == 0, "filename '{}' exists already! Change version id, or leave blank.".format(filename)

    input_dim = len(signal.columns)

    data_args = {
        'target_dim': target_dim,
        'input_dim': input_dim,
        'test_split': test_split,
        'val_split': val_split,
        'hlf': hlf, 
        'eflow': eflow,
        'eflow_base': eflow_base,
        'seed': seed,
        'filename': filename,
        'filepath': os.path.abspath(filename),
        'qcd_path': qcd_path,
        'signal_path': signal_path,
    }

    all_train, test = qcd.train_test_split(test_split, seed)
    train, val = all_train.train_test_split(val_split, seed)

    train_norm = train.norm(out_name="qcd train norm", **norm_args)
    val_norm = train.norm(val, out_name="qcd val norm", **norm_args)
    
    test_norm = test.norm(out_name="qcd test norm", **norm_args)
    signal_norm = signal.norm(out_name="signal norm", **norm_args)

    train.name = "qcd training data"
    test.name = "qcd test data"
    val.name = "qcd validation data"

    instance = trainer.trainer(filename)

    aes = models.base_autoencoder()
    aes.add(input_dim)
    for elt in interm_architecture:
        aes.add(elt, activation='relu')
    aes.add(target_dim, activation='relu')
    for elt in reversed(interm_architecture):
        aes.add(elt, activation='relu')
    aes.add(input_dim, activation='linear')

    ae = aes.build()
    ae.summary()
    train_args = {
        'batch_size': batch_size, 
        'loss': loss, 
        'optimizer': optimizer,
        'epochs': epochs,
        'learning_rate': learning_rate,
    }

    print "TRAINING WITH PARAMS >>>"
    for arg in train_args:
        print arg, ":", train_args[arg]

    if train_me:
        ae = instance.train(
            x_train=train_norm.data,
            x_test=val_norm.data,
            y_train=train_norm.data,
            y_test=val_norm.data,
            model=ae,
            force=True,
            use_callbacks=True,
            custom_objects=custom_objects, 
            **train_args
        )
    else:
        ae = instance.load_model(custom_objects=custom_objects)

    [data_err, signal_err], [data_recon, signal_recon] = utils.get_recon_errors([test_norm, signal_norm], ae)
    roc_dict = utils.roc_auc_dict(data_err, signal_err, metrics=['mae', 'mse']).values()[0]
    result_args = dict([(r + '_auc', roc_dict[r]['auc']) for r in roc_dict])

    utils.dump_summary_json(result_args, train_args, data_args, norm_args)

    return filename
