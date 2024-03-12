"""
Class 'Stack3d' used to process stacks from FIB-SEM images.
The class has 'pathdir' attribute wich is the path to the directory where the
data is stored and 'params' attribute which is a dictionary of processing
parameters
"""
import os
import shutil
import sys
import glob
import time
from importlib import import_module
from pathlib import Path
from threading import Thread
from multiprocessing import Pool, Queue

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from tomli import load
from tomlkit import dump

from pystack3d.utils_multiprocessing import (worker_init, step_wrapper,
                                             initialize_args)

PROCESS_STEPS = ['cropping', 'bkg_removal', 'intensity_rescaling',
                 'registration_calculation', 'registration_transformation',
                 'destriping', 'resampling', 'cropping_final']
CMAP = plt.get_cmap("tab10")

VERSION = "2024.1"


class Stack3d:
    """
    Class dedicated to Stack3d workflow management

    Attributes
    ----------
    pathdir: Path
        Path related to the dirname where the .tif images are stored with the
        related metadata
    last_step_dir: str
        Dirname related to the last step of the workflow execution.
        For the first step, 'last_step_dir' corresponds to 'pathdir'

    Arguments
    ---------
    pathdir: Path, optional
        Path related to the dirname where the .tif images are stored with the
        related metadata.
        If None, consider the directory where the python script has been
        executed
    """

    def __init__(self, input_name=None):

        if str(input_name).endswith('.toml'):
            self.fname_toml = input_name
            with open(self.fname_toml, 'rb') as fid:
                self.params = load(fid)
                self.pathdir = self.pathdir["dirname"]

        elif os.path.isdir(input_name) or input_name is None:
            input_name = input_name or '.'
            self.pathdir = Path(input_name)
            fnames = list(self.pathdir.glob("*.toml"))
            if len(fnames) == 0:
                raise IOError(f"there is no '.toml' file in {input_name}")
            elif len(fnames) > 1:
                raise IOError(f"there is too much '.toml' file in {input_name}")
            else:
                self.fname_toml = fnames[0]

            with open(self.fname_toml, 'rb') as fid:
                self.params = load(fid)
                fname = self.fname_toml.name
                print("\n***************************************************")
                print(f"WARNING: 'dirname' from {fname} is NOT USED")
                print("***************************************************\n")
        else:
            raise IOError(f"'input_name' {input_name} is not a valid input")

        if self.params["channels"] in [None, '']:
            self.params["channels"] = ['.']
        self.last_step_dir = self.pathdir

    def __str__(self):
        msg = f"Pathdir = {self.pathdir}\n"
        msg += f"Channels = {self.params['channels']}"
        return msg

    def create_partition(self, input_dirname, nproc, overlay=0):
        """
        Return a filenames partition related to a multiprocessing execution

        Parameters
        ----------
        input_dirname: str or Path
            Pathname of the input filenames to consider
        nproc: int
            Number of cpus to consider to create the partition
        overlay: int, optional
            Number of overlaid frames to ensure continuity in the processing

        Returns
        -------
        fnames_part: list of lists of str
            List of filenames partitions
        ind_part: list of lists of int
            List of indexes partitions
        """
        fnames = sorted(input_dirname.glob('*.tif'))

        # apply slices reduction to the 1rst process step
        if len(self.params['history']) == 0:
            ind_min, ind_max = self.params['ind_min'], self.params['ind_max']
            fnames = fnames[ind_min:ind_max + 1]

        nslices = len(fnames)
        msg = f'nproc ({nproc}) is not suited for ({nslices}) slices\n'
        assert nslices >= nproc, msg

        # define pairs of start-end indexes for each thread
        inds = list(range(nslices))
        nbel = int(len(inds) / nproc)
        pairs = [(i * nbel, (i + 1) * nbel - 1) for i in range(nproc)]
        pairs[-1] = ((nproc - 1) * nbel, inds[-1])

        inds_part = [inds[max(0, imin): min(inds[-1], imax + overlay) + 1]
                     for (imin, imax) in pairs]

        fnames_part = []
        for inds in inds_part:
            fnames_part += [[fnames[ind] for ind in inds]]

        return fnames_part, inds_part, nslices

    def eval(self, process_steps=None, nproc=None, serial=True, show_pbar=True):
        """
        Method to apply a process step to the stack object

        Parameters
        ----------
        process_steps: list of str or str
            Name(s) of the processing step(s) to apply to the stack
            If None: 'process_steps' is defined from 'params'
        nproc: int, optional
            Number of cpus to use.
            If None: 'nproc' is defined from 'params'
            If nproc=0: the workflow is executed without calculation
        serial: bool, optional
            Activation key to consider a serialized workflow. If False, each
            process takes data located in the 'input' data folder
        show_pbar: bool, optional
            Activation key to display the progress bar during the processing
        """
        dir_process = self.pathdir / 'process'
        history = self.params['history']

        if process_steps is None:
            process_steps = self.params['process_steps']

        if isinstance(process_steps, str):
            process_steps = [process_steps]

        for x in process_steps:
            assert x in PROCESS_STEPS, f"'{x}' is not a available process step"
            assert x not in history, f"'{x}' has already been processed"

        if nproc is None:
            nproc = self.params['nproc']

        assert type(nproc) is int and nproc > 0, 'nproc not a positive int'

        # process step calculation
        for process_step in process_steps:

            history = self.params['history']

            # input directory
            last_step_dir = self.pathdir  # default input directory
            if len(history) > 0:
                if history[-1] != 'registration_calculation':
                    last_step_dir = dir_process / history[-1]
                elif len(history) > 1:
                    last_step_dir = dir_process / history[-2]

            channels = self.params["channels"]
            if process_step == 'registration_calculation':
                channels = [channels[0]]

            for channel in channels:

                # process kwargs (to be overwritten))
                kwargs = self.params[process_step].copy()

                print(process_step, (channel != '.') * f"channel {channel}")

                input_dirname = last_step_dir / channel

                if process_step == 'registration_calculation':
                    output_dirname = dir_process / process_step
                else:
                    output_dirname = dir_process / process_step / channel

                kwargs.update({'output_dirname': output_dirname})

                # ouput_dirname cleaning-up
                os.makedirs(output_dirname, exist_ok=True)
                shutil.rmtree(output_dirname)
                os.makedirs(output_dirname / 'outputs')

                # create partition
                overlay = 0
                if process_step in ['registration_calculation', 'resampling']:
                    overlay = 1
                out = self.create_partition(input_dirname, nproc, overlay)
                fnames_part, inds_part, nslices = out
                kwargs.update({'fnames': fnames_part})

                queue_incr = Queue()
                args = initialize_args(process_step, kwargs, nproc, nslices)
                worker_args = (queue_incr, *args)
                pbar_args = (queue_incr, nslices + (nproc - 1) * overlay, nproc)

                if nproc == 1:

                    kwargs.update({'fnames': fnames_part[0]})
                    kwargs.update({'inds_partition': inds_part[0]})

                    worker_init(*worker_args)

                    if show_pbar:
                        Thread(target=pbar_update, args=pbar_args).start()

                    step_wrapper(process_step, kwargs)

                else:

                    args = []
                    for fnames, inds in zip(fnames_part, inds_part):
                        kwargs.update({'fnames': fnames})
                        kwargs.update({'inds_partition': inds})
                        args += [[process_step, {**kwargs}]]

                    with Pool(nproc,
                              initargs=worker_args,
                              initializer=worker_init) as pool:
                        results = pool.starmap_async(step_wrapper, args)
                        if show_pbar:
                            pbar_update(*pbar_args)
                        results.wait()
                    results.get()

                plot(process_step, output_dirname)

            # 'history' parameter updating and saving
            if serial:
                self.params['history'] = self.params['history'] + [process_step]
                with open(self.fname_toml, 'w') as fid:
                    dump(self.params, fid)


def plot(process_step, output_dirname):
    """ Plot statistics and specific values related to the 'process_step' """
    fname = output_dirname / 'outputs' / 'stats.npy'
    if os.path.isfile(fname):
        stats = np.load(fname)
        labels = ['Min', 'Max', 'Mean']
        sfx = [' (input)', ' (output)', ' (reformatted output)']
        fig, ax = plt.subplots(3, 1, figsize=(8, len(labels) * 2))
        fig.canvas.manager.set_window_title(process_step)
        fig.tight_layout()
        for k, label in enumerate(labels):
            ax[k].plot(stats[:, 0, k], c=CMAP(0), label=label + sfx[0])
            ax[k].plot(stats[:, 1, k], c=CMAP(1), label=label + sfx[1], ls='--')
            ax[k].plot(stats[:, 2, k], c=CMAP(1), label=label + sfx[2])
            ax[k].legend(loc=9, ncols=3)
        ax[-1].set_xlabel('# Frames', labelpad=-1)
        plt.savefig(output_dirname / 'outputs' / 'stats.png')

    if process_step == 'cropping_final':
        process_step = 'cropping'

    module = import_module(f"pystack3d.{process_step}")
    if hasattr(module, 'plot'):
        getattr(module, "plot")(output_dirname)


def pbar_update(queue_incr, nslices, nproc):
    """ Progress bar """
    percent = 0
    finished = 0
    pbar = "\r[{:{}}] {:.0f}% {:.2f}s"
    t0 = time.time()
    while finished < nproc:
        val = queue_incr.get()
        if val == 'finished':
            finished += 1
        else:
            percent += (val * 100 / nslices)
            t1 = time.time()
        sys.stdout.write(pbar.format("*" * int(percent), 100, percent, t1 - t0))
    print()
