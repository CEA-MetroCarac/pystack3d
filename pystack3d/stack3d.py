"""
Class 'Stack3d' used to process stacks from FIB-SEM images.
The class has 'pathdir' attribute wich is the path to the directory where the
data is stored and 'params' attribute which is a dictionary of processing
parameters
"""
import os
import shutil
import sys
import time
from importlib import import_module
from pathlib import Path
from threading import Thread
from multiprocessing import Pool, Queue

import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile, TiffWriter, imwrite
from tomli import load

from pystack3d.utils import imread_3d_skipping, get_tags, dumps_params
from pystack3d.utils_multiprocessing import worker_init, step_wrapper, initialize_args

PROCESS_STEPS = ['cropping', 'bkg_removal',
                 'intensity_rescaling', 'intensity_rescaling_area',
                 'registration_calculation', 'registration_transformation',
                 'destriping', 'resampling', 'cropping_final']
CMAP = plt.get_cmap("tab10")
ASSETS = os.path.join(__file__, "..", "assets")

plt.rcParams['savefig.dpi'] = 300


class Stack3d:
    """
    Class dedicated to the Stack3d workflow

    Attributes
    ----------
    pathdir: Path
        Path related to the dirname where the .tif images are stored with the
        related metadata
    last_step_dir: str
        Dirname related to the last step of the workflow execution.
        For the first step, 'last_step_dir' corresponds to 'pathdir'
    fname_toml: str
        Filename of the '.toml' file defining the workflow parameters
    params: dict
        Dictionary defining the workflow parameters
    queue_incr: Queue object
        Queue used to increment the progress bar during the workflow execution

    Arguments
    ---------
    input_name: str, optional
        Pathname related to the project dirname where the .tif images are stored
         OR the filename related to the .toml file defining all the workflow
         parameters (requiring to set the project 'dirname' parameter inside).
        If None, consider the directory where the python script has been
        launched and the .toml and .tif files located inside.
    ignore_error: bool, optional
        Ignore IOError in some case associated with the pystack3d-napari appli launching
    queue_incr: Queue object, optional
        Possibility to pass the Queue used by the progress bar
    """

    def __init__(self, input_name=None, ignore_error=False, queue_incr=None):

        self.fname_toml = None
        self.params = {'history': '', 'ind_min': 0, 'ind_max': 99999}
        self.queue_incr = queue_incr or Queue()

        # Single big .tif file processing
        if str(input_name).endswith('.tif'):
            fname = Path(input_name)
            dirname_out = fname.parent / 'images'
            os.makedirs(dirname_out, exist_ok=True)
            with TiffFile(fname) as tif:
                for i, page in enumerate(tif.pages):
                    img = page.asarray()
                    imwrite(dirname_out / f"img_{i:03d}.tif", img)
            self.pathdir = fname.parent
            self.params["channels"] = ['images']
            self.last_step_dir = self.pathdir
            return

        elif str(input_name).endswith('.toml'):
            self.fname_toml = input_name
            with open(self.fname_toml, 'rb') as fid:
                self.params = load(fid)
                self.pathdir = Path(self.params["dirname"])

        elif input_name is None or os.path.isdir(input_name):
            input_name = input_name or os.getcwd()
            self.pathdir = Path(input_name)
            fnames = list(self.pathdir.glob("*.toml"))
            if len(fnames) == 0:
                src = Path(ASSETS) / 'params.toml'
                dst = self.pathdir / 'params.toml'
                shutil.copy2(src, dst)
                msg = "\n***************************************************\n"
                msg += "No '.toml' file has been found in {}\n"
                msg += "A default '.toml' file has been put in your directory\n"
                msg += "You have now to adapt the parameters values inside"
                msg += "\n***************************************************\n"
                if ignore_error:
                    fnames = [dst]
                else:
                    raise IOError(msg.format(input_name))
            if len(fnames) > 1:
                msg = "\n***************************************************\n"
                msg += "More than 1 '.toml' file have been found in {}\n"
                msg += "Please, select and pass to Stack3d(input_name='...') " \
                       "the one you want to work with\n"
                msg += "\n***************************************************\n"
                raise IOError(msg.format(input_name))
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

    def channels(self, process_step):
        """ Return the list of channels according to the 'process_step' """
        channels = self.params["channels"]
        if process_step == 'registration_calculation':
            channels = [channels[0]]
        return channels

    def process_dirname(self, process_step, channel):
        """ Return the process dirname wrt to 'process_step' and 'channel' """
        if process_step == 'input':
            process_dirname = self.pathdir / channel
        elif process_step == 'registration_calculation':
            process_dirname = self.pathdir / 'process' / process_step
        else:
            process_dirname = self.pathdir / 'process' / process_step / channel
        return process_dirname

    def fnames(self, input_dirname):
        """ Return the .tif filenames related to the input_dirname """
        fnames = sorted(input_dirname.glob('*.tif'))

        # apply slices reduction to the 1rst process step
        if len(self.params['history']) == 0:
            ind_min, ind_max = self.params['ind_min'], self.params['ind_max']
            fnames = fnames[ind_min:ind_max + 1]

        return fnames

    def shape(self, fnames):
        """ Return the shape of the stack """
        with TiffFile(fnames[0]) as tiff:
            shape = tiff.pages[0].shape
        return tuple((len(fnames), shape[0], shape[1]))

    def overlay(self, process_step):
        """ Return the overlay related to 'process_step'"""
        return 1 if process_step in ['registration_calculation', 'resampling'] else 0

    def create_partition(self, fnames, nproc, overlay=0):
        """
        Return a filenames partition related to a multiprocessing execution

        Parameters
        ----------
        fnames: list of str or list of Path
            List of the input filenames to consider
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
        nslices = len(fnames)
        msg = f'nproc ({nproc}) is not suited for ({nslices}) slices\n'
        msg += '(should be strictly less)\n'
        assert nslices >= nproc, msg

        # define pairs of start and end indexes for each thread
        nbel = nslices / nproc
        ind_list = list(range(nslices))
        pairs = [(int(i * nbel), int((i + 1) * nbel)) for i in range(nproc)]
        inds_part = [ind_list[imin:imax + overlay] for (imin, imax) in pairs]

        fnames_part = []
        for inds in inds_part:
            fnames_part += [[fnames[ind] for ind in inds]]

        return fnames_part, inds_part

    def eval(self, process_steps=None, nproc=None, serial=True, show_pbar=True, pbar_init=False):
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
        pbar_init: bool; optional
            Activation key to pass 'ntot' as 1rst queue_incr.put().
            (tricky mode for the appli to catch this data via the Queue)
        """
        dir_process = self.pathdir / 'process'

        if process_steps is None:
            process_steps = self.params['process_steps']

        if isinstance(process_steps, str):
            process_steps = [process_steps]

        for k, x in enumerate(process_steps):
            assert x in PROCESS_STEPS, f"'{x}' is not a available process step"
            assert x not in process_steps[:k], f"'{x}' is called several times"

        if nproc is None:
            nproc = self.params['nproc']
        self.params['nproc'] = nproc

        assert isinstance(nproc, int) and nproc > 0, 'nproc not a positive int'

        # process step calculation
        for process_step in process_steps:

            history = self.params['history']
            if process_step in history:
                print(f"'{process_step}' has already been processed")
                continue

            # input directory
            last_step_dir = self.pathdir  # default input directory
            if len(history) > 0:
                if history[-1] != 'registration_calculation':
                    last_step_dir = dir_process / history[-1]
                elif len(history) > 1:
                    last_step_dir = dir_process / history[-2]

            for channel in self.channels(process_step):

                print(process_step, (channel != '.') * f"channel {channel}")

                input_dirname = last_step_dir / channel
                output_dirname = self.process_dirname(process_step, channel)
                fnames = self.fnames(input_dirname)  # .tif filenames to handle
                shape = self.shape(fnames)  # stack shape

                # ouput_dirname cleaning-up
                os.makedirs(output_dirname, exist_ok=True)
                shutil.rmtree(output_dirname)
                os.makedirs(output_dirname / 'outputs')

                # process kwargs (to be overwritten)
                kwargs = self.params[process_step].copy()
                kwargs.update({'output_dirname': output_dirname})
                kwargs.update({'fnames': fnames})  # req. for 'resampling' init

                # create partition
                overlay = self.overlay(process_step)
                fnames_parts, inds_parts = self.create_partition(fnames, nproc, overlay)

                args = initialize_args(process_step, kwargs, nproc, shape)
                worker_args = (self.queue_incr, *args)
                pbar_args = (self.queue_incr, len(fnames), overlay, nproc)

                if pbar_init:
                    ntot = len(fnames) + (nproc - 1) * overlay
                    self.queue_incr.put(ntot)

                if nproc == 1:

                    kwargs.update({'fnames': fnames_parts[0]})
                    kwargs.update({'inds_partition': inds_parts[0]})

                    worker_init(*worker_args)

                    if show_pbar:
                        Thread(target=pbar_update, args=pbar_args).start()

                    step_wrapper(process_step, kwargs)

                else:

                    args = []
                    for fnames, inds in zip(fnames_parts, inds_parts):
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

                plot(process_step, output_dirname, input_dirname, kwargs)

            # 'history' parameter updating and saving
            if serial and self.fname_toml:
                self.params['history'] = self.params['history'] + [process_step]
                self.fname_toml.write_text(dumps_params(self.params), encoding='utf-8')

    def cleanup(self):
        """ Remove the .tif files except the ones related to the last process"""
        if len(self.params['history']) >= 2:
            for process_step in self.params['history'][:-1]:
                for channel in self.channels(process_step):
                    dirname = self.process_dirname(process_step, channel)
                    for fname in dirname.glob("*.tif"):
                        fname.unlink()

    def concatenate_tif(self, process_step=None, save_metadata=True,
                        dirname_out=None, name_out='big.tif'):
        """
        Concatenate .tif files issued from a process step into a single one

        Parameters
        ----------
        process_step: str, optional
            Process step name to handle.
            If 'input' concatenate the input .tif files
            If None (default) concatenate the .tif files related to the last
            process execution extracted from the 'history' in the .toml file
        save_metadata: bool, optional
            Activation key to save metadata in the .tif file
        dirname_out: str or WindowsPath, optional
            Dirname associated to the big .tif file saving.
            If None, consider the same dirname as the input dirname related to
            the 'process_step'
        name_out: str, optional
            Name of the big .tif file
        """
        if isinstance(dirname_out, str):
            dirname_out = Path(dirname_out)

        valid_process_step = [None, 'input'] + PROCESS_STEPS
        assert process_step in valid_process_step, "process_step is not correct"

        if process_step is None:
            if len(self.params['history']) > 0:
                process_step = self.params['history'][-1]
            else:
                raise IOError("No process step in the 'history'")

        for channel in self.channels(process_step):

            print(f"concatenate - {process_step}",
                  (channel != '.') * f"channel {channel}")

            dirname_in = self.process_dirname(process_step, channel)
            dirname_out = dirname_out or dirname_in

            fname_out = dirname_out / name_out
            fnames_in = sorted(list(dirname_in.glob('*.tif')))
            if fname_out in fnames_in:
                fnames_in.remove(fname_out)

            pbar_args = (self.queue_incr, len(fnames_in))
            thread = Thread(target=pbar_update, args=pbar_args)
            thread.start()

            with TiffWriter(dirname_out / name_out, bigtiff=True) as tiff_out:
                for fname in fnames_in:
                    self.queue_incr.put(1)
                    with TiffFile(fname) as tiff_in:
                        arr = tiff_in.asarray()
                        tags, extra_tags = get_tags(tiff_in)
                        extra_tags = extra_tags if save_metadata else None
                        tiff_out.write(arr, extratags=extra_tags,
                                       compression=tags[259].value)
                self.queue_incr.put('finished')

            thread.join()


def plot(process_step, output_dirname, input_dirname, kwargs):
    """ Plot statistics and specific values related to the 'process_step' """
    fname = output_dirname / 'outputs' / 'stats.npy'
    if os.path.isfile(fname):
        stats = np.load(fname)
        labels = ['Min', 'Max', 'Mean']
        sfx = [' (input)', ' (output)', ' (reformatted output)']
        fig, ax = plt.subplots(3, 1, figsize=(8, len(labels) * 2.5),
                               gridspec_kw={'hspace': 0.6})
        fig.canvas.manager.set_window_title(process_step)
        for k, label in enumerate(labels):
            ax[k].plot(stats[:, 0, k], c=CMAP(0), label=label + sfx[0])
            ax[k].plot(stats[:, 1, k], c=CMAP(1), label=label + sfx[1], ls='--')
            ax[k].plot(stats[:, 2, k], c=CMAP(1), label=label + sfx[2])
            ax[k].legend(loc=9, ncols=3, bbox_to_anchor=(0.5, 1.3))

        ax[-1].set_xlabel('# Frames', labelpad=-1)
        plt.savefig(output_dirname / 'outputs' / 'stats.png')
        plt.close(fig)

    if process_step == 'bkg_removal':
        plot_stats_xy(input_dirname, output_dirname,
                      skip_factors=kwargs['skip_factors'])

    if process_step == 'cropping_final':
        process_step = 'cropping'

    module = import_module(f"pystack3d.{process_step}")
    if hasattr(module, 'plot'):
        getattr(module, "plot")(output_dirname)


def plot_stats_xy(input_dirname, output_dirname, skip_factors=(10, 10, 10)):
    """ Plot statistics in x et y directions considering 'skip_factors' when
        loading the full 3D stack """

    labels = ['Min', 'Max', 'Mean']
    sfx = [' (input)', ' (output)']
    axis_labels = ['X', 'Y']

    figs = []
    axes = []
    for axis in axis_labels:
        fig, ax = plt.subplots(3, 1, figsize=(8, len(labels) * 2.5),
                               gridspec_kw={'hspace': 0.6})
        fig.canvas.manager.set_window_title(f"stats_{axis}")
        figs.append(fig)
        axes.append(ax)

    for i, dirname in enumerate([input_dirname, output_dirname]):
        fnames = sorted(Path(dirname).glob("*.tif"))
        arr_3d = imread_3d_skipping(fnames, skip_factors=skip_factors)
        arr_3d = np.swapaxes(arr_3d, 0, 1)
        arr_3d = np.swapaxes(arr_3d, 1, 2)
        arr_3d = arr_3d[::-1, :, :]

        for axis in range(2):
            stats = [np.min(arr_3d, axis=(axis, 2)),
                     np.max(arr_3d, axis=(axis, 2)),
                     np.mean(arr_3d, axis=(axis, 2))]

            for k, label in enumerate(labels):
                x = skip_factors[axis] * np.arange(len(stats[k]))
                axes[axis][k].plot(x, stats[k], c=CMAP(i), label=label + sfx[i])
                axes[axis][k].legend(loc=9, ncols=3, bbox_to_anchor=(0.5, 1.05))
            axes[axis][-1].set_xlabel(axis_labels[axis], labelpad=-1)

    for i, fig in enumerate(figs):
        fig.savefig(output_dirname / 'outputs' / f"stats_{axis_labels[i]}.png")
        plt.close(fig)


def pbar_update(queue_incr, nslices, overlay, nproc):
    """ Progress bar """
    overlays = (nproc - 1) * overlay
    ntot = nslices + overlays
    pbar = "\r[{:50}] {:.0f}% {:.0f}/{} {:.2f}s " + f"ncpus={nproc}"
    if overlays != 0:
        pbar += f" ({(nproc - 1) * overlay} overlay{(overlays > 1) * 's'})"
    count = 0
    finished = 0
    t0 = time.time()
    while finished < nproc:
        val = queue_incr.get()
        if val == 'finished':
            finished += 1
        else:
            count += val
            percent = 100 * count / ntot
            cursor = "*" * int(percent / 2)
            exec_time = time.time() - t0
        sys.stdout.write(pbar.format(cursor, percent, count, ntot, exec_time))
    print()
