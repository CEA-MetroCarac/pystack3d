"""
utilities functions to extract metadata from FIB-SEM acquisitions in the
.a3d-setup parameters file and a .tif file (corresponding to the first slice)
"""
import pprint
from pathlib import Path
import numpy as np
from tomlkit import load, dump
import tifffile
from lxml.etree import tostring, fromstring, parse, XMLPullParser

ASSETS = Path(__file__).parent / "assets"

PARAMS = {
    'JobName': 'JobName',
    'AutoStigROILeft': 'Settings/AutoTune/AutoStigAndFocus/ROILeft',
    'AutoStigROIRight': 'Settings/AutoTune/AutoStigAndFocus/ROIRight',
    'AutoStigROITop': 'Settings/AutoTune/AutoStigAndFocus/ROITop',
    'AutoStigROIBottom': 'Settings/AutoTune/AutoStigAndFocus/ROIBottom',
    'DetectorA': 'Settings/Imaging/Detector',
    'DetectorB': 'Settings/Imaging/DetectorB',
    'Fov': 'Settings/Imaging/DefaultAcquisition/FOV',
    'PixelSize': 'Settings/Imaging/DefaultAcquisition/PixelSize',
    'SliceThickness': 'Settings/Imaging/DefaultAcquisition/Interval/Distance'
                      '/Interval',
    'ImageROILeft': 'Settings/ImagingXROI/FIBShape/Nodes/Node/X',
    'ImageROIRight': 'Settings/ImagingXROI/FIBShape/Nodes/Node/Y',
    'ImageROITop': 'Settings/ImagingXROI/FIBShape/Nodes/Node/X',
    'ImageROIBottom': 'Settings/ImagingXROI/FIBShape/Nodes/Node/Y',
    'Width': 'Image/Width',
    'Height': 'Image/Height',
    'BoundingBoxLeft': 'Image/BoundingBox.Left',
    'BoundingBoxRight': 'Image/BoundingBox.Right',
    'BoundingBoxTop': 'Image/BoundingBox.Top',
    'BoundingBoxBottom': 'Image/BoundingBox.Bottom',
    'ZIndex': 'ATLAS3D/Slice/ZIndex',
    'ZPos': 'ATLAS3D/Slice/ZPos',
    'SEMCorrectionX': 'ATLAS3D/Slice/SEMCorrectionX',
    'SEMCorrectionY': 'ATLAS3D/Slice/SEMCorrectionY',
    'ExtI': 'BeamInfo/item[@name="Ext I"]'
}


def params_from_metadata(stack_dir,
                         fname_toml_ref=None, save=False, verbosity=True):
    """
    Return processing parameters for a 3D FIB-SEM acquisition from metadata
    extracted from raw data directory

    Parameters
    ----------
    stack_dir: Path or str
        Pathname of the acquisition folder. Must contain the Atlas3D.a3d-setup
        file and slices as TIF in subdirectories for each channel
    fname_toml_ref: Path or str, optional
        Pathname of the .toml file taken as reference for the parameters file.
        If None, consider the RAW params.toml located in the 'assets' folder
    save: bool, optional
        Key to save a 'params_from_metadata.toml' in the 'stack_dir'
    verbosity: bool, optional, default True
        Activation key for verbosity displaying

    Returns
    -------
    params: dict
        Dictionary with parameters for the Stack3d FIB-SEM data processing
    """
    stack_dir = Path(stack_dir)

    if fname_toml_ref is None:
        fname_toml_ref = ASSETS / 'params.toml'

    with open(fname_toml_ref, 'r') as fid:  # pylint: disable=W1514
        params = load(fid)

    # extract metadata from setup file issued from atlas engine acquisition
    xml_ETroot = read_tags(stack_dir / 'Atlas3D.a3d-setup')
    channels = []
    for X in ['A', 'B']:
        channels += [param_from_xml_ETroot(f'Detector{X}', xml_ETroot)]
    pxsize = float(param_from_xml_ETroot('PixelSize', xml_ETroot))
    fov = int(float(param_from_xml_ETroot('Fov', xml_ETroot)) / pxsize)
    dz = float(param_from_xml_ETroot('SliceThickness', xml_ETroot))

    # extract metadata from the first slice (.tif) of the first channel
    channel_directory = stack_dir / channels[0]
    fnames = sorted(channel_directory.glob('*.tif'))
    if len(fnames) == 0:
        msg = f'channel directory {channel_directory} contains no .tif file'
        raise ValueError(msg)
    fname_xml = stack_dir / 'first_slice_metadata.xml'
    xml_ETroot = read_tags(fnames[0], fname_xml=fname_xml)

    if verbosity:
        print(f'Metadata from first slice saved in {stack_dir}')

    shape = (int(param_from_xml_ETroot('Width', xml_ETroot)),
             int(param_from_xml_ETroot('Height', xml_ETroot)))

    msg = 'First slice is not square (but should be)'
    assert shape[0] == shape[1], msg

    msg = 'FOV and size of first slice are not equal (but should be)'
    assert shape[0] == fov or shape[0] == fov + 1, msg

    xmin = int(param_from_xml_ETroot('BoundingBoxLeft', xml_ETroot))
    xmax = int(param_from_xml_ETroot('BoundingBoxRight', xml_ETroot))
    ymin = fov - int(param_from_xml_ETroot('BoundingBoxBottom', xml_ETroot))
    ymax = fov - int(param_from_xml_ETroot('BoundingBoxTop', xml_ETroot))

    # update params
    params["channels"] = tuple(channels)
    params['cropping']['area'] = [xmin - 20, xmax + 20, ymin - 20, ymax + 20]
    params['resampling']['dz'] = dz

    if save:
        fname_toml = stack_dir / 'params_from_metadata.toml'
        with open(fname_toml, "w") as fid:  # pylint: disable=W1514
            dump(params, fid)

    return params


def currents_from_metadata(fnames):
    """
    Return currents from metadata

    Parameters
    ----------
    fnames: list of n-str
        List of the '.tif' filenames

    Returns
    -------
    currents: np.ndarray((n))
        Currents values extracted from the metadata
    """
    currents = []
    for fname in fnames:
        xml_ETroot = read_tags(fname)
        current = param_from_xml_ETroot('ExtI', xml_ETroot).split(' ')[0]
        currents.append(float(current))

    currents = np.asarray(currents)

    return currents


def read_tags(fname, print_tags=False, fname_xml=None, encoding="iso-8859-1"):
    """
    Extract metadata from an image made with fibics Atlas Engine

    Parameters
    ----------
    fname: Path or str
        Pathname of the .tif file (or parameter file) to read metadata from
    print_tags: Bool, optional
        key to control metadata print as they are read. Default False
    fname_xml: Path or str, optional
        Pathname to save the resulting xml tags file
    encoding: str, optional
        encoding of the file to read

    Returns
    -------
    ETroot : root of the ElementTree object resulting from the xml parsing
    """
    # Pretty printer for nested list display in console or file
    ppp = pprint.PrettyPrinter(indent=2, depth=4)
    # Parser for xml file reading
    parser = XMLPullParser(encoding=encoding, remove_blank_text=True)

    # Read file
    extension = Path(fname).suffix
    if extension == '.tif':
        with tifffile.TiffFile(fname) as fid:
            # extracting string containing all fibics info
            # pylint: disable=E1136  # pylint/issues/3139
            xml_str = fid.series[0].keyframe.tags[51023].value
            # removing the encoding declaration
            xml_str = xml_str.split('\n')[1]
            # converting this string to an ElementTree object
            xml_etree = fromstring(xml_str, parser=parser).getroottree()
    elif extension == '.a3d-setup':
        with open(fname, encoding=encoding) as fid:
            xml_etree = parse(fid, parser=parser)

    else:
        msg = "Metadata extraction from a .{} file has not been implemented"
        raise NotImplementedError(msg.format(extension))

    # print to console
    if print_tags:
        ppp.pprint(tostring(xml_etree, pretty_print=True))

    # saving metadata to an xml file
    if fname_xml is not None:
        with open(fname_xml, "wb") as fid:
            xml_str = tostring(xml_etree, pretty_print=True)
            fid.write(xml_str)

    return xml_etree.getroot()


def param_from_xml_ETroot(param_name, ETroot_from_xml, full_output=False):
    """
    Extract a parameter from a lxml.Element object using the xpath of the
    parameters stored in a predefined dictionary.

    Parameters
    ----------
    param_name: str
        Name of the parameter to extract as defined in the dictionnaries in
        this function
    ETroot_from_xml: lxml.Element object
        lxml.Element object parsed from an xml string or file as returned by
        read_tags()
    full_output: bool, optional
        Activation key to return value AND tag

    Returns
    -------
    value: str
        Value of the parameter (as a string)
    tag: str, optional
        Parameter name in the original XML (if full_output is True)
    """
    # Corrections to ensure compatibility with previous Atlas3D-Setup versions
    if ETroot_from_xml.attrib == {}:
        PARAMS.update(
            {'PixelSize': 'Settings/Imaging/FibicsRasterInfo/PixelSizeX',
             'Fov': 'Settings/Imaging/FOV',
             'SliceThickness': 'Settings/Imaging/Interval',
             })

    # xpath fix in the case where several sub-nodes have the same name.
    # selection from the rank order (not the optimal solution. To revisit later)
    rank = 2 if param_name in {"imageROIRight", "imageROIBottom"} else 0
    param = ETroot_from_xml.findall(PARAMS[param_name])[rank]
    value, tag = param.text, param.tag

    if full_output:
        return value, tag
    else:
        return value
