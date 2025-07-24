"""
utilities functions
"""
import numpy as np
from tifffile import TiffFile, TiffWriter
from tomlkit import dumps, table, document, inline_table, array


def reformat_params(params):
    """ Return tomlkit.document from 'params' dict """
    doc = document()
    for section_name, section_data in params.items():
        if isinstance(section_data, dict):
            section = table()
            for key, value in section_data.items():
                value = '' if value is None else value
                if section_name == "destriping" and key == "filters":
                    inline_array = array()
                    inline_array.multiline(False)
                    for filt in value:
                        t = inline_table()
                        t.update(filt)
                        inline_array.append(t)
                    section[key] = inline_array
                else:
                    section[key] = value
            doc[section_name] = section
        else:
            doc[section_name] = section_data
    return doc


def dumps_params(params):
    """ Return cleaned dumps 'params' """

    text = dumps(reformat_params(params))

    lines = text.splitlines(keepends=True)
    while lines and lines[-1].strip() == "":
        lines[-1] = lines[-1].replace('\r', '')
        if lines[-1].strip() == "" and not lines[-1].endswith('\n'):
            lines.pop()

    return ''.join(lines)


def mask_creation(arr, threshold_min=None, threshold_max=None):
    """ Return mask with true values in range [threshold_min, threshold_max] """
    mask = np.ones_like(arr, dtype=bool)
    if threshold_min is not None:
        mask *= arr >= threshold_min
    if threshold_max is not None:
        mask *= arr <= threshold_max
    return mask


def outputs_saving(output_dirname, fname, img, img_res, stats):
    """ Append stats and save img_res after reformatting """
    img_res2 = img_reformatting(img_res, img.dtype)

    stats.append([[img.min(), img.max(), img.mean()],
                  [img_res.min(), img_res.max(), img_res.mean()],
                  [img_res2.min(), img_res2.max(), img_res2.mean()]])

    save_tif(img_res2, fname, output_dirname / fname.name)


def get_tags(fid, extract_extra_tags=True):
    """ Return tags and extra_tags from a TiffFile """
    tags = fid.pages[0].tags
    extra_tags = None
    if extract_extra_tags:
        private_tag_codes = tuple(int(k) for k in tags.keys() if k > 32768)
        extra_tags = tuple(tags[k].astuple() for k in private_tag_codes)
    return tags, extra_tags


def save_tif(arr, fname, fname_out):
    """ Save arr in a 'fname_out' .tif file preserving the 'fname' metadata """
    with TiffFile(fname) as fid:
        tags, extra_tags = get_tags(fid)

    compression = tags[259].value if 259 in tags else None
    with TiffWriter(fname_out) as fid:
        fid.write(arr, extratags=extra_tags, compression=compression)


def img_reformatting(arr, dtype):
    """ Reformat the array in the machine limits related to the data type """
    try:
        info = np.iinfo(dtype)
    except:
        info = np.finfo(dtype)
    return np.clip(arr, info.min, info.max).astype(dtype)


def imread_3d_skipping(fnames, skip_factors=(10, 10, 10)):
    """ Return a 3D-Array from .tif files wrt skip factors in all directions """
    with TiffFile(fnames[0]) as tiff:
        arr0 = tiff.asarray()
    shape = arr0.shape
    dtype = arr0.dtype

    arr_red = np.zeros((int(np.ceil(len(fnames) / skip_factors[2])),
                        int(np.ceil(shape[0] / skip_factors[1])),
                        int(np.ceil(shape[1] / skip_factors[0]))), dtype=dtype)

    for k, fname in enumerate(fnames[::skip_factors[2]]):
        with TiffFile(fname) as tiff:
            img = tiff.asarray()
        arr_red[k, ...] = img[::skip_factors[1], ::skip_factors[0]]

    return arr_red


def skipping(arr, skip_factors):
    """ Skip the array in each axis """
    arr_red = arr.copy()
    for k in range(arr_red.ndim):
        arr_red = np.swapaxes(arr_red, 0, k)
        arr_red = arr_red[::skip_factors[k], ...]
        arr_red = np.swapaxes(arr_red, 0, k)
    return arr_red


def division(num):
    """ Return the highest dividers pair. Ex: 6 -> (3, 2), 12 -> (4, 3) """
    for i in reversed(range(1, int(num ** 0.5) + 1)):
        if num % i == 0:
            return num // i, i


def cumdot(mat):
    """ Return the cumulative matrix product """
    mat = np.asarray(mat)
    cum_mat = np.empty(mat.shape)
    cum_mat[0] = mat[0]
    for i in range(1, mat.shape[0]):
        cum_mat[i] = cum_mat[i - 1] @ mat[i]
    return cum_mat


if __name__ == "__main__":
    import tempfile
    from tomli import load
    from pystack3d import ASSETS

    fname_toml = Path(ASSETS) / "params.toml"
    fname_toml_out = Path(tempfile.gettempdir()) / "params_out.toml"

    with open(fname_toml, 'rb') as fid:
        params = load(fid)

    filters = params['destriping']['filters']
    filters.append(filters[0])
    print(params['destriping']['filters'])

    fname_toml_out.write_text(dumps_params(params), encoding='utf-8')

    with open(fname_toml_out, 'rb') as fid:
        params = load(fid)
    print(params['destriping']['filters'])
