#!/usr/bin/env python
"""
Contains functions often used in the scripts of this directory

$Id: common.py 1179 2015-05-28 11:50:04Z vladan $
Author: Vladan Lucic 
"""
__version__ = "$Revision: 1179 $"

# ToDo: see if this should become a superclass for some of the scripts

import imp
import sys
import os
import os.path
import time
import platform
import pickle
from copy import copy, deepcopy
import logging

import numpy

import tdct.pyto
import tdct.pyto.attributes as attributes


#################################################
#
# Module import
#

def __import__(name, path):
    """
    Imports a module that is outside of python path.

    Modified from Python Library Reference.

    Arguments:
      - name: module name (file name without '.py')
      - path: directory where the module resides, relative to the directory
        of the calling function

    Returns: module, on None if module not found
    """

    # do not check if the module has already been imported, because a different
    # module with the same name might have been already imported
    #try:
    #    return sys.modules[name]
    #except KeyError:
    #    pass

    # import
    fp, pathname, description = imp.find_module(name, [path])
    try:
        return imp.load_module(name, fp, pathname, description)
    finally:
        # since we may exit via an exception, close fp explicitly.
        if fp:
            fp.close()


##################################################
#
# File name related
#

def get_file_base(file_):
    """
    Returns base and root of the image file name
    """
    (dir, base) = os.path.split(file_)
    (root, ext) = os.path.splitext(base)
    return base, root

def format_param(value=None, name='', format=None):
    """
    Makes parameter strings to be used for file names.

    Arguments:
      - value: parameter value
      - name: parameter name
      - format: parameter format
    """

    if value is not None:
        value_str = (format % value).strip()
        value_long_str = name + value_str
    else:
        value_str = ''
        value_long_str = ''

    return value_str, value_long_str

def make_file_name(directory='', prefix='', insert_root=True, 
                   reference=None, param_name='', param_value=None, 
                   param_format=None, suffix=''):
    """
    Returns a labels file name of the form:

      <directory> / <prefix> + reference_base + formated_param + <suffix>

    where: 
      - reference_base: arg reference without the directory and extension parts
      obtained by self.format_param(), added if arg insert_root is True
      - formated_param: formated parameter value obtained using 
      self.format_param()

    Arguments:
      - directory: directory
      - prefix: file name prefix
      - insert_root: insert reference file root in the file name
      - reference: reference file name
      - param_name: name of the parameter
      - param_value: value of a parameter to be inserted in the name
      - param_format: parameter value format
      - suffix: file name suffix
    """

    # format parameter
    value_str, value_long_str = format_param(value=param_value, name=param_name,
                                             format=param_format)

    # extract root from the reference file
    ref_base, root = get_file_base(reference)

    if insert_root:
        base = prefix + root + value_long_str + suffix
    else:
        base = prefix + value_long_str + suffix
    file_name = os.path.join(directory, base)

    return file_name
    
##################################################
#
# Reading image files
#

def read_image(file_name):
    """
    Reads image file and returns an segmentation.Grey object

    Argument:
      - file_name: image file name
    """
    image = pyto.segmentation.Grey.read(file=file_name)
    return image

def read_labels(
        file_name, ids, label_ids=None, shape=None, suggest_shape=None, 
        byte_order=None, data_type=None, array_order=None, shift=None,
        clean=False, offset=None, check=True):
    """
    Reads file(s) containing labels.

    Works on single and multiple boundaries files. In the latter case, ids
    are shifted so that they do not overlap and the boundaries from different 
    files are merged.

    The label file shape is determined using the first found of the following:
      - argument shape
      - shape given in the labels file header (em im or mrc format) 
      - argument suggest_shape

    If the file is in em or mrc format data type, byte order
    and array order are not needed (should be set to None). 

    Arguments:
      - file_name: labels file name
      - ids: ids of all labels that are kept, non-specified labels may be 
      removed and set to background depending on arg clean
      - clean: flag indicating if non-specified labels are removed 
      - label_ids: ids of labels that are actually used as labels (segments), 
      used only to return shifted_label_ids. If None arg ids is used
      - byte_order: labels byte order
      - data_type: labels data_type
      - array_order: labels array order
      - shape, suggest_shape: image shape, see above
      - shift: id shift between subsequent label files, if None determined
    automatically
      - offset: label offset (currently not used)
      - check: if True checks if there are ids without a boundary (data 
      elements), or disconnected boundaries

    Returns (labels, label_ids) where:
      - labels: (Segment) labels, from single file or merged
      - label_ids: (list of ndarrays) label ids, each list element contains
      ids from one boundary file (shifted in case of multiple files)
    """

    # set label_ids if not given
    if label_ids is None:
        label_ids = ids

    # read
    if is_multi_file(file_name=file_name):
        bound, multi_boundary_ids = read_multi_labels(
            file_name=file_name, ids=ids, label_ids=label_ids, shift=shift, 
            shape=shape, suggest_shape=suggest_shape, 
            byte_order=byte_order, data_type=data_type, array_order=array_order,
            clean=clean)
    else:
        bound = read_single_labels(
            file_name=file_name, ids=ids, shape=shape, 
            suggest_shape=suggest_shape, byte_order=byte_order, 
            data_type=data_type, array_order=array_order, clean=clean)
        multi_boundary_ids = [label_ids]

    # offset
    bound.offset = offset

    # check
    if check:
        nonun = bound.findNonUnique()
        if len(nonun['many']) > 0:
            logging.warning(
                "The following discs are disconnected: " + str(nonun.many))
        if len(nonun['empty']) > 0:
            logging.warning(
                "The following discs do not exist: " + str(nonun.empty))

    return bound, multi_boundary_ids

def is_multi_file(file_name):
    """
    Returns True if multiple files are given.

    Argument:
      - file_name: one file name or a list (tuple) of file names
    """
    if isinstance(file_name, str):
        return False
    elif isinstance(file_name, (tuple, list)):
        return True
    else:
        raise ValueError(
            "File name " + str(file_name) + " has to be either a string / "
            + "unicode (one file) or a tuple (multiple files).")    

def read_single_labels(
        file_name, ids, shape=None, byte_order=None, data_type=None, 
        array_order=None, suggest_shape=None, clean=False):
    """
    Reads and initializes labels from a sigle labels file.

    The label file shape is determined using the first found of the following:
      - argument shape
      - shape given in the labels file header (em im or mrc format) 
      - argument suggest_shape

    If the file is in em or mrc format data type, byte order
    and array order are not needed (should be set to None). 

    Arguments:
      - file_name: labels file name
      - ids: ids of all labels that are kept, non-specified labels may be 
      removed and set to background depending on arg clean
      - clean: flag indicating if non-specified labels are removed 
      - byte_order: labels byte order
      - data_type: labels data_type
      - array_order: labels array order
      - shape, suggest_shape: image shape, see above

    Returns (Segment) labels.
    """

    # find shape
    shape = find_shape(file_name=file_name, shape=shape,
                       suggest_shape=suggest_shape)

    # read labels file and make a Segment object
    bound = pyto.segmentation.Segment.read(
        file=file_name, ids=ids, clean=clean, 
        byteOrder=byte_order, dataType=data_type,
        arrayOrder=array_order, shape=shape)

    return bound

def read_multi_labels(
        file_name, ids, label_ids, shift=None, shape=None, suggest_shape=None, 
        byte_order=None, data_type=None, array_order=None, clean=False):
    """
    Reads and initializes labels form multiple labels file. The label ids
    are shifted so that they do not overlap and the labels are merged.

    The label file shape is determined using the first found of the following:
      - argument shape
      - shape given in the labels file header (em im or mrc format) 
      - argument suggest_shape

    If the file is in em or mrc format data type, byte order
    and array order are not needed (should be set to None). 

    Arguments:
      - file_name: labels file name
      - ids: ids of all labels that are kept, non-specified labels may be 
      removed and set to background depending on arg clean
      - clean: flag indicating if non-specified labels are removed 
      - label_ids: ids of labels that are actually used as labels (segments), 
      used only to return shifted_label_ids
      - byte_order: labels byte order
      - data_type: labels data_type
      - array_order: labels array order
      - shape, suggest_shape: image shape, see above
      - shift: id shift between subsequent label files, if None determined
    automatically

    Returns (labels, shifted_label_ids) where:
      - labels: (Segment) merged labels
      - shifted_label_ids: (list of ndarrays) shifted ids
    """

    # read all labels files and combine them in a single Segment object
    bound = pyto.segmentation.Segment()
    curr_shift = 0
    shifted_vesicle_ids = []
    for (l_name, a_ids, v_ids) in zip(file_name, ids, label_ids):

        # find shape
        found_shape = find_shape(file_name=l_name, shape=shape,
                                 suggest_shape=suggest_shape)

        # read
        curr_bound = pyto.segmentation.Segment.read(
            file=l_name, ids=a_ids,
            clean=clean, byteOrder=byte_order, dataType=data_type,
            arrayOrder=array_order, shape=found_shape)

        bound.add(new=curr_bound, shift=curr_shift, dtype='int16')
        shifted_vesicle_ids.append(numpy.array(v_ids) + curr_shift)
        if shift is None:
            curr_shift = None
        else:
            curr_shift += shift

    return bound, shifted_vesicle_ids

def find_shape(file_name, shape=None, suggest_shape=None):
    """
    Determines image file shape using the first found of the following:
      - argument shape; returns shape
      - shape given in the image file header (if in em or mrc format); returns
      None (shape to be read from file header)
      - argument suggest_shape; returns suggest_shape

    Arguments:
      - file_name: image file name
      - shape: (list or tuple) file shape (see above)
      - suggest_shape: (list or tuple) file shape (see above)

    Returns: image file shape found
    """

    if shape is not None:

        # specified by arg shape
        result_shape = shape

    else:

        bound_io = pyto.io.ImageIO(file=file_name)
        bound_io.setFileFormat()
        if (bound_io.fileFormat == 'em') or (bound_io.fileFormat == 'mrc'):

            # specified in header
            result_shape = None
 
        elif suggest_shape is not None:

            # specified by arg suggest_shape
            result_shape = suggest_shape

        else: 

            # not found
            raise ValueError("Shape of file " + file_name + 
                             "was not specified.") 

    return result_shape
        

##################################################
#
# Writting result files
#

def make_top_header():
    """
    Returns header lines containing machine and files info
    """

    # machine info
    mach_name, mach_arch = machine_info()

    # out file names
    script_file_name = sys.modules[__name__].__file__

    # general 
    header = ["#",
        "# Machine: " + mach_name + " " + mach_arch,
        "# Date: " + time.asctime(time.localtime()),
        "#"]
    header.extend(format_file_info(
            name=script_file_name, description="Input script", 
            extra=("  "+__version__)))
    header.append("# Working directory: " + os.getcwd())
    header.append("#")

    return header

def machine_info():
    """
    Returns machine name and machine architecture strings
    """
    mach = platform.uname() 
    mach_name = mach[1]
    mach_arch = str([mach[0], mach[4], mach[5]])

    return mach_name, mach_arch

def format_file_info(name, description, ids=None, extra=''):
    """
    Returns a list of string(s) containing file, description and file creation
    time. Works also if more than one name is given. If arg ids is specified
    ids are added too.
    
    Arguments:
      - name: file name
      - description: file description
      - ids: ids
      - extra: other info
    """

    if name is None: return []

    if is_multi_file(file_name=name):

        # multi file
        lines = ["# " + description + ":"]
        for one_name in name:
            try:
                file_time = time.asctime(time.localtime(
                        os.path.getmtime(one_name)))
            except OSError:
                file_time = 'not written'
            lines.extend(["#     " + one_name + " (" + file_time + ")"])
            if ids is not None:
                lines.extend(["#     Ids:" + str(ids)])

    else:

        # single_file
        try:
            file_time = time.asctime(time.localtime(os.path.getmtime(name)))
        except OSError:
            file_time = 'not written'
        lines = [("# " + description + ": " + name + " (" + file_time + ")" 
                  + extra)] 

    return lines

##################################################
#
# Writting data files
#

def write_labels(labels, name, data_type, inset=False, ids=None, 
                 length=None, pixel=1, casting='unsafe'):
    """
    Writes labels as an array.

    If arg ids is specified, modifies object labels by removing all segments
    that are not specified in arg ids.

    Arguments:
      - labels: (Labels) labels object (e.g. segmentation)
      - name: labels file name
      - data_type: data type
      - inset: the data is repositioned to this inset, if None labels.data 
      array is written without repositioning
      - ids: if not None, only the given ids are retained 
      - length: (list aor ndarray) length in each dimension in nm (used 
      only for mrc format)
      - pixel: pixel size in nm, used only for mrc files and if length is
      not None
      - casting: Controls what kind of data casting may occur: 'no', 
      'equiv', 'safe', 'same_kind', 'unsafe'. Identical to numpy.astype()
      method.
    """

    # expand if needed
    init_inset = labels.inset
    if inset is not None:
        labels.useInset(inset=inset, mode='abs', expand=True)

    # remove ids if needed
    if ids is not None:
        labels_clean = deepcopy(labels)
        labels_clean.keep(ids=ids)
    else:
        labels_clean = labels

    # see about adjusting the data type if needed

    # write
    labels_clean.write(file=name, dataType=data_type, length=length, 
                       pixel=pixel, casting=casting)

    # revert to the original inset
    if inset is not None:
        labels.useInset(inset=init_inset, mode='abs')

##################################################
#
# Pickle files
#

def read_pickle(file_name, compact=[], inset=None, image=[]):
    """
    Reads pickles

    If arg inset is specified, images specified by arg image are brought to
    the given inset. Objects specified in arg compact are expanded.

    Arguments:
      - file_name: pickle file name
      - image: list of names of attribute holding images (can be attributes of
      ... of objects)
      - compact: list of names of attribute that should be expanded (can
      be attributes of ... of objects)

    Returns: unpickled object
    """

    # unpickle
    obj = pickle.load(open(file_name))

    # expand
    for c_name in compact:
        compact_obj = attributes.getattr_deep(obj, c_name)
        compact_obj.expand()

    # bring images to inset
    if inset is not None:
        for image_name in image:
            image_obj = attributes.getattr_deep(obj, image_name)        
            image_obj.useInset(inset=inset, mode='abs', 
                               useFull=True, expand=True)

    return obj

def write_pickle(obj, file_name, image=[], compact=[]):
    """
    Pickles (and writes) large objects.

    Images specified by arg image are reduced to smallest insets and 
    objects specified in arg compact are compactified.

    Arguments:
      - obj: object to be pickled
      - file_name: pickle file name
      - image: list of names of attribute holding images (can be attributes of
      ... of objects)
      - compact: list of names of attribute that should be compactified (can
      be attributes of ... of objects)
    """

    # compactify contacts
    for c_name in compact:
        contacts = attributes.getattr_deep(obj, c_name)
        contacts.compactify()

    # reduce data size
    full_insets = []
    for image_name in image:
         image_obj = attributes.getattr_deep(obj, image_name)
         full_insets.append(image_obj.inset)
         image_obj.makeInset()

    # write 
    out_file = open(file_name, 'wb')
    pickle.dump(obj, out_file, -1)

    # expand 
    for c_name in compact:
        compact_obj = attributes.getattr_deep(obj, c_name)
        compact_obj.expand()

    # recover image insets
    for image_name, inset in zip(image, full_insets):
        image_obj = attributes.getattr_deep(obj, image_name)
        image_obj.useInset(inset=inset, mode='abs', useFull=True, expand=True)
    
