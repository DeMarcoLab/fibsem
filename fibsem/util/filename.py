import os

def _get_extension(filename: str) -> str:
    if filename.endswith(".ome.tiff"): # special case for OME-TIFF files (double extension)
        return ".ome.tiff"
    else:
        return os.path.splitext(filename)[1]
    
def _get_basename(filename: str) -> str:
    return filename.removesuffix(_get_extension(filename))

def _get_basename_and_extension(filename: str) -> tuple:
    return _get_basename(filename), _get_extension(filename)

def get_unique_filename(filename):
    if not os.path.exists(filename):
        return filename

    basename, ext = _get_basename_and_extension(filename)
    idx = 1
    while os.path.exists(filename):
        filename = f"{basename}-{idx}{ext}"
        idx += 1

    return filename