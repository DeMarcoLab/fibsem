"""

Functions related to attribute get/set python built-ins. 

# Author: Vladan Lucic, Max Planck Institute of Biochemistry
# $Id: attributes.py 334 2008-12-23 11:32:15Z vladan $
"""

__version__ = "$Revision: 334 $"


def getattr_deep(object, name):
    """
    Like built-in getattr, but name can contain dots indicating that it is 
    an attribute of an attribte ... of object.

    Arguments:
      - object: objects
      - name: attribute (of an attribute ...) of object
    """
    
    # split name in attributes (list)
    if isinstance(name, str):
        attributes = name.split('.')
    else:
        attributes = name

    # get attribute
    for attr in attributes:
        object = getattr(object, attr)

    return object

def setattr_deep(object, name, value, mode='_'):
    """
    Like built-in setattr, but if name contains dots it is changed according
    to the mode. If mode is '_' dots are replaced by underscores, and if it
    is 'last', only the part after the rightmost dot is used as name.

    Arguments:
      - object: object
      - name: attribute name
      - value: value
      - mode: determines how a name containing dots is transformed 
    """
    name = get_deep_name(name=name, mode=mode)
    setattr(object, name, value)

def get_deep_name(name, mode='_'):
    """
    Returns name transformed by mode. If mode is '_' dots in name are 
    replaced by underscores, and if it is 'last', only the part after the 
    rightmost dot is used as name.

    Arguments:
      - name: attribute name
      - mode: determines how a name containing dots is transformed 
    """

    if mode == '_':
        name = name.replace('.', '_')

    elif mode == 'last':
        attributes = name.split('.')
        name = attributes.pop()

    else:
        raise ValueError("Argument mode can be '_', or 'last' but not " + mode)

    return name
    
