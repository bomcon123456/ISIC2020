
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/utils.ipynb

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return o
    if isinstance(o, Iterable): return list(o)
    return [o]