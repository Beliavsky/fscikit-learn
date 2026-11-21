"""Read-only provenance for the tested sklearn Cython tree extensions."""
from pathlib import Path
import ctypes
import hashlib
import sklearn
import sklearn.tree._partitioner as partitioner
import sklearn.tree._splitter as splitter
import sklearn.tree._tree as tree
import sklearn.tree._criterion as criterion


def tree_runtime_provenance():
    # Cython publishes the address and ABI type through its variable capsule.
    # Reading the c_float value does not modify the reference installation.
    capsule=partitioner.__pyx_capi__['FEATURE_THRESHOLD']
    name=ctypes.pythonapi.PyCapsule_GetName
    name.argtypes=[ctypes.py_object];name.restype=ctypes.c_char_p
    pointer=ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes=[ctypes.py_object,ctypes.c_char_p];pointer.restype=ctypes.c_void_p
    abi=name(capsule);assert abi is not None and abi.endswith(b'float32_t'),abi
    address=pointer(capsule,abi);assert address
    threshold=float(ctypes.c_float.from_address(address).value)
    root=Path(sklearn.__file__).parent
    modules=[partitioner,splitter,tree,criterion]
    return dict(feature_threshold=threshold,capsule_abi=abi.decode('ascii'),
                binary_sha256={str(Path(m.__file__).relative_to(root)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                               for m in modules})
