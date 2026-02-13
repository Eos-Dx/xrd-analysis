"""DIFRA HDF5 I/O Helpers

Provides safe "transactional" write operations and HDF5 object reference helpers.
All operations open the file, perform writes, flush, and close immediately to 
minimize thread-safety risks and improve crash recovery.
"""

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from . import schema


@contextlib.contextmanager
def open_h5_append(file_path: Union[str, Path], create_if_missing: bool = False):
    """Context manager for opening HDF5 file in append mode.
    
    Usage:
        with open_h5_append("container.h5") as f:
            f.attrs["key"] = "value"
            # Changes are flushed and file is closed on exit
    
    Args:
        file_path: Path to HDF5 file
        create_if_missing: If True, create file if it doesn't exist (mode 'a'),
                          else raise error if missing (mode 'r+')
    
    Yields:
        h5py.File in read-write mode
    """
    file_path = Path(file_path)
    mode = "a" if create_if_missing else "r+"
    
    if mode == "r+" and not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    f = h5py.File(file_path, mode)
    try:
        yield f
        f.flush()
    finally:
        f.close()


def create_empty_container(
    file_path: Union[str, Path],
    container_id: str,
    container_type: str,
    root_attrs: Optional[Dict[str, Any]] = None
) -> None:
    """Create a new empty HDF5 container with required root attributes.
    
    Args:
        file_path: Where to create the container
        container_id: Unique 16-char hex container ID
        container_type: "technical" or "session"
        root_attrs: Additional root-level attributes
    """
    file_path = Path(file_path)
    if file_path.exists():
        raise FileExistsError(f"Container already exists: {file_path}")
    
    if not schema.validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")
    
    if container_type not in [schema.CONTAINER_TYPE_TECHNICAL, schema.CONTAINER_TYPE_SESSION]:
        raise ValueError(f"Invalid container type: {container_type}")
    
    with h5py.File(file_path, "w") as f:
        # Write required root attributes
        f.attrs[schema.ATTR_CONTAINER_ID] = container_id
        f.attrs[schema.ATTR_CONTAINER_TYPE] = container_type
        f.attrs[schema.ATTR_SCHEMA_VERSION] = schema.SCHEMA_VERSION
        
        # Write additional root attrs if provided
        if root_attrs:
            for key, value in root_attrs.items():
                f.attrs[key] = value


def set_attrs(
    file_path: Union[str, Path],
    path: str,
    attrs: Dict[str, Any]
) -> None:
    """Set attributes on a group or dataset.
    
    Args:
        file_path: Container file path
        path: HDF5 path to group or dataset (e.g. "/technical" or "/measurements/pt_001/meas_000000001")
        attrs: Dictionary of attributes to set
    """
    with open_h5_append(file_path) as f:
        obj = f[path]
        for key, value in attrs.items():
            obj.attrs[key] = value


def create_group_if_missing(
    file_path: Union[str, Path],
    group_path: str,
    attrs: Optional[Dict[str, Any]] = None
) -> None:
    """Create a group if it doesn't exist, optionally with attributes.
    
    Args:
        file_path: Container file path
        group_path: Full path to group (e.g. "/technical/config")
        attrs: Optional attributes to set on the group
    """
    with open_h5_append(file_path, create_if_missing=True) as f:
        if group_path not in f:
            grp = f.create_group(group_path)
        else:
            grp = f[group_path]
        
        if attrs:
            for key, value in attrs.items():
                grp.attrs[key] = value


def write_dataset(
    file_path: Union[str, Path],
    dataset_path: str,
    data: Union[np.ndarray, str, bytes],
    attrs: Optional[Dict[str, Any]] = None,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    overwrite: bool = False
) -> None:
    """Write a dataset with optional compression and attributes.
    
    Args:
        file_path: Container file path
        dataset_path: Full path including dataset name (e.g. "/technical/tech_evt_001/det_primary/raw_signal")
        data: Array, string, or bytes to store
        attrs: Optional attributes for the dataset
        compression: Compression algorithm ("gzip", "lzf", or None)
        compression_opts: Compression level (1-9 for gzip)
        overwrite: If True, delete existing dataset first
    """
    with open_h5_append(file_path, create_if_missing=True) as f:
        if dataset_path in f:
            if overwrite:
                del f[dataset_path]
            else:
                raise ValueError(f"Dataset already exists: {dataset_path}")
        
        # Handle string data specially
        if isinstance(data, str):
            dt = h5py.string_dtype(encoding='utf-8')
            dset = f.create_dataset(dataset_path, data=data, dtype=dt)
        elif isinstance(data, bytes):
            dset = f.create_dataset(dataset_path, data=np.frombuffer(data, dtype=np.uint8))
        else:
            # Numeric array
            dset = f.create_dataset(
                dataset_path,
                data=data,
                compression=compression,
                compression_opts=compression_opts
            )
        
        if attrs:
            for key, value in attrs.items():
                dset.attrs[key] = value


def copy_group(
    src_file: Union[str, Path],
    src_group: str,
    dst_file: Union[str, Path],
    dst_group: str = None
) -> None:
    """Copy an entire group from one container to another.
    
    Args:
        src_file: Source container path
        src_group: Source group path (e.g. "/technical")
        dst_file: Destination container path
        dst_group: Destination group path (defaults to same as src_group)
    """
    if dst_group is None:
        dst_group = src_group
    
    src_file = Path(src_file)
    dst_file = Path(dst_file)
    
    if not src_file.exists():
        raise FileNotFoundError(f"Source container not found: {src_file}")
    
    with h5py.File(src_file, "r") as src:
        if src_group not in src:
            raise KeyError(f"Source group not found: {src_group} in {src_file}")
        
        with h5py.File(dst_file, "a") as dst:
            if dst_group in dst:
                del dst[dst_group]
            
            # Use h5py's efficient group copy
            src.copy(src_group, dst, name=dst_group)


# =============== HDF5 Object Reference Helpers ===============

def create_reference(
    file_path: Union[str, Path],
    target_path: str
) -> h5py.Reference:
    """Create an HDF5 object reference to a group or dataset.
    
    Note: References are only valid within the file they were created in.
    
    Args:
        file_path: Container file path
        target_path: Path to target object (e.g. "/technical/poni/poni_primary")
    
    Returns:
        h5py.Reference object
    """
    with open_h5_append(file_path) as f:
        if target_path not in f:
            raise KeyError(f"Target not found: {target_path}")
        return f[target_path].ref


def dereference(
    file_handle: h5py.File,
    ref: h5py.Reference
) -> h5py.Group:
    """Dereference an HDF5 object reference.
    
    Args:
        file_handle: Open h5py.File
        ref: Reference to dereference
    
    Returns:
        Referenced group or dataset
    """
    return file_handle[ref]


def set_reference_attr(
    file_path: Union[str, Path],
    obj_path: str,
    attr_name: str,
    target_path: str
) -> None:
    """Set a single reference as an attribute.
    
    Args:
        file_path: Container file path
        obj_path: Path to object that will hold the reference attribute
        attr_name: Attribute name (e.g. "poni_ref")
        target_path: Path to referenced object
    """
    with open_h5_append(file_path) as f:
        if obj_path not in f:
            raise KeyError(f"Object not found: {obj_path}")
        if target_path not in f:
            raise KeyError(f"Target not found: {target_path}")
        
        obj = f[obj_path]
        target = f[target_path]
        obj.attrs[attr_name] = target.ref


def set_reference_list_attr(
    file_path: Union[str, Path],
    obj_path: str,
    attr_name: str,
    target_paths: List[str]
) -> None:
    """Set a list of references as an attribute.
    
    Args:
        file_path: Container file path
        obj_path: Path to object that will hold the reference list attribute
        attr_name: Attribute name (e.g. "analytical_measurement_refs")
        target_paths: List of paths to referenced objects
    """
    with open_h5_append(file_path) as f:
        if obj_path not in f:
            raise KeyError(f"Object not found: {obj_path}")
        
        obj = f[obj_path]
        
        if not target_paths:
            # Empty list: store as empty array with ref dtype
            obj.attrs[attr_name] = np.array([], dtype=h5py.ref_dtype)
        else:
            # Create array of references
            refs = []
            for target_path in target_paths:
                if target_path not in f:
                    raise KeyError(f"Target not found: {target_path}")
                refs.append(f[target_path].ref)
            
            obj.attrs[attr_name] = np.array(refs, dtype=h5py.ref_dtype)


def append_reference_to_list_attr(
    file_path: Union[str, Path],
    obj_path: str,
    attr_name: str,
    target_path: str
) -> None:
    """Append a reference to an existing reference list attribute.
    
    Args:
        file_path: Container file path
        obj_path: Path to object with reference list attribute
        attr_name: Attribute name
        target_path: Path to new target to append
    """
    with open_h5_append(file_path) as f:
        if obj_path not in f:
            raise KeyError(f"Object not found: {obj_path}")
        if target_path not in f:
            raise KeyError(f"Target not found: {target_path}")
        
        obj = f[obj_path]
        new_ref = f[target_path].ref
        
        # Get existing refs
        if attr_name in obj.attrs:
            existing_refs = obj.attrs[attr_name]
            if existing_refs.size == 0:
                # Was empty, create new array
                obj.attrs[attr_name] = np.array([new_ref], dtype=h5py.ref_dtype)
            else:
                # Append to existing - convert to list, append, rebuild array with ref dtype
                ref_list = list(existing_refs) + [new_ref]
                obj.attrs[attr_name] = np.array(ref_list, dtype=h5py.ref_dtype)
        else:
            # Create new list with single ref
            obj.attrs[attr_name] = np.array([new_ref], dtype=h5py.ref_dtype)


def get_reference_targets(
    file_handle: h5py.File,
    obj_path: str,
    attr_name: str
) -> List[str]:
    """Get paths of all objects referenced by a reference list attribute.
    
    Args:
        file_handle: Open h5py.File
        obj_path: Path to object with reference attribute
        attr_name: Attribute name
    
    Returns:
        List of paths to referenced objects
    """
    obj = file_handle[obj_path]
    if attr_name not in obj.attrs:
        return []
    
    refs = obj.attrs[attr_name]
    if refs.size == 0:
        return []
    
    paths = []
    for ref in refs:
        target = file_handle[ref]
        paths.append(target.name)
    
    return paths


def get_container_info(file_path: Union[str, Path]) -> Dict:
    """Get container metadata and basic information.
    
    Args:
        file_path: Path to container
        
    Returns:
        Dict with metadata
    """
    with h5py.File(file_path, 'r') as f:
        info = {
            'path': str(file_path),
            'container_type': f.attrs.get(schema.ATTR_CONTAINER_TYPE, 'unknown'),
            'schema_version': f.attrs.get(schema.ATTR_SCHEMA_VERSION, 'unknown'),
        }
        
        # Add type-specific info
        if info['container_type'] == 'session':
            info['sample_id'] = f.attrs.get(schema.ATTR_SAMPLE_ID, 'unknown')
            info['study_name'] = f.attrs.get(schema.ATTR_STUDY_NAME, 'unknown')
            info['operator_id'] = f.attrs.get(schema.ATTR_OPERATOR_ID, 'unknown')
        
        return info


def verify_container_readable(file_path: Union[str, Path]) -> bool:
    """Check if file is a readable HDF5 container.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if readable
    """
    try:
        with h5py.File(file_path, 'r') as f:
            return schema.ATTR_CONTAINER_TYPE in f.attrs
    except:
        return False


def open_h5_readonly(file_path: Union[str, Path]):
    """Open HDF5 file in read-only mode.
    
    Args:
        file_path: Path to file
        
    Returns:
        Context manager for h5py File
    """
    return h5py.File(file_path, 'r')
