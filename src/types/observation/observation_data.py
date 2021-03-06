# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys

from collections.abc import MutableMapping, Mapping

from typing import NamedTuple

import numpy as np

from astropy import units as u

from pshmem import MPIShared

from .mpi import MPI, comm_equivalent, comm_equal

from .utils import (
    Logger,
    AlignedI8,
    AlignedU8,
    AlignedI16,
    AlignedU16,
    AlignedI32,
    AlignedU32,
    AlignedI64,
    AlignedU64,
    AlignedF32,
    AlignedF64,
    dtype_to_aligned,
)

from .intervals import IntervalList

from .timing import function_timer

from .accelerator import (
    use_accel_jax,
    use_accel_omp,
    accel_enabled,
    accel_present,
    accel_create,
    accel_delete,
    accel_update_device,
    accel_update_host,
)

if use_accel_jax:
    import jax
    import jax.numpy as jnp


class DetectorData(object):
    """Class representing a logical collection of co-sampled detector data.

    This class works like an array of detector data where the first dimension is the
    number of detectors and the second dimension is the data for that detector.  The
    data for a particular detector may itself be multi-dimensional, with the first
    dimension the number of samples.

    The data in this container may be sliced by both detector indices and names, as
    well as by sample range.

    Example:
        Imagine we have 3 detectors and each has 10 samples.  We want to store a
        4-element value at each sample using 4-byte floats.  We would do::

            detdata = DetectorData(["d01", "d02", "d03"], (10, 4), np.float32)

        and then we can access the data for an individual detector either by index
        or by name with::

            detdata["d01"] = np.ones((10, 4), dtype=np.float32)
            firstdet = detdata[0]

        slicing by index and by a list of detectors is possible::

            array_view = detdata[0:-1, 2:4]
            array_view = detdata[["d01", "d03"], 3:8]

    Args:
        detectors (list):  A list of detector names in exactly the order you wish.
            This order is fixed for the life of the object.
        shape (tuple):  The shape of the data *for each detector*.  The first element
            of this shape should be the number of samples.
        dtype (numpy.dtype):  A numpy-compatible dtype for each element of the detector
            data.  The only supported types are 1, 2, 4, and 8 byte signed and unsigned
            integers, 4 and 8 byte floating point numbers, and 4 and 8 byte complex
            numbers.
        units (Unit):  Optional scalar unit associated with this data.
        view_data (array):  (Internal use only) This makes it possible to create
            DetectorData instances that act as a view on an existing array.

    """

    def __init__(
        self, detectors, shape, dtype, units=u.dimensionless_unscaled, view_data=None
    ):
        log = Logger.get()

        self._set_detectors(detectors)
        self._units = units

        (
            self._storage_class,
            self.itemsize,
            self._dtype,
            self._shape,
            self._flatshape,
        ) = self._data_props(detectors, shape, dtype)

        self._fullsize = 0
        self._memsize = 0
        self._raw = None
        self._raw_jax = None

        if view_data is None:
            # Allocate the data
            self._allocate()
            self._is_view = False
        else:
            # We are provided the data
            if self._shape != view_data.shape:
                msg = (
                    "view data shape ({}) does not match constructor shape ({})".format(
                        view_data.shape, self._shape
                    )
                )
                log.error(msg)
                raise RuntimeError(msg)
            self._data = view_data
            self._is_view = True

    def _set_detectors(self, detectors):
        log = Logger.get()
        self._detectors = detectors
        if len(self._detectors) == 0:
            msg = "You must specify a list of at least one detector name"
            log.error(msg)
            raise ValueError(msg)
        self._name2idx = {y: x for x, y in enumerate(self._detectors)}

    def _data_props(self, detectors, detshape, dtype):
        log = Logger.get()
        dt = np.dtype(dtype)
        storage_class, itemsize = dtype_to_aligned(dtype)

        # Verify that our shape contains only integral values
        flatshape = len(detectors)
        for d in detshape:
            if not isinstance(d, (int, np.integer)):
                msg = "input shape contains non-integer values"
                log.error(msg)
                raise ValueError(msg)
            flatshape *= d

        shp = [len(detectors)]
        shp.extend(detshape)
        shp = tuple(shp)
        return (storage_class, itemsize, dt, shp, flatshape)

    def _allocate(self):
        log = Logger.get()
        self._fullsize = self._flatshape
        self._memsize = self.itemsize * self._fullsize
        recreate = False
        if self._raw is not None:
            if self.accel_present():
                msg = "Reallocation of DetectorData which is staged to accelerator- "
                msg += "Deleting device copy and re-allocating."
                log.verbose(msg)
                self.accel_delete()
                recreate = True
            del self._raw
        self._raw = self._storage_class.zeros(self._fullsize)
        self._flatdata = self._raw.array()[: self._flatshape]
        self._data = self._flatdata.reshape(self._shape)
        if recreate:
            self.accel_create()

    @property
    def detectors(self):
        return list(self._detectors)

    def keys(self):
        return list(self._detectors)

    def indices(self, names):
        """Return the detector indices of the specified detectors.

        Args:
            names (iterable):  The detector names.

        Returns:
            (array):  The detector indices.

        """
        return np.array([self._name2idx[x] for x in names], dtype=np.int32)

    @property
    def dtype(self):
        return self._dtype

    @property
    def units(self):
        return self._units

    @property
    def shape(self):
        return self._shape

    @property
    def detector_shape(self):
        return tuple(self._shape[1:])

    def memory_use(self):
        return self._memsize

    @property
    def data(self):
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        return self._data

    @property
    def flatdata(self):
        return self._flatdata

    def change_detectors(self, detectors):
        """Modify the list of detectors.

        This attempts to re-use the underlying memory and just change the detector
        mapping to that memory.  This is useful if memory allocation is expensive.
        If the new list of detectors is longer than the original, a new memory buffer
        is allocated.  If the new list of detectors is shorter than the original, the
        buffer is kept and only a subset is used.

        The return value indicates whether the underlying memory was re-allocated.

        Args:
            detectors (list):  A list of detector names in exactly the order you wish.

        Returns:
            (bool):  True if the data was re-allocated, else False.

        """
        log = Logger.get()
        if self._is_view:
            msg = "Cannot resize a DetectorData view"
            log.error(msg)
            raise RuntimeError(msg)

        if detectors == self._detectors:
            # No-op
            return

        # Get the new data properties
        (storage_class, itemsize, dt, shp, flatshape) = self._data_props(
            detectors, self._shape[1:], self._dtype
        )

        self._set_detectors(detectors)

        if flatshape > self._fullsize:
            # We have to reallocate...
            self.clear()
            self._shape = shp
            self._flatshape = flatshape
            self._allocate()
            realloced = True
        else:
            # We can re-use the existing memory
            self._shape = shp
            self._flatshape = flatshape
            if use_accel_jax and self.accel_present():
                # FIXME:  Is there really no way to "clear" a jax array?
                self._raw_jax = jnp.zeros_like(self._raw_jax)
                self._flatdata = self._raw_jax[: self._flatshape]
            else:
                self._flatdata = self._raw.array()[: self._flatshape]
                self._flatdata[:] = 0
            self._data = self._flatdata.reshape(self._shape)
            realloced = False

        # Any time we change detectors (even without an alloc), it invalidates
        # the contents of the memory.  If the buffer is staged to a device,
        # its contents will be stale.  We could call self.accel_update_device()
        # here to reset it to zero, but that would cause an extra host to device
        # transfer.  Calling code should instead take care when using
        # change_detectors.

        return realloced

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        if hasattr(self, "_data"):
            del self._data
        if not self._is_view:
            if hasattr(self, "_flatdata"):
                del self._flatdata
            if hasattr(self, "_raw"):
                if self.accel_present():
                    log = Logger.get()
                    msg = "clear() of DetectorData which is staged to accelerator- "
                    msg += "Deleting device copy."
                    log.verbose(msg)
                    if use_accel_omp:
                        accel_delete(self._raw)
                    elif use_accel_jax:
                        del self._raw_jax
                        self._raw_jax = None
                if self._raw is not None:
                    self._raw.clear()
                del self._raw
                self._raw = None

    def __del__(self):
        self.clear()

    def _det_axis_view(self, key):
        if isinstance(key, (int, np.integer)):
            # Just one detector by index
            view = key
        elif isinstance(key, str):
            # Just one detector by name
            view = self._name2idx[key]
        elif isinstance(key, slice):
            # We are slicing detectors by index
            view = key
        else:
            # Assume that our key is at least iterable
            try:
                test = iter(key)
                view = list()
                for k in key:
                    view.append(self._name2idx[k])
                view = tuple(view)
            except TypeError:
                log = Logger.get()
                msg = "Detector indexing supports slice, int, string or "
                msg += f"iterable, not '{key}'"
                log.error(msg)
                raise TypeError(msg)
        return view

    def _get_view(self, key):
        if isinstance(key, (tuple, Mapping)):
            # We are slicing in both detector and sample dimensions
            if len(key) > len(self._shape):
                log = Logger.get()
                msg = "DetectorData has only {} dimensions".format(len(self._shape))
                log.error(msg)
                raise TypeError(msg)
            view = [self._det_axis_view(key[0])]
            for k in key[1:]:
                view.append(k)
            # for s in range(len(self._shape) - len(key)):
            #     view += (slice(None, None, None),)
            return tuple(view)
        else:
            # Only detector slice
            view = self._det_axis_view(key)
            # for s in range(len(self._shape) - 1):
            #     view += (slice(None, None, None),)
            return view

    def __getitem__(self, key):
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        view = self._get_view(key)
        return self._data[view]

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete individual elements")
        return

    def __setitem__(self, key, value):
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        view = self._get_view(key)
        self._data[view] = value

    def view(self, key):
        """Create a new DetectorData instance that acts as a view of the data.

        Args:
            key (tuple/slice):  This is an indexing on detector or both detector and
                sample, the same as you would use to access data elements.

        Returns:
            (DetectorData):  A new instance whose data is a view of the current object.

        """
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        full_view = self._get_view(key)
        view_dets = self.detectors[full_view[0]]
        return DetectorData(
            view_dets,
            self._data[full_view].shape[1:],
            self._dtype,
            view_data=self._data[full_view],
        )

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._detectors)

    def __repr__(self):
        val = None
        if self._is_view:
            val = "<DetectorData (view)"
        else:
            val = "<DetectorData"
        val += " {} detectors each with shape {}, type {}, units {}:".format(
            len(self._detectors), self._shape[1:], self._dtype, self._units
        )
        if self._shape[1] <= 4:
            for d in self._detectors:
                vw = self.data[self._get_view(d)]
                val += "\n  {} = [ ".format(d)
                for i in range(self._shape[1]):
                    val += "{} ".format(vw[i])
                val += "]"
        else:
            for d in self._detectors:
                vw = self.data[self._get_view(d)]
                val += "\n  {} = [ {} {} ... {} {} ]".format(
                    d, vw[0], vw[1], vw[-2], vw[-1]
                )
        val += "\n>"
        return val

    def __eq__(self, other):
        if self.detectors != other.detectors:
            return False
        if self.dtype.char != other.dtype.char:
            return False
        if self.shape != other.shape:
            return False
        if self.units != other.units:
            return False
        if not np.allclose(self.data, other.data):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class DetDataManager(MutableMapping):
    """Class used to manage DetectorData objects in an Observation.

    New objects can be created several ways.  The "create()" method:

        ob.detdata.create(name, sample_shape=None, dtype=None, detectors=None)

    gives full control over creating the named object and specifying the shape of
    each detector sample.  The detectors argument can be used to restrict the object
    to include only a subset of detectors.

    You can also create a new object by assignment from an existing DetectorData
    object or a dictionary of detector arrays.  For example:

        ob.detdata[name] = DetectorData(ob.local_detectors, ob.n_local_samples, dtype)

        ob.detdata[name] = {
            x: np.ones((ob.n_local_samples, 2), dtype=np.int16)
                for x in ob.local_detectors
        }

    Where the right hand side object must have only detectors that are included in
    the ob.local_detectors and the first dimension of shape must be the number of
    local samples.

    It is also possible to create a new object by assigning an array.  In that case
    the array must either have the full size of the DetectorData object
    (n_det x n_sample x sample_shape) or must have dimensions
    (n_sample x sample_shape), in which case the array is copied to all detectors.
    For example:

        ob.detdata[name] = np.ones(
            (len(ob.local_detectors), ob.n_local_samples, 4), dtype=np.float32
        )

        ob.detdata[name] = np.ones(
            (ob.n_local_samples,), dtype=np.float32
        )

    After creation, you can access a given DetectorData object by name with standard
    dictionary syntax:

        ob.detdata[name]

    And delete it as well:

        del ob.detdata[name]

    """

    def __init__(self, dist):
        self.samples = dist.samps[dist.comm.group_rank].n_elem
        self.detectors = dist.dets[dist.comm.group_rank]
        self._internal = dict()

    def _data_shape(self, sample_shape):
        dshape = None
        if sample_shape is None or len(sample_shape) == 0:
            dshape = (self.samples,)
        elif len(sample_shape) == 1 and sample_shape[0] == 1:
            dshape = (self.samples,)
        else:
            dshape = (self.samples,) + sample_shape
        return dshape

    def create(
        self,
        name,
        sample_shape=None,
        dtype=np.float64,
        detectors=None,
        units=u.dimensionless_unscaled,
    ):
        """Create a local DetectorData buffer on this process.

        This method can be used to create arrays of detector data for storing signal,
        flags, or other timestream products on each process.

        If the named detector data already exists in an observation, then additional
        checks are done that the sample_shape and dtype match the existing object.
        If so, then the DetectorData.change_detectors() method is called to re-use
        this existing memory buffer if possible.

        Args:
            name (str): The name of the detector data (signal, flags, etc)
            sample_shape (tuple): Use this shape for the data of each detector sample.
                Use None or an empty tuple if you want one element per sample.
            dtype (np.dtype): Use this dtype for each element.
            detectors (list):  Only construct a data object for this set of detectors.
                This is useful if creating temporary data within a pipeline working
                on a subset of detectors.
            units (Unit):  Optional scalar unit associated with this data.

        Returns:
            None

        """
        log = Logger.get()

        if detectors is None:
            detectors = self.detectors
        else:
            for d in detectors:
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)

        data_shape = self._data_shape(sample_shape)

        if name in self._internal:
            msg = "detdata '{}' already exists".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        # Create the data object
        self._internal[name] = DetectorData(detectors, data_shape, dtype, units=units)

        return

    def ensure(
        self,
        name,
        sample_shape=None,
        dtype=np.float64,
        detectors=None,
        units=u.dimensionless_unscaled,
    ):
        """Ensure that the observation has the named detector data.

        If the named detdata object does not exist, it is created.  If it does exist
        and the sample shape and dtype are compatible, then it is checked whether the
        specified detectors are already included.  If not, it calls the
        DetectorData.change_detectors() method to re-use this existing memory buffer if
        possible.

        The return value is true if the data already exists and includes the specified
        detectors.

        Args:
            name (str): The name of the detector data (signal, flags, etc)
            sample_shape (tuple): Use this shape for the data of each detector sample.
                Use None or an empty tuple if you want one element per sample.
            dtype (np.dtype): Use this dtype for each element.
            detectors (list):  Ensure that these detectors exist in the object.
            units (Unit):  Optional scalar unit associated with this data.

        Returns:
            (bool):  True if the data exists.

        """
        log = Logger.get()

        if detectors is None:
            detectors = self.detectors
        else:
            for d in detectors:
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)

        data_shape = self._data_shape(sample_shape)

        existing = True

        if name in self._internal:
            # The object already exists.  Check properties.
            dt = np.dtype(dtype)
            if dt != self._internal[name].dtype:
                msg = "Detector data '{}' already exists with dtype {}.".format(
                    name, self._internal[name].dtype
                )
                log.error(msg)
                raise RuntimeError(msg)
            if data_shape != self._internal[name].detector_shape:
                msg = "Detector data '{}' already exists with det shape {}.".format(
                    name, self._internal[name].detector_shape
                )
                log.error(msg)
                raise RuntimeError(msg)
            if units != self._internal[name].units:
                msg = "Detector data '{}' already exists with units {}.".format(
                    name, self._internal[name].units
                )
                log.error(msg)
                raise RuntimeError(msg)
            # Ok, we can re-use this.  Are the detectors already included in the data?
            internal_dets = set(self._internal[name].detectors)
            for test_det in detectors:
                if test_det not in internal_dets:
                    # At least one detector is not included
                    existing = False
                    realloced = self._internal[name].change_detectors(detectors)
                    break
        else:
            # Create the data object
            existing = False
            self.create(
                name,
                sample_shape=sample_shape,
                dtype=dtype,
                detectors=detectors,
                units=units,
            )
        return existing

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        if key in self._internal:
            self._internal[key].clear()
            del self._internal[key]

    def __setitem__(self, key, value):
        if isinstance(value, DetectorData):
            # We have an input detector data object.  Verify dimensions
            for d in value.detectors:
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)
            if value.shape[1] != self.samples:
                msg = f"Assignment DetectorData object has {value.shape[1]} samples "
                msg += "instead of {self.samples} in the observation"
                raise ValueError(msg)
            if key not in self._internal:
                # Create it first
                self.create(
                    key,
                    sample_shape=value.detector_shape[1:],
                    dtype=value.dtype,
                    detectors=value.detectors,
                )
            else:
                if value.detector_shape != self._internal[key].detector_shape:
                    msg = "Assignment value has wrong detector shape"
                    raise ValueError(msg)
            for d in value.detectors:
                self._internal[key][d] = value[d]
        elif isinstance(value, Mapping):
            # This is a dictionary of detector arrays
            sample_shape = None
            dtype = None
            for d, ddata in value.items():
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)
                if ddata.shape[0] != self.samples:
                    msg = "Assigment dictionary detector {d} has {ddata.shape[0]} "
                    msg += f"samples instead of {self.samples} in the observation"
                    raise ValueError(msg)
                if sample_shape is None:
                    sample_shape = ddata.shape[1:]
                    dtype = ddata.dtype
                else:
                    if sample_shape != ddata.shape[1:]:
                        msg = "All detector arrays must have the same shape"
                        raise ValueError(msg)
                    if dtype != ddata.dtype:
                        msg = "All detector arrays must have the same type"
                        raise ValueError(msg)
            if key not in self._internal:
                self.create(
                    key,
                    sample_shape=sample_shape,
                    dtype=dtype,
                    detectors=sorted(value.keys()),
                )
            else:
                if (self.samples,) + sample_shape != self._internal[key].detector_shape:
                    msg = "Assignment value has wrong detector shape"
                    raise ValueError(msg)
            for d, ddata in value.items():
                self._internal[key][d] = ddata
        else:
            # This must be just an array- verify the dimensions
            shp = value.shape
            if shp[0] == self.samples:
                # This is a single detector array, being assigned to all detectors
                sample_shape = None
                if len(shp) > 1:
                    sample_shape = shp[1:]
                if key not in self._internal:
                    self.create(
                        key,
                        sample_shape=sample_shape,
                        dtype=value.dtype,
                        detectors=self.detectors,
                    )
                else:
                    fullshape = (self.samples,)
                    if sample_shape is not None:
                        fullshape += sample_shape
                    if fullshape != self._internal[key].detector_shape:
                        msg = "Assignment value has wrong detector shape"
                        raise ValueError(msg)
                for d in self.detectors:
                    self._internal[key][d] = value
            elif shp[0] == len(self.detectors):
                # Full sized array
                if shp[1] != self.samples:
                    msg = "Assignment value has wrong number of samples"
                    raise ValueError(msg)
                sample_shape = None
                if len(shp) > 2:
                    sample_shape = shp[2:]
                if key not in self._internal:
                    self.create(
                        key,
                        sample_shape=sample_shape,
                        dtype=value.dtype,
                        detectors=self.detectors,
                    )
                else:
                    fullshape = (self.samples,)
                    if sample_shape is not None:
                        fullshape += sample_shape
                    if fullshape != self._internal[key].detector_shape:
                        msg = "Assignment value has wrong detector shape"
                        raise ValueError(msg)
                self._internal[key][:] = value
            else:
                # Incompatible
                msg = "Assignment of detector data from an array only supports full "
                msg += "size or single detector"
                raise ValueError(msg)

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        for k in self._internal.keys():
            self._internal[k].clear()

    def __repr__(self):
        val = "<DetDataManager {} local detectors, {} samples".format(
            len(self.detectors), self.samples
        )
        for k in self._internal.keys():
            val += "\n    {}: shape={}, dtype={}, units='{}'".format(
                k,
                self._internal[k].shape,
                self._internal[k].dtype,
                self._internal[k].units,
            )
        val += ">"
        return val


class SharedDataType(NamedTuple):
    """The shared data object and a string specifying the comm type."""

    shdata: MPIShared
    type: str

class IntervalsManager(MutableMapping):
    """Class for creating and storing interval lists in an observation.

    Named lists of intervals are accessed by dictionary style syntax ([] brackets).
    When making new interval lists, these can be added directly on each process, or
    some helper functions can be used to create the appropriate local interval lists
    given a global set of ranges.

    Args:
        dist (DistDetSamp):  The observation data distribution.

    """

    # This could be anything, just has to be unique
    all_name = "ALL_OBSERVATION_SAMPLES"

    def __init__(self, dist, local_samples):
        self.comm = dist.comm
        self.comm_col = dist.comm_col
        self.comm_row = dist.comm_row
        self._internal = dict()
        self._del_callbacks = dict()
        self._local_samples = local_samples

    def create_col(self, name, global_timespans, local_times, fromrank=0):
        """Create local interval lists on the same process column.

        Processes within the same column of the observation data distribution have the
        same local time range.  This function takes the global time ranges provided,
        computes the intersection with the local time range of this process column,
        and creates a local named interval list on each process in the column.

        Args:
            name (str):  The key to use in the local intervals dictionary.
            global_times (list):  List of start, stop tuples containing time ranges
                within the observation.
            local_times (array):  The local timestamps on this process.
            fromrank (int):  Get the list from this process rank of the observation
                column communicator.  Input arguments on other processes are ignored.

        """
        if self.comm_col is not None:
            # Broadcast to all processes in this column
            n_global = 0
            if global_timespans is not None:
                n_global = len(global_timespans)
            n_global = self.comm_col.bcast(n_global, root=fromrank)
            if n_global == 0:
                global_timespans = list()
            else:
                global_timespans = self.comm_col.bcast(global_timespans, root=fromrank)
        # Every process creates local intervals
        lt = local_times
        if isinstance(lt, MPIShared):
            lt = local_times.data
        self._internal[name] = IntervalList(lt, timespans=global_timespans)

    def create(self, name, global_timespans, local_times, fromrank=0):
        """Create local interval lists from global time ranges on one process.

        In some situations, a single process has loaded data from the disk, queried a
        database, etc and has information about some time spans that are global across
        the observation.  This function automatically creates the named local interval
        list consisting of the intersection of the local sample range with these global
        intervals.

        Args:
            name (str):  The key to use in the local intervals dictionary.
            global_timespans (list):  List of start, stop tuples containing time ranges
                within the observation.
            local_times (array):  The local timestamps on this process.
            fromrank (int):  Get the list from this process rank of the observation
                communicator.  Input arguments on other processes are ignored.

        """
        send_col_rank = 0
        send_row_rank = 0
        if self.comm.comm_group is not None:
            col_rank = 0
            if self.comm_col is not None:
                col_rank = self.comm_col.rank
            # Find the process grid ranks of the incoming data
            if self.comm.group_rank == fromrank:
                if self.comm_col is not None:
                    send_col_rank = self.comm_col.rank
                if self.comm_row is not None:
                    send_row_rank = self.comm_row.rank
            send_col_rank = self.comm.comm_group.bcast(send_col_rank, root=0)
            send_row_rank = self.comm.comm_group.bcast(send_row_rank, root=0)
            # Broadcast data along the row
            if col_rank == send_col_rank:
                if self.comm_row is not None:
                    n_global = 0
                    if global_timespans is not None:
                        n_global = len(global_timespans)
                    n_global = self.comm_row.bcast(n_global, root=send_row_rank)
                    if n_global == 0:
                        global_timespans = list()
                    else:
                        global_timespans = self.comm_row.bcast(
                            global_timespans, root=send_row_rank
                        )
        # Every process column creates their local intervals
        self.create_col(name, global_timespans, local_times, fromrank=send_col_rank)

    def register_delete_callback(self, key, fn):
        self._del_callbacks[key] = fn

    # Mapping methods

    def _real_key(self, key):
        if key is None:
            if self.all_name not in self._internal:
                # Create fake intervals
                faketimes = -1.0 * np.ones(self._local_samples, dtype=np.float64)
                self._internal[self.all_name] = IntervalList(
                    faketimes, samplespans=[(0, self._local_samples - 1)]
                )
            return self.all_name
        else:
            return key

    def __getitem__(self, key):
        key = self._real_key(key)
        return self._internal[key]

    def __delitem__(self, key):
        key = self._real_key(key)
        if key in self._del_callbacks:
            try:
                self._del_callbacks[key](key)
                del self._del_callbacks[key]
            except:
                pass
        if key in self._internal:
            del self._internal[key]

    def __setitem__(self, key, value):
        if not isinstance(value, IntervalList):
            raise ValueError("Value must be an IntervalList instance.")
        self._internal[key] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        self._internal.clear()

    def __del__(self):
        if hasattr(self, "_internal"):
            self.clear()

    def __repr__(self):
        val = "<IntervalsManager {} lists".format(len(self._internal))
        for k in self._internal.keys():
            val += "\n  {}: {} intervals".format(k, len(self._internal[k]))
        val += ">"
        return val
