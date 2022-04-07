# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

class PixelDistribution(object):
    """Class representing the distribution of submaps.

    This object is used to describe the properties of a pixelization scheme and which
    "submaps" are strored on each process.  The size of the submap can be tuned to
    balance storage (smaller submap size means fewer wasted pixels stored) and ease of
    indexing (larger submap size means faster global-to-local pixel lookups).

    Args:
        n_pix (int): the total number of pixels.
        n_submap (int): the number of submaps to use.
        local_submaps (array): the list of local submaps (integers).
        comm (mpi4py.MPI.Comm): The MPI communicator or None.

    """

    def __init__(self, n_pix=None, n_submap=1000, local_submaps=None, comm=None):
        self._n_pix = n_pix
        self._n_submap = n_submap
        if self._n_submap > self._n_pix:
            msg = "Cannot create a PixelDistribution with more submaps ({}) than pixels ({})".format(
                n_submap, n_pix
            )
            raise RuntimeError(msg)
        self._n_pix_submap = self._n_pix // self._n_submap
        if self._n_pix % self._n_submap != 0:
            self._n_pix_submap += 1

        self._local_submaps = local_submaps
        self._comm = comm

        self._glob2loc = None
        self._n_local = 0

        if self._local_submaps is not None and len(self._local_submaps) > 0:
            if np.max(self._local_submaps) > self._n_submap - 1:
                raise RuntimeError("local submap indices out of range")
            self._n_local = len(self._local_submaps)
            self._glob2loc = AlignedI64.zeros(self._n_submap)
            self._glob2loc[:] = -1
            for ilocal_submap, iglobal_submap in enumerate(self._local_submaps):
                self._glob2loc[iglobal_submap] = ilocal_submap

        self._submap_owners = None
        self._owned_submaps = None
        self._alltoallv_info = None
        self._all_hit_submaps = None

    @property
    def comm(self):
        """(mpi4py.MPI.Comm): The MPI communicator used (or None)"""
        return self._comm

    @property
    def n_pix(self):
        """(int): The global number of pixels."""
        return self._n_pix

    @property
    def n_pix_submap(self):
        """(int): The number of pixels in each submap."""
        return self._n_pix_submap

    @property
    def n_submap(self):
        """(int): The total number of submaps."""
        return self._n_submap

    @property
    def n_local_submap(self):
        """(int): The number of submaps stored on this process."""
        return self._n_local

    @property
    def local_submaps(self):
        """(array): The list of local submaps or None if process has no data."""
        return self._local_submaps

    @property
    def all_hit_submaps(self):
        """(array): The list of submaps local to atleast one process."""
        if self._all_hit_submaps is None:
            hits = np.zeros(self._n_submap)
            hits[self._local_submaps] += 1
            if self._comm is not None:
                self._comm.Allreduce(MPI.IN_PLACE, hits)
            self._all_hit_submaps = np.argwhere(hits != 0).ravel()
        return self._all_hit_submaps

    @property
    def global_submap_to_local(self):
        """(array): The mapping from global submap to local."""
        return self._glob2loc

    @function_timer
    def global_pixel_to_submap(self, gl):
        """Convert global pixel indices into the local submap and pixel.

        Args:
            gl (array): The global pixel numbers.

        Returns:
            (tuple):  A tuple of arrays containing the local submap index (int) and the
                pixel index local to that submap (int).

        """
        if len(gl) == 0:
            return (np.zeros_like(gl), np.zeros_like(gl))
        if np.max(gl) >= self._n_pix:
            log = Logger.get()
            msg = "Global pixel indices exceed the maximum for the pixelization"
            log.error(msg)
            raise RuntimeError(msg)
        return libtoast_global_to_local(gl, self._n_pix_submap, self._glob2loc)

        # global_sm = np.floor_divide(gl, self._n_pix_submap, dtype=np.int64)
        # submap_pixel = np.mod(gl, self._n_pix_submap, dtype=np.int64)
        # local_sm = np.array([self._glob2loc[x] for x in global_sm], dtype=np.int64)
        # return (local_sm, submap_pixel)

    @function_timer
    def global_pixel_to_local(self, gl):
        """Convert global pixel indices into local pixel indices.

        Args:
            gl (array): The global pixel numbers.

        Returns:
            (array): The local raw (flat packed) buffer index for each pixel.

        """
        if len(gl) == 0:
            return np.zeros_like(gl)
        if np.max(gl) >= self._n_pix:
            log = Logger.get()
            msg = "Global pixel indices exceed the maximum for the pixelization"
            log.error(msg)
            raise RuntimeError(msg)
        local_sm, pixels = libtoast_global_to_local(
            gl, self._n_pix_submap, self._glob2loc
        )
        local_sm *= self._n_pix_submap
        pixels += local_sm
        return pixels

    def __repr__(self):
        val = "<PixelDistribution {} pixels, {} submaps, submap size = {}>".format(
            self._n_pix, self._n_submap, self._n_pix_submap
        )
        return val

    @property
    def submap_owners(self):
        """The owning process for every hit submap.

        This information is used in several other operations, including serializing
        PixelData objects to a single process and also communication needed for
        reducing data globally.
        """
        if self._submap_owners is not None:
            # Already computed
            return self._submap_owners

        self._submap_owners = np.empty(self._n_submap, dtype=np.int32)
        self._submap_owners[:] = -1

        if self._comm is None:
            # Trivial case
            if self._local_submaps is not None and len(self._local_submaps) > 0:
                self._submap_owners[self._local_submaps] = 0
        else:
            # Need to compute it.
            local_hit_submaps = np.zeros(self._n_submap, dtype=np.uint8)
            local_hit_submaps[self._local_submaps] = 1

            hit_submaps = None
            if self._comm.rank == 0:
                hit_submaps = np.zeros(self._n_submap, dtype=np.uint8)

            self._comm.Reduce(local_hit_submaps, hit_submaps, op=MPI.LOR, root=0)
            del local_hit_submaps

            if self._comm.rank == 0:
                total_hit_submaps = np.sum(hit_submaps.astype(np.int32))
                tdist = distribute_uniform(total_hit_submaps, self._comm.size)

                # The target number of submaps per process
                target = [x[1] for x in tdist]

                # Assign the submaps in rank order.  This ensures better load
                # distribution when serializing some operations and also reduces needed
                # memory copies when using Alltoallv.
                proc_offset = 0
                proc = 0
                for sm in range(self._n_submap):
                    if hit_submaps[sm] > 0:
                        self._submap_owners[sm] = proc
                        proc_offset += 1
                        if proc_offset >= target[proc]:
                            proc += 1
                            proc_offset = 0
                del hit_submaps

            self._comm.Bcast(self._submap_owners, root=0)
        return self._submap_owners

    @property
    def owned_submaps(self):
        """The submaps owned by this process."""
        if self._owned_submaps is not None:
            # Already computed
            return self._owned_submaps
        owners = self.submap_owners
        if self._comm is None:
            self._owned_submaps = np.array(
                [x for x, y in enumerate(owners) if y == 0], dtype=np.int32
            )
        else:
            self._owned_submaps = np.array(
                [x for x, y in enumerate(owners) if y == self._comm.rank],
                dtype=np.int32,
            )
        return self._owned_submaps


class PixelData(object):
    """Distributed map-domain data.

    The distribution information is stored in a PixelDistribution instance passed to
    the constructor.  Each process has local data stored in one or more "submaps".

    Although multiple processes may have the same submap of data stored locally, only
    one process is considered the "owner".  This ownership is used when serializing the
    data and when doing reductions in certain cases.  Ownership can be set to either
    the lowest rank process which has the submap or to a balanced distribution.

    Args:
        dist (PixelDistribution):  The distribution of submaps.
        dtype (numpy.dtype):  A numpy-compatible dtype for each element of the data.
            The only supported types are 1, 2, 4, and 8 byte signed and unsigned
            integers, 4 and 8 byte floating point numbers, and 4 and 8 byte complex
            numbers.
        n_value (int):  The number of values per pixel.

    """

    def __init__(self, dist, dtype, n_value=1):
        log = Logger.get()

        self._dist = dist
        self._n_value = n_value

        # construct a new dtype in case the parameter given is shortcut string
        ttype = np.dtype(dtype)

        self.storage_class = None
        if ttype.char == "b":
            self.storage_class = AlignedI8
        elif ttype.char == "B":
            self.storage_class = AlignedU8
        elif ttype.char == "h":
            self.storage_class = AlignedI16
        elif ttype.char == "H":
            self.storage_class = AlignedU16
        elif ttype.char == "i":
            self.storage_class = AlignedI32
        elif ttype.char == "I":
            self.storage_class = AlignedU32
        elif (ttype.char == "q") or (ttype.char == "l"):
            self.storage_class = AlignedI64
        elif (ttype.char == "Q") or (ttype.char == "L"):
            self.storage_class = AlignedU64
        elif ttype.char == "f":
            self.storage_class = AlignedF32
        elif ttype.char == "d":
            self.storage_class = AlignedF64
        elif ttype.char == "F":
            raise NotImplementedError("No support yet for complex numbers")
        elif ttype.char == "D":
            raise NotImplementedError("No support yet for complex numbers")
        else:
            msg = "Unsupported data typecode '{}'".format(ttype.char)
            log.error(msg)
            raise ValueError(msg)
        self._dtype = ttype

        self.mpitype = None
        self.mpibytesize = None
        if self._dist.comm is not None:
            self.mpibytesize, self.mpitype = mpi_data_type(self._dist.comm, self._dtype)

        self._shape = (
            self._dist.n_local_submap,
            self._dist.n_pix_submap,
            self._n_value,
        )
        self._flatshape = (
            self._dist.n_local_submap * self._dist.n_pix_submap * self._n_value
        )
        self._n_submap_value = self._dist.n_pix_submap * self._n_value

        self.raw = self.storage_class.zeros(self._flatshape)
        self.data = self.raw.array().reshape(self._shape)
        self.data_jax = None

        # Allreduce quantities
        self._all_comm_submap = None
        self._all_send = None
        self._all_send_raw = None
        self._all_recv = None
        self._all_recv_raw = None

        # Alltoallv quantities
        self._send_counts = None
        self._send_displ = None
        self._recv_counts = None
        self._recv_displ = None
        self._recv_locations = None
        self.receive = None
        self._receive_raw = None
        self.reduce_buf = None
        self._reduce_buf_raw = None

    @property
    def distribution(self):
        """(PixelDistribution): The distribution information."""
        return self._dist

    @property
    def dtype(self):
        """(numpy.dtype): The data type of the values."""
        return self._dtype

    @property
    def n_value(self):
        """(int): The number of non-zero values per pixel."""
        return self._n_value

    def __getitem__(self, key):
        return np.array(self.data[key], dtype=self._dtype, copy=False)

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete individual memory elements")
        return

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        val = "<PixelData {} values per pixel, dtype = {}, dist = {}>".format(
            self._n_value, self._dtype, self._dist
        )
        return val

    def duplicate(self):
        """Create a copy of the data with the same distribution.

        Returns:
            (PixelData):  A duplicate of the instance with copied data but the same
                distribution.

        """
        dup = PixelData(self.distribution, self.dtype, n_value=self.n_value)
        dup.raw[:] = self.raw
        return dup
