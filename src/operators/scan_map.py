
import traitlets

import numpy as np
from ..utils import Logger, AlignedF64
from ..traits import trait_docs, Int, Unicode, Bool
from ..timing import function_timer
from ..pixels import PixelDistribution, PixelData
from ..observation import default_values as defaults
from .operator import Operator

import jax.numpy as jnp
import jax

def scan_map(mapdata, nmap, submap, subpix, weights):
    """
    Sample a map into a timestream.

    This uses a local piece of a distributed map and the local pointing matrix
    to generate timestream values.

    Args:
        mapdata (Pixels):  The local piece of the map.
        nmap (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only submap)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap.
        weights (array, float64):  The pointing matrix weights (size: nsample*nmap).

    Returns:
        tod (array, float64):  The timestream on which to accumulate the map values.
    """
    # TODO the following computations should not be needed if we take only jax arrays as inputs
    # gets number of pixels in each submap
    npix_submap = mapdata.distribution.n_pix_submap
    # converts mapdata to a jax array
    mapdata = mapdata.raw.array()

    # turns mapdata into an array of shape nsamp*nmap
    mapdata = jnp.reshape(mapdata, newshape=(-1,npix_submap,nmap)) # TODO this should not be needed
    submapdata = mapdata[submap,subpix,:]

    # zero-out samples with invalid indices
    # by default JAX will put any value where the indices were invalid instead of erroring out
    valid_samples = (subpix >= 0) & (submap >= 0)
    submapdata = jnp.where(valid_samples[:,jnp.newaxis], submapdata, 0.0)

    # does the computation
    tod = jnp.sum(submapdata * weights, axis=1)
    return tod

#--------------------------------------------------------------------------------------------------

class ScanMap(Operator):
    """
    Operator which uses the pointing matrix to scan timestream values from a map.

    The map must be a PixelData instance with either float32 or float64 values.  The
    values can either be accumulated or subtracted from the input timestream, and the
    input timestream can be optionally zeroed out beforehand.
    """
    API = Int(0, help="Internal interface version for this operator")
    det_data = Unicode(defaults.det_data, help="Observation detdata key for the timestream data")
    view = Unicode(None, allow_none=True, help="Use this view of the data in all observations")
    pixels = Unicode(defaults.pixels, help="Observation detdata key for pixel indices")
    weights = Unicode(defaults.weights, allow_none=True, help="Observation detdata key for Stokes weights")
    map_key = Unicode(None, allow_none=True, help="The Data key where the map is located")
    subtract = Bool(False, help="If True, subtract the map timestream instead of accumulating")
    zero = Bool(False, help="If True, zero the data before accumulating / subtracting")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        map_data = data[self.map_key]
        map_dist = map_data.distribution

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)

            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                view_samples = None
                if vw.start is None:
                    # This is a view of the whole obs
                    view_samples = ob.n_local_samples
                else:
                    view_samples = vw.stop - vw.start

                # Temporary array, re-used for all detectors
                maptod_raw = AlignedF64.zeros(view_samples)
                maptod = maptod_raw.array()

                for det in dets:
                    # The pixels, weights, and data.
                    pix = views.detdata[self.pixels][ivw][det]
                    if self.weights is None:
                        wts = np.ones(pix.size, dtype=np.float64)
                    else:
                        wts = views.detdata[self.weights][ivw][det]
                    ddata = views.detdata[self.det_data][ivw][det]

                    # Get local submap and pixels
                    local_sm, local_pix = map_dist.global_pixel_to_submap(pix)

                    maptod[:] = 0.0

                    scan_map(map_data, 
                             map_data.n_value, 
                             local_sm.astype(np.int64), 
                             local_pix.astype(np.int64), 
                             wts.astype(np.float64), 
                             maptod)

                    # zero-out if needed
                    if self.zero:
                        ddata[:] = 0.0

                    # Add or subtract.  Note that the map scanned timestream will have
                    # zeros anywhere that the pointing is bad, but those samples (and
                    # any other detector flags) should be handled at other steps of the
                    # processing.
                    if self.subtract:
                        ddata[:] -= maptod
                    else:
                        ddata[:] += maptod


