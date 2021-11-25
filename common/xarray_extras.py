import numpy as np
import xarray as xr
import pathlib

from common.log import logger
from common.utils import short_dataset_repr
import common.longitude


def get_lon_label(ds):
    ds_dims = ds.dims
    if 'longitude' in ds_dims:
        label = 'longitude'
    elif 'lon' in ds_dims:
        label = 'lon'
    else:
        raise ValueError('neither "longitude" nor "lon" dimension found in ds')
    return label


def get_lat_label(ds):
    ds_dims = ds.dims
    if 'latitude' in ds_dims:
        label = 'latitude'
    elif 'lat' in ds_dims:
        label = 'lat'
    else:
        raise ValueError('neither "latitude" nor "lat" dimension found in ds')
    return label


def get_lon_lat_label(ds):
    return get_lon_label(ds), get_lat_label(ds)


def dropna_and_flatten(da):
    # alternative method for many variables:
    # ds.where(ds[v] > 0).to_dataframe().dropna(how='all', subset=list(ds)).reset_index()
    if not isinstance(da, xr.DataArray):
        raise TypeError(f'da must be an xarray.DataArray; got {type(da)}')
    # TODO: make it lazy (values -> data);
    # but then a lack of explicit sizes of idxs is a problem when constructing an xarray Dataset
    idxs = np.nonzero(da.notnull().values)
    idxs = (('idx', idx) for idx in idxs)
    idxs = xr.Dataset(dict(zip(da.dims, idxs)))
    return da.isel(idxs)


def normalize_longitude(ds, lon_label=None, smallest_lon_coord=-180., keep_attrs=False):
    if lon_label is None:
        lon_label = get_lon_label(ds)
    lon_coords = ds[lon_label]
    aligned_lon_coords = common.longitude.normalize_longitude(lon_coords, smallest_lon_coord=smallest_lon_coord)
    if keep_attrs:
        aligned_lon_coords = aligned_lon_coords.assign_attrs(lon_coords.attrs)

    if not lon_coords.equals(aligned_lon_coords):
        old_lon_coords_monotonic = lon_coords.indexes[lon_label].is_monotonic_increasing
        ds = ds.assign_coords({lon_label: aligned_lon_coords})
        if old_lon_coords_monotonic:
            smallest_lon_idx = aligned_lon_coords.argmin(dim=lon_label).item()
            ds = ds.roll(shifts={lon_label: -smallest_lon_idx}, roll_coords=True)
        else:
            ds = ds.sortby(lon_label)
    return ds


def open_dataset_from_netcdf_with_disk_chunks(url, chunks='auto', **kwargs):
    """
    Open a dataset from a netCDF file using on disk chunking. The parameter 'chunks' mimics xarray.open_zarr behaviour.
    :param url: str; path to a netCDF file.
    :param chunks: 'auto' or dict or None; default 'auto'. If 'auto', open with chunks on disk; if a dictionary is given,
    it must map dimensions into chunks sizes (size -1 means the whole dimension length); the dictionary updates chunk
    sizes found in the file; if None, open without chunking.
    :param kwargs: extra keyword arguments passed to xarray.open_dataset
    :return: xarray Dataset
    """
    if chunks is not None:
        with xr.open_dataset(url) as ds:
            chunks_by_dim = {}
            dims = ds.dims
            for v in list(ds.data_vars) + list(ds.coords):
                if v not in dims:
                    v_dims = ds[v].sizes
                    v_chunks = ds[v].encoding.get('chunksizes')
                    if v_chunks is not None:
                        if len(v_dims) == len(v_chunks):
                            for dim, chunk in zip(v_dims, v_chunks):
                                chunks_by_dim.setdefault(dim, {})[v] = chunk
                        else:
                            logger().warning(f'variable {v}: sizes={v_dims}, chunks={v_chunks}; '
                                             f'ignoring the chunks specification for this variable')
        chunk_by_dim = {}
        for d, chunks_by_var in chunks_by_dim.items():
            chunk = min(chunks_by_var.values())
            if chunk < dims[d]:
                chunk_by_dim[d] = chunk
        if chunks != 'auto':
            chunk_by_dim.update(chunks)
        if all(chunk_by_dim[d] in (dims[d], -1) for d in chunk_by_dim):
            chunk_by_dim = None
        ds = xr.open_dataset(url, chunks=chunk_by_dim, **kwargs)
    else:
        ds = xr.open_dataset(url, **kwargs)
    return ds


def open_dataset_with_disk_chunks(url, chunks='auto', **kwargs):
    """
    Open a dataset from a netCDF or zarr file using on disk chunking.
    The parameter 'chunks' mimics xarray.open_zarr behaviour.
    :param url: str; path to a netCDF or zarr file.
    :param chunks: 'auto' or dict or None; default 'auto'. If 'auto', open with chunks on disk; if a dictionary is given,
    it must map dimensions into chunks sizes (size -1 means the whole dimension length); the dictionary updates chunk
    sizes found in the file; if None, open without chunking.
    :param kwargs: extra keyword arguments passed to xarray.open_dataset
    :return: xarray Dataset
    """
    fmt = pathlib.PurePath(url).suffix
    if fmt == '.nc':
        return open_dataset_from_netcdf_with_disk_chunks(url, chunks=chunks, **kwargs)
    elif fmt == '.zarr':
        return xr.open_zarr(url, chunks=chunks, **kwargs)
    else:
        raise ValueError(f'unknown format: {fmt}; must be .nc or .zarr')


def make_coordinates_increasing(ds, coord_labels, allow_sorting=True):
    """
    Sorts coordinates
    :param ds: an xarray Dataset or DataArray
    :param coord_labels: a string or an interable of strings - labels of dataset's coordinates
    :param allow_sorting: bool; default True; indicate if sortby method is allowed to be used as a last resort
    :return: ds with a chosen coordinate(s) in increasing order
    """
    if isinstance(coord_labels, str):
        coord_labels = (coord_labels, )
    for coord_label in coord_labels:
        if not ds.indexes[coord_label].is_monotonic_increasing:
            if ds.indexes[coord_label].is_monotonic_decreasing:
                ds = ds.isel({coord_label: slice(None, None, -1)})
            elif allow_sorting:
                ds = ds.sortby(coord_label)
            else:
                raise ValueError(f'{short_dataset_repr(ds)} has coordinate {coord_label} which is neither increasing nor decreasing')
    return ds


def is_coord_regularly_gridded(coord, abs_err):
    """
    Checks if a coordinate variable is regularly gridded (spaced)
    :param coord: a 1-dim array-like
    :param abs_err: float or timedelta; maximal allowed absolute error when checking for equal spaces
    :return: bool
    """
    coord = np.asanyarray(coord)
    if len(coord.shape) != 1:
        raise ValueError(f'coord must be 1-dimensional')
    err = np.abs(np.diff(coord, n=2))
    return np.all(err <= abs_err)


def regrid(ds, target_coords, method='mean', tolerance=1e-3, skipna=None, keep_attrs=False, **agg_method_kwargs):
    """
    Regrid coordinates of a dataset (or a data array). Coordinates are assumed to be a center of a grid cell.
    Coarser grids are obtained from regular aggregation; to this end, both the initial and target grids must
    each be equally spaced and the target grid must be more coarsed than the initial one. If method is 'linear'
    or 'nearest', a simple re-sampling via interpolation is applied.

    :param ds: an xarray Dataset or DataArray
    :param target_coords: dict with keys being dimensions of ds to regrid and values being new coordinates in a form of
    an array-like object
    :param method: 'mean', 'sum', 'max', 'min', etc.
    (see http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
    for an exhaustive list), or 'linear' or 'nearest' for re-sampling
    :param tolerance: float or numpy.timedelta64; default=1e-3; a tolerance for checking coordinates alignment, etc.
    :param skipna: bool, optional; default behaviour is to skip NA values if they are of float type; for more see
    http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
    :param keep_attrs: bool, optional; If True, the attributes (attrs) will be copied from the original object
    to the new one. If False (default), the new object will be returned without attributes.
    :param agg_method_kwargs: keyword arguments passed to an aggregation method; see
    http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
    :return: same type as ds
    """
    # check if target dimensions are contained in dimensions of ds
    ds_dims = ds.dims
    for coord_label in target_coords:
        if coord_label not in ds_dims:
            raise ValueError(f'{coord_label} not found among ds dimensions: {list(ds_dims)}')

    ds = make_coordinates_increasing(ds, target_coords.keys())

    if method in ['linear', 'nearest']:
        interpolator_kwargs = {'fill_value': 'extrapolate'} if method == 'nearest' else None
        regridded_ds = ds.interp(coords=target_coords, method=method, assume_sorted=True, kwargs=interpolator_kwargs)
    else:
        # check if target coordinates are equally spaced
        for coord_label, target_coord in target_coords.items():
            if not is_coord_regularly_gridded(target_coord, tolerance):
                raise ValueError(f'{coord_label} is not be regularly gridded: {target_coord}')
        # check if coordinates of ds are equally spaced
        for coord_label in target_coords:
            if not is_coord_regularly_gridded(ds[coord_label], tolerance):
                raise ValueError(f'ds has {coord_label} coordinate not regularly gridded: {ds[coord_label]}')

        # trim the domain of ds to target_coords, if necessary
        trim_by_dim = {}
        for coord_label, target_coord in target_coords.items():
            target_coord = np.asanyarray(target_coord)
            n_target_coord, = target_coord.shape
            target_coord_min, target_coord_max = target_coord.min(), target_coord.max()
            if n_target_coord >= 2:
                step = (target_coord_max - target_coord_min) / (n_target_coord - 1)
                if ds[coord_label].min() < target_coord_min - step / 2 or \
                        ds[coord_label].max() > target_coord_max + step / 2:
                    trim_by_dim[coord_label] = slice(target_coord_min - step / 2, target_coord_max + step / 2)
        if trim_by_dim:
            ds = ds.sel(trim_by_dim)

        # coarsen
        window_size = {}
        for coord_label, target_coord in target_coords.items():
            if len(ds[coord_label]) % len(target_coord) != 0:
                raise ValueError(f'resolution of {coord_label} not compatible: '
                                 f'{len(ds[coord_label])} must be a multiple of {len(target_coord)}\n'
                                 f'ds[{coord_label}] = {ds[coord_label]}\n'
                                 f'target_coord = {target_coord}')
            window_size[coord_label] = len(ds[coord_label]) // len(target_coord)
        coarsen_kwargs = {'keep_attrs': keep_attrs} if keep_attrs is not None else {}
        coarsen_ds = ds.coarsen(dim=window_size, boundary='exact', coord_func='mean', **coarsen_kwargs)
        coarsen_ds_agg_method = getattr(coarsen_ds, method)
        if skipna is not None:
            agg_method_kwargs['skipna'] = skipna
        if keep_attrs is not None:
            agg_method_kwargs['keep_attrs'] = keep_attrs
        if isinstance(coarsen_ds, xr.core.rolling.DataArrayCoarsen):
            agg_method_kwargs.pop('keep_attrs', None)
        regridded_ds = coarsen_ds_agg_method(**agg_method_kwargs)

        # adjust coordinates of regridded_ds so that they fit to target_coords
        try:
            regridded_ds = regridded_ds.sel(target_coords, method='nearest', tolerance=tolerance)
        except KeyError:
            raise ValueError(f"target grid is not compatible with a source grid; "
                             f"check grids or adjust 'tolerance' parameter\n"
                             f"regridded_ds={regridded_ds}\n"
                             f"target_coords={target_coord}")
        regridded_ds = regridded_ds.assign_coords({r_dim: target_coords[r_dim]
                                                   for r_dim in set(regridded_ds.dims).intersection(target_coords)})
    return regridded_ds


def regrid_lon_lat(ds, target_resol_ds=None, longitude=None, latitude=None, method='mean', tolerance=1e-3,
                   longitude_circular=None, skipna=None, keep_attrs=False, **agg_method_kwargs):
    """
    Regrid longitude and latitude coordinates of a dataset (or a data array). Coordinates are assumed to be a center
    of a grid cell. Coarser grids are obtained from regular aggregation; to this end, both the initial and target grids
    must each be equally spaced and the target grid must be more coarsed than the initial one. If method is 'linear'
    or 'nearest', a simple re-sampling via interpolation is applied.

    :param ds: an xarray Dataset or DataArray
    :param target_resol_ds: an xarray Dataset with new longitude and latitude coordinates;
    alternatively (and exclusively) longitude and latitude parameters can be given
    :param longitude: an array-like; optional
    :param latitude: an array-like; optional
    :param method: 'mean', 'sum', 'max', 'min', etc.
    (see http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
    for an exhaustive list), or 'linear' or 'nearest' for re-sampling
    :param tolerance: float, default=1e-3; a tolerance for checking coordinates alignment, etc.
    :param longitude_circular: bool, optional; if True then then longitude coordinates are considered as circular
    :param skipna: bool, optional; default behaviour is to skip NA values if they are of float type; for more see
    http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
    :param keep_attrs: bool, optional; If True, the attributes (attrs) will be copied from the original object
    to the new one. If False (default), the new object will be returned without attributes.
    :param agg_method_kwargs: keyword arguments passed to an aggregation method; see
    http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
    :return: same type as ds
    """
    # prepare target longitude and latitude coordinates
    if target_resol_ds is None and (longitude is None or latitude is None):
        raise ValueError('target_resol_ds or longitude and latitude must be set')
    if target_resol_ds is not None and (longitude is not None or latitude is not None):
        raise ValueError('either target_resol_ds or longitude and latitude must be set, but not both')
    if target_resol_ds is not None:
        lon, lat = get_lon_lat_label(target_resol_ds)
        longitude = target_resol_ds[lon]
        latitude = target_resol_ds[lat]
    # get labels of longitude and latitude dimension for ds
    lon_label, lat_label = get_lon_lat_label(ds)

    # handle overlapping longitude coordinate if necessary
    longitude_ori = None
    if longitude_circular:
        # remove target longitude coordinate which is overlapping mod 360
        if abs(abs(longitude[-1] - longitude[0]) - 360.) <= tolerance:
            longitude_ori = longitude
            longitude = longitude[:-1]
        # remove ds' longitude coordinate which is overlapping mod 360
        if abs(abs(ds[lon_label][-1] - ds[lon_label][0]) - 360.) <= tolerance:
            ds = ds.isel({lon_label: slice(None, -1)})

    # if necessary, align longitude coordinates of ds to the target longitude by normalizing and rolling or sorting
    smallest_longitude_coord = (np.amin(np.asanyarray(longitude)) + np.amax(np.asanyarray(longitude)) - 360.) / 2
    ds = normalize_longitude(ds, lon_label=lon_label, smallest_lon_coord=smallest_longitude_coord, keep_attrs=True)

    # duplicate left- and right-most longitude coordinate to facilitate interpolation, if necessary
    if method in ['linear', 'nearest'] and longitude_circular:
        lon_coord = ds[lon_label].values
        extended_lon_coord = np.concatenate(([lon_coord[-1]], lon_coord, [lon_coord[0]]))
        extended_lon_coord_normalized = np.array(extended_lon_coord)
        extended_lon_coord_normalized[0] = extended_lon_coord_normalized[0] - 360.
        extended_lon_coord_normalized[-1] = extended_lon_coord_normalized[-1] + 360.
        lon_attrs = ds[lon_label].attrs
        ds = ds\
            .sel({lon_label: extended_lon_coord})\
            .assign_coords({lon_label: extended_lon_coord_normalized})
        ds[lon_label].attrs = lon_attrs

    # do regridding
    ds = regrid(ds, {lon_label: longitude, lat_label: latitude}, method=method, tolerance=tolerance,
                skipna=skipna, keep_attrs=keep_attrs, **agg_method_kwargs)

    # re-establish longitude coordinate which overlaps mod 360
    if longitude_circular and longitude_ori is not None:
        lon_idx = np.arange(len(longitude_ori))
        lon_idx[-1] = 0
        ds = ds.isel({lon_label: lon_idx}).assign_coords({lon_label: longitude_ori})

    return ds


def drop_duplicated_coord(ds, dim):
    _, idx = np.unique(ds[dim], return_index=True)
    if len(idx) != len(ds[dim]):
        ds = ds.isel({dim: idx})
    return ds
