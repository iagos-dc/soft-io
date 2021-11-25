import numpy as np
import pandas as pd
import xarray as xr

from common import xarray_extras, utils
from softio.common import GEO_REGIONS, GEO_REGION, REGION_DIM
from softio.postprocess import emission_contrib_postprocess


TOP_HEIGHT = 'top_height'
BOTTOM_HEIGHT = 'bottom_height'
DELTA_HEIGHT = 'delta_height'

HEIGHT_DIM = 'height'


def get_height_ds(height_coord):
    height_dim, = height_coord.dims
    top_height = height_coord.astype('f8')
    bottom_height = xr.DataArray(np.concatenate(([0.], top_height[:-1])),
                                 coords=dict(height=height_coord), dims=[height_dim])
    height_ds = xr.Dataset({TOP_HEIGHT: top_height,
                            BOTTOM_HEIGHT: bottom_height,
                            DELTA_HEIGHT: top_height - bottom_height})
    return height_ds


def get_lon_lat_region_mask(region_da):
    """
    Produces a region_mask bool DataArray based on region_da (e.g. as stored in the variable GFED_regions_0_5deg in
    /o3p/iagos/softio/EMISSIONS/GFED3_fire_regions.nc). It is gridded with lon-lat coordinates as region_da
    and 'region' coordinates with values 0, 1, ..., 14. An extra non-dimensional coordinate 'geo_region'
    with region codes is provided.
    :param region_da: xarray DataArray gridded with lon-lat dimensions with values being region codes: 1-14 or 0.
    :return: xarray DataArray region_mask with bool values; gridded with lon-lat dimensions of region_da and
    """
    geo_regions = list(GEO_REGIONS)
    try:
        geo_regions.remove('TOTAL')
    except ValueError:
        pass
    geo_regions.insert(0, 'RESI')
    geo_region_by_id = dict(enumerate(geo_regions, start=0))
    region_da = region_da.astype('i4')
    lon_dim, lat_dim = xarray_extras.get_lon_lat_label(region_da)
    da = xr.DataArray(
        False,
        coords={REGION_DIM: np.array(list(geo_region_by_id.keys()), dtype='i4'),
                lon_dim: region_da[lon_dim],
                lat_dim: region_da[lat_dim]},
        dims=[REGION_DIM, lon_dim, lat_dim],
        name='region_mask'
    )
    da = da.assign_coords({GEO_REGION: (REGION_DIM, list(geo_region_by_id.values()))})
    da.loc[{REGION_DIM: region_da}] = True
    return da


# prepare region_mask
default_region_ds = xr.open_dataset('/o3p/iagos/softio/EMISSIONS/GFED3_fire_regions.nc').squeeze(drop=True)
default_region_mask = get_lon_lat_region_mask(default_region_ds.GFED_regions_0_5deg)


def regrid_time(da, target_time_coords):
    """
    Regrid the time coordinate of da to target_time_coords.
    Time coordinates are assumed to indicate the beginning of a time period.
    da must have time coordinates more coarse than target_time_coords
    :param da: xarray.DataArray
    :param target_time_coords: xarray.DataArray
    :return: xarray.DataArray
    """
    return da.sel({'time': target_time_coords}, method='ffill').assign_coords({'time': target_time_coords})


def get_emission_contrib_by_region(residence_time, emission, injection_rate, region_mask=None, postprocess=True,
                                   output_time_granularity=None):
    """
    NOTE: The lon-lat grid of residence_time must be the same or more coarse than that of emission and injection_rate.
    Time periods of residence_time must be the same or finer than time periods of emission (and injection rate).
    Time periods of residence_time, emission and injection_rate are represented in coordinates of 'time' dimension and
    refer to beginning time of a time period.
    Emission and injection rate must have the same coordinates in dimensions they share.
    :param residence_time: xarray DataArray containing lon-lat dimensions and height dimension; in units s m3 kg-1
    :param emission: xarray DataArray containing lon-lat dimensions and having units kg m-2 s-1
    :param injection_rate: xarray DataArray containing height dimension; in units m-1
    :param region_mask: a boolean xarray DataArray with lon-lat and region dimensions; if None (default),
    uses 'default_region_mask'
    :return: an xarray DataArray containing region dimension; in ppb
    """
    if region_mask is None:
        region_mask = default_region_mask
    res_time_units = residence_time.attrs.get('units')
    if res_time_units != 's m3 kg-1':
        raise ValueError(f'residence_time units must be s m3 kg-1; got {res_time_units}')

    # lon-lat dimension labels
    lon_dim, lat_dim = xarray_extras.get_lon_lat_label(residence_time)

    # regrid data to match a lon-lat grid of residence_time
    emission = xarray_extras.regrid_lon_lat(emission, target_resol_ds=residence_time, method='mean',
                                            longitude_circular=True, keep_attrs=True)
    try:
        xarray_extras.get_lon_lat_label(injection_rate)
        regrid_injection_rate = True
    except ValueError:
        regrid_injection_rate = False
    if regrid_injection_rate:
        injection_rate = xarray_extras.regrid_lon_lat(injection_rate, target_resol_ds=residence_time, method='mean',
                                                      longitude_circular=True, keep_attrs=True)
    region_mask = xarray_extras.regrid_lon_lat(region_mask.astype('f8'), target_resol_ds=residence_time,
                                               method='nearest', longitude_circular=True, keep_attrs=True)

    rt_time_periods = residence_time['time']
    rt_dt = abs(utils.unique(np.unique(rt_time_periods.diff('time'))))

    # apply output_time_granularity to emission and injection_rate, if set
    if output_time_granularity is not None:
        out_dt = pd.Timedelta(output_time_granularity)
        if out_dt <= pd.Timedelta(0):
            raise ValueError(f'output_time_granularity must be positive; '
                             f'got output_time_granularity={output_time_granularity}, which gives out_dt={out_dt}')
        if out_dt > pd.Timedelta('24H'):
            raise ValueError(f'output_time_granularity cannot exceed 24h; '
                             f'got output_time_granularity={output_time_granularity}, which gives out_dt={out_dt}')
        if pd.Timedelta('24H') % out_dt != pd.Timedelta(0):
            raise ValueError(f'output_time_granularity must be an integer fraction of 24h; '
                             f'got output_time_granularity={output_time_granularity}, which gives out_dt={out_dt}')
        if out_dt % rt_dt != pd.Timedelta(0):
            raise ValueError(f'output_time_granularity must be a multiple of temporal resolution of residence time={rt_dt} '
                             f'got output_time_granularity={output_time_granularity}, which gives out_dt={out_dt}')

        # regrid time coord of emission and injection_rate so that they have out_dt granularity
        if 'time' not in emission.dims:
            # add 'time' dimension to emission
            emission = emission.expand_dims({'time': np.datetime64('1900-01-01')})

        time_periods_with_out_dt = pd.date_range(start=rt_time_periods.min().values,
                                                 end=rt_time_periods.max().values + np.timedelta64(24, 'h'),
                                                 freq=output_time_granularity,
                                                 normalize=True)
        emission = regrid_time(emission, time_periods_with_out_dt)
        if 'time' in injection_rate.dims:
            injection_rate = regrid_time(injection_rate, time_periods_with_out_dt)

    # coarsen time dimension of residence_time to match that of emission or output_time_granularity, if set
    if 'time' in emission.dims:
        emission_time = emission['time']\
            .sel({'time': rt_time_periods}, method='ffill')\
            .assign_coords({'time': rt_time_periods})
        residence_time = residence_time.assign_coords({'emission_time': emission_time})
        residence_time = residence_time.groupby('emission_time').sum('time').rename({'emission_time': 'time'})
    else:
        residence_time = residence_time.sum('time')

    # compute contribution
    contribution = ((residence_time * injection_rate).sum(HEIGHT_DIM)
                    * emission * region_mask).sum([lon_dim, lat_dim]) * 1e9
    # must load DaskArray to numpy array, because otherwise
    # contrib_da.loc[{GEO_REGION: 'TOTAL'}] = contrib_da.sum(GEO_REGION)
    # is not available in emission_contrib_postprocess
    contribution = contribution.load()

    # setup metadata
    contribution.name = 'emission_contrib'
    contribution.attrs = dict(
        standard_name='mixing_ratio',
        long_name='Emission contribution from region',
        units='1e-9',
    )

    if postprocess:
        contribution = emission_contrib_postprocess(contribution)
    return contribution


def get_injection_rate(inj_height, height_coord):
    """
    Calculates injection rate (in m-1) based on injection height.
    The integral of the injection rate wrt height (in m) is 1.
    :param inj_height: xarray DataArray or a scalar with injection height (in m above the ground).
    :param height_coord: xarray DataArray with height of upper layers of vertical grid cells
    :return: xarray DataArray
    """
    height_ds = get_height_ds(height_coord)
    first_height = height_ds[TOP_HEIGHT][0].item()
    inj_height_corrected = np.maximum(inj_height, first_height)
    inj_coeff = (inj_height_corrected - height_ds[BOTTOM_HEIGHT]) / height_ds[DELTA_HEIGHT]
    inj_coeff = inj_coeff.clip(0., 1.)
    return inj_coeff / inj_height_corrected
