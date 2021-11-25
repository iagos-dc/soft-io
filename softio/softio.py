import glob
import numpy as np
import pandas as pd
import xarray as xr
import pathlib

from common import xarray_extras as xrx
from common import utils
import fpsim
import softio.core
import softio.gfas
import softio.maccity


# prepare mean air density
vertical_profile_of_air_density = pd.read_csv('/home/wolp/data/vertical_profile_of_air_density.csv', index_col='height_in_m', dtype=np.float32)['density_in_kg_per_m3']
vertical_profile_of_air_density = xr.DataArray.from_series(vertical_profile_of_air_density).rename({'height_in_m': 'height'})
vertical_profile_of_air_density['height'] = vertical_profile_of_air_density.height.astype('f4')
vertical_profile_of_air_density.name = 'air_density'
vertical_profile_of_air_density.attrs = dict(long_name='air_density', units='kg m-3')


def rt_transform_by_air_density(rt, oro):
    height_ds = softio.core.get_height_ds(rt['height'])
    top_height = height_ds[softio.core.TOP_HEIGHT]
    bottom_height = height_ds[softio.core.BOTTOM_HEIGHT]
    mean_height = 0.5 * (top_height + bottom_height)
    density = vertical_profile_of_air_density.interp(coords={'height': mean_height + oro}).astype(np.float32)
    rt = rt / density
    rt.attrs['units'] = 's m3 kg-1'
    return rt


def get_co_contrib(emission_inventory, fpsim_ds=None, fpsim_dir=None, time_granularity=None):
    """

    :param fpsim_dir:
    :param emission_inventory: str; possible values: 'gfas', 'maccity', 'ceds2'
    :return:
    """
    if fpsim_ds is None and fpsim_dir is None or fpsim_ds is not None and fpsim_dir is not None:
        raise ValueError('either fpsim_ds or fpsim_dir must be set')
    if fpsim_ds is None:
        output_dir = pathlib.PurePath(fpsim_dir, 'output')
        # grid_time_url = utils.unique(glob.glob(str(pathlib.PurePath(output_dir, 'grid_time_*.nc'))))
        grid_time_url = utils.unique(glob.glob(str(pathlib.PurePath(output_dir, 'grid_time_*'))))  # accepting .zarr too
        with fpsim.open_fp_dataset(str(grid_time_url)) as fpsim_ds:
            return get_co_contrib(emission_inventory, fpsim_ds=fpsim_ds, time_granularity=time_granularity)
    else:
        fpsim_ds = xrx.normalize_longitude(fpsim_ds)
        rt = fpsim_ds['spec001_mr']
        rt_units = rt.attrs['units']
        ind_source = int(fpsim_ds.attrs['ind_source'])
        ind_receptor = int(fpsim_ds.attrs['ind_receptor'])

        if ind_source == 2 and ind_receptor == 2:
            # must change residence time units from 's' to 's m3 kg-1' by dividing by air density at output grid cell
            rt = rt_transform_by_air_density(rt, fpsim_ds['ORO'])
        elif not (ind_source == 1 and ind_receptor == 2):
            raise ValueError(f'FLEXPART output with ind_source={ind_source}, ind_receptor={ind_receptor} '
                             f'and residence time units={rt_units} cannot be handled')

        emission_inventory = emission_inventory.lower()
        if emission_inventory == 'gfas':
            return softio.gfas.get_softio_gfas_results(rt, time_granularity=time_granularity)
        if emission_inventory == 'macc':
            return softio.maccity.get_softio_maccity_results(rt, time_granularity=time_granularity)
        if emission_inventory == 'ceds2':
            return softio.ceds2.get_softio_ceds2_results(rt, time_granularity=time_granularity)
        else:
            raise ValueError(f'unknown emission inventory: {emission_inventory}')
