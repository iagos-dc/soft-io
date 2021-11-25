import xarray as xr

from common import xarray_extras as xrx
import softio.core


GFAS_INJ_HEIGHT = 'apt'
GFAS_EMISSION_FLUX = 'cofire'


# prepare gfas_ds
gfas_v1_2_urls = [
    '/o3p/iagos/softio/EMISSIONS/GFASv1.2_CO_2003_2017.nc',
    '/o3p/iagos/softio/EMISSIONS/GFASv1.2_CO_20171031_20180130.nc',
    '/o3p/iagos/softio/EMISSIONS/GFASv1.2_CO_20171130_20200530.nc',
    '/o3p/iagos/softio/EMISSIONS/GFASv1.2_CO_since_20191130.nc'
]

gfas_ds = {}
for i, url in enumerate(gfas_v1_2_urls):
    gfas_ds[i] = xrx.open_dataset_with_disk_chunks(url, chunks={'longitude': -1, 'latitude': -1})
    gfas_ds[i] = xrx.drop_duplicated_coord(gfas_ds[i], 'time')
    if i > 0:
        min_t = gfas_ds[i].time.min()
        gfas_ds[i - 1] = gfas_ds[i - 1].where(gfas_ds[i - 1].time < min_t, drop=True)
gfas_ds = xr.concat(gfas_ds.values(), 'time')


def get_softio_gfas_results(fp_rt, time_granularity=None):
    apt_coarsen_with_max = xrx.regrid_lon_lat(gfas_ds[GFAS_INJ_HEIGHT], target_resol_ds=fp_rt,
                                              method='max', longitude_circular=True, keep_attrs=True)
    injection_rate = softio.core.get_injection_rate(apt_coarsen_with_max, fp_rt.height)
    emission_contrib = softio.core.get_emission_contrib_by_region(fp_rt,
                                                                  gfas_ds[GFAS_EMISSION_FLUX],
                                                                  injection_rate,
                                                                  output_time_granularity=time_granularity)
    return emission_contrib
