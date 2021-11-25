from common import xarray_extras as xrx
import softio.core


CEDS2_EMISSION_FLUX = 'CO_em_anthro'


# prepare ceds2_ds
ceds2_url = '/o3p/wolp/CEDSv2/CO-em-anthro_input4MIPs_emissions_CMIP_CEDS-2021-04-21_gn_200001-201912.nc'
ceds2_ds = xrx.open_dataset_with_disk_chunks(ceds2_url, chunks={'sector': -1, 'lon': -1, 'lat': -1})
ceds2_ds = ceds2_ds.assign_coords({'time': ceds2_ds['time_bnds'].isel({'bound': 0}).astype('M8[ns]')})
ceds2_ds = ceds2_ds.sum('sector')


def get_softio_ceds2_results(fp_rt, time_granularity=None):
    injection_rate = softio.core.get_injection_rate(1000., fp_rt.height)
    emission_contrib = softio.core.get_emission_contrib_by_region(fp_rt,
                                                                  ceds2_ds[CEDS2_EMISSION_FLUX],
                                                                  injection_rate,
                                                                  output_time_granularity=time_granularity)
    return emission_contrib
