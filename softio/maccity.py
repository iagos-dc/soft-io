from common import xarray_extras as xrx
import softio.core


MACCITY_EMISSION_FLUX = 'emiss_sum'


# prepare maccity_ds
maccity_url = '/o3p/iagos/softio/EMISSIONS/MACCity-anthro_Glb_0.5x0.5_anthro_CO_1960_2020.nc'
maccity_ds = xrx.open_dataset_with_disk_chunks(maccity_url) #, chunks={'lon': -1, 'lat': -1})


def get_softio_maccity_results(fp_rt, time_granularity=None):
    injection_rate = softio.core.get_injection_rate(1000., fp_rt.height)
    emission_contrib = softio.core.get_emission_contrib_by_region(fp_rt,
                                                                  maccity_ds[MACCITY_EMISSION_FLUX],
                                                                  injection_rate,
                                                                  output_time_granularity=time_granularity)
    return emission_contrib
