import xarray as xr
from softio.common import GEO_REGION, CO_CONTRIB, REGION_DIM


def emission_contrib_postprocess(contrib_da):
    contrib_da = contrib_da * 28.97 / 28.  # molar weight adjustment (see reademissions_gfas.f90:543)
    contrib_da[GEO_REGION] = xr.where(contrib_da[GEO_REGION] == 'RESI', 'TOTAL', contrib_da[GEO_REGION])
    contrib_da = contrib_da\
        .set_index({REGION_DIM: GEO_REGION})\
        .rename({REGION_DIM: GEO_REGION})
    # .assign_coords({'geo_region': contrib_da['geo_region']})\

    contrib_da.loc[{GEO_REGION: 'TOTAL'}] = contrib_da.sum(GEO_REGION)
    return xr.Dataset({CO_CONTRIB: contrib_da})

