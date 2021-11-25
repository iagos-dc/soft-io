import holoviews as hv
import geoviews as gv
from holoviews import opts

from common import xarray_extras
import cartopy.crs as ccrs


plate_carree = ccrs.PlateCarree()


def lon_lat_plot(da, coastline=False, hover=False):
    if da.name is None:
        da = da.rename('no_name')
    lon_label, lat_label = xarray_extras.get_lon_lat_label(da)
    ds_image = hv.Dataset(da.to_dataset())\
                 .to(gv.Image,
                     kdims=[lon_label, lat_label],
                     vdims=[da.name])
    ds_image.opts(projection=plate_carree,
                  data_aspect=1,
                  cmap='coolwarm', colorbar=True,
                  show_bounds=True,
                  )
    if hover:
        ds_image.opts(opts.Image(tools=['hover']))
    if coastline:
        ds_image = ds_image * gv.feature.coastline()
    return ds_image
