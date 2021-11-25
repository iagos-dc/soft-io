import os
import itertools
import hashlib
import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable, Mapping
from collections import OrderedDict
import json
import tqdm
import dask.bag

from common.log import logger


def short_dataset_repr(ds):
    if isinstance(ds, xr.Dataset):
        ds_id = ds.attrs.get('id')
        ds_vars = list(ds)
        ds_repr = f'Dataset(id={ds_id}, vars={ds_vars})'
    elif isinstance(ds, xr.DataArray):
        ds_name = ds.name
        for attr in ['standard_name', 'short_name', 'name', 'long_name']:
            if ds_name is not None:
                break
            ds_name = ds.attrs.get(attr)
        ds_repr = f'DataArray({ds_name})'
    else:
        ds_repr = str(ds)
    return ds_repr


def short_time_repr(t, fmt='%Y-%m-%d %H:%M:%S'):
    if isinstance(t, xr.DataArray):
        t = t.values
    t = np.array(t)
    if len(t.shape) > 1:
        raise ValueError(f't must be either a scalar or a 1-d array; got t.shape={t.shape}')
    if len(t.shape) == 0:
        return pd.Timestamp(t.item()).strftime(fmt)
    else:
        return [pd.Timestamp(t_item).strftime(fmt) for t_item in t]


class Proxy:
    def __init__(self, fun):
        self._fun = fun

    def __call__(self, *args, **kwargs):
        return lambda: self._fun(*args, **kwargs)


def proxy(func):
    def decorated_func(*args, **kwargs):
        return lambda: func(*args, **kwargs)
    return decorated_func


def map_collection(func, collection, ignore_errors=False):
    collection_type = type(collection)
    if isinstance(collection, Mapping):
        if not ignore_errors:
            d = {k: func(v) for k, v in collection.items()}
        else:
            d = {}
            for k, v in collection.items():
                try:
                    d[k] = func(v)
                except Exception as e:
                    logger().exception(f'error while applying {func} to {v} for key={k}', exc_info=e)
        return collection_type(d)
    elif isinstance(collection, Iterable):
        if not ignore_errors:
            c = map(func, collection)
        else:
            c = []
            for v in collection:
                try:
                    c.append(func(v))
                except Exception as e:
                    logger().exception(f'error while applying {func} to {v}', exc_info=e)
        return collection_type(c)
    else:
        raise ValueError(f'collection must be iterable while its type is {collection_type}')


def unique(an_iterable):
    iterator = iter(an_iterable)
    try:
        item = next(iterator)
    except StopIteration:
        raise ValueError('an iterable has no elements')
    try:
        another_item = next(iterator)
        raise ValueError(f'an iterable has more than one element: it has {[item, another_item] + list(iterator)}')
    except StopIteration:
        return item


def groupby(an_iterable, key):
    """
    Group an iterable by a key

    :param an_iterable: an iterable to group
    :param key: a callable
    :return: a dictionary (key, list of items)
    """

    group_by_key = {}
    for item in an_iterable:
        k = key(item)
        group_by_key.setdefault(k, []).append(item)
    return group_by_key


def hash_of_xarray_obj_dim_coords(xarray_obj):
    """
    Returns a hash code (as a 32-length hex code, i.e. string of length 64) of coordinates for each dimension
    of an xarray object (Dataset or DataArray)
    :param xarray_obj: an xarray Dataset or DataArraywith
    :return: a dictionary dim: str containing a hash code
    """
    def hash_of_pandas_index(idx):
        return hashlib.sha256(pd.util.hash_pandas_object(idx).values).hexdigest()
    return {dim: hash_of_pandas_index(idx) for dim, idx in xarray_obj.indexes.items()}


class IteratorSplitter:
    def __init__(self, an_iterable):
        self.iterator = iter(an_iterable)
        self.began = False
        self.finished = False

    def _try_next(self):
        try:
            self.current_item = next(self.iterator)
        except StopIteration:
            self.finished = True

    def _init_if_not_began(self):
        if not self.began:
            self.began = True
            self._try_next()

    def peek(self):
        self._init_if_not_began()
        if self.finished:
            raise StopIteration
        return self.current_item

    def take_while(self, predicate):
        self._init_if_not_began()
        while not self.finished and predicate(self.current_item):
            yield self.current_item
            self._try_next()


def numpy_1d_array_has_non_decreasing_numerical_values(arr):
    return (np.diff(arr) >= 0.).all()


def numpy_1d_array_has_strictly_increasing_numerical_values(arr):
    return (np.diff(arr) > 0.).all()


def numpy_1d_array_has_non_decreasing_datetime_values(arr):
    return (np.diff(arr) >= np.timedelta64(0, 's')).all()


def numpy_1d_array_has_strictly_increasing_datetime_values(arr):
    return (np.diff(arr) > np.timedelta64(0, 's')).all()


def optional_arg_helper(arg):
    return (arg,) if arg is not None else ()


def fun_args_to_string(*args, **kwargs):
    fun_args = dict(args=args, kwargs=kwargs)
    try:
        res = json.dumps(fun_args)
    except Exception as e:
        logger().exception(f'cannot dump fun_args={fun_args} to string with json; dumping None instead', exc_info=e)
        res = json.dumps(None)
    return res


def fun_args_from_string(s):
    try:
        res = json.loads(s)
    except Exception as e:
        logger().exception(f'cannot parse json={s}; return None as a parse result instead', exc_info=e)
        res = None
    return res


def yearmonth_check(yearmonth):
    if not isinstance(yearmonth, str) or len(yearmonth) != 6:
        return None
    try:
        year = int(yearmonth[:4])
        month = int(yearmonth[4:])
    except ValueError:
        return None
    if 1970 <= year <= 2100 and 1 <= month <= 12:
        return year, month
    else:
        return None


def yearmonth_filter(s, years=None, months=None, yearmonths=None):
    """
    Checks if the string represents YYYYMM and if so, wheter it falls into criteria
    :param s: string
    :param years: list or pandas.Interval, optional
    :param months: list, optional
    :param yearmonths: list, optional
    :return: bool
    """
    yearmonth_check_result = yearmonth_check(s)
    if yearmonth_check_result is None:
        return False
    year, month = yearmonth_check_result
    if yearmonths and (s not in yearmonths):
        return False
    if years and (year not in years):
        return False
    if months and (month not in months):
        return False
    return True


def walk_direntries(path, filter_fun=None, aux_entry_fun=None, sort=True, progress_bar=False):
    # TODO: remove progress_bar from here
    with os.scandir(path) as direntries:
        direntries_filtered = filter(filter_fun, direntries) if filter_fun is not None else direntries
        direntries_sorted = sorted(direntries_filtered, key=lambda entry: entry.name) if sort else list(direntries_filtered)

    if progress_bar:
        direntries_sorted = tqdm.tqdm(direntries_sorted)

    for direntry in direntries_sorted:
        if aux_entry_fun is not None:
            yield direntry, aux_entry_fun(direntry)
        else:
            yield direntry


def walk_yearmonths(path, years=None, months=None, yearmonths=None):
    for yearmonth_direntry in \
            walk_direntries(path,
                            filter_fun=lambda entry: entry.is_dir() and
                                                     yearmonth_filter(entry.name, years=years, months=months, yearmonths=yearmonths)):
        yield yearmonth_direntry.name, yearmonth_direntry.path


def parallel_for_each_bag(iterable_of_args, action, dask_client=None, npartitions=16):
    if dask_client is None:
        results = [action(*args) for args in iterable_of_args]
    else:
        bag = dask.bag.from_sequence(iterable_of_args, npartitions=npartitions)
        results = bag.map(lambda args: action(*args)).compute()
    return results


def parallel_for_each(iterable_of_args, action, dask_client=None):
    if dask_client is None:
        results = [action(*args) for args in iterable_of_args]
    else:
        futures = [dask_client.submit(action, *args) for args in iterable_of_args]
        results = [future.result() for future in futures]
    return results


def parallel_for_each_2(action, *iterables, dask_client=None):
    if dask_client is None:
        results = [action(*args) for args in zip(*iterables)]
    else:
        futures = dask_client.map(action, *iterables)
        results = dask_client.gather(futures)
    return results


def file_size(fname, default=None):
    try:
        return os.stat(fname).st_size
    except:
        return default


def is_between(x, a, b, abs_error=None):
    if not isinstance(x, Iterable):
        lower, upper = (a, b) if a <= b else (b, a)
    else:
        df = pd.DataFrame({0: a, 1: b})
        lower, upper = df.min(axis='columns'), df.max(axis='columns')
        x = pd.Series(x)
    if abs_error:
        return (lower - abs_error <= x) & (x <= upper + abs_error)
    else:
        return (lower <= x) & (x <= upper)


def is_strictly_increasing(alist):
    return all(a < b for a, b in zip(alist, alist[1:]))


def to_tuple(indices):
    if not isinstance(indices, tuple):
        return indices,
    return indices


def linear_search(x, an_iterable):
    """

    :param x:
    :param an_iterable:
    :return: a smallest i for which the i-th element of an_iterable is > x
    """
    for i, v in zip(itertools.count(), an_iterable):
        if v > x:
            return i
    raise ValueError(f'no element of {an_iterable} is > than {x}')


def sandwiching_values(x, an_iterable):
    """

    :param x:
    :param an_iterable: must be a non-decreasing sequence
    :return: a smallest i for which the i-th element of an_iterable is > x
    """
    it = iter(zip(itertools.count(), an_iterable))
    try:
        i, v = next(it)
        if v > x:
            raise ValueError(f'all elements of an_iterable are > x={x}')
        v_new = None
        for i, v_new in it:
            if x <= v_new:
                return (i-1, v), (i, v_new)
            v = v_new
        if v_new is None:
            raise ValueError(f'an_iterable has only one element')
        raise ValueError(f'all elements of an_iterable are < x={x}')
    except StopIteration:
        raise ValueError(f'an_iterable has no element')


def sandwiching_values_by_binary_search(x, a, b, f, aux=None):
    if a > b:
        raise ValueError(f'there are no elements; (lat,lon)={aux}')
    if x < f(a):
        # TODO: uncomment the line below
        #logger.warning(f'smallest element={f(a)} is > x={x}; (lat,lon)={aux}')
        return (a, ), (f(a), )
    if f(b) < x:
        # TODO: uncomment the line below
        #logger.warning(f'greatest element={f(b)} is < x={x}; (lat,lon)={aux}')
        return (b, ), (f(b), )
    if a == b:
        return (a, ), (f(a), )
    else:
        return _rec_sandwiching_values_by_binary_search(x, a, b, f)


def strict_sandwiching_values_by_binary_search(x, a, b, f):
    if a >= b:
        raise ValueError(f'there is at most one element')
    if x < f(a):
        raise ValueError(f'all elements are > x={x}')
    if f(b) < x:
        raise ValueError(f'all elements are < x={x}')
    return _rec_sandwiching_values_by_binary_search(x, a, b, f)


def _rec_sandwiching_values_by_binary_search(x, a, b, f):
    """
    Assumes that a+1 <= b

    :param x:
    :param a:
    :param b:
    :param f:
    :return:
    """
    if a + 1 == b:
        return (a, b), (f(a), f(b))
    c = (a + b) // 2
    if x <= f(c):
        return _rec_sandwiching_values_by_binary_search(x, a, c, f)
    else:
        return _rec_sandwiching_values_by_binary_search(x, c, b, f)


class NearestPoint:
    def __init__(self, lat, lon, index):
        self.lat = lat
        self.lon = lon
        self.index = index

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'[lat: {self.lat}, lon: {self.lon}, index: {self.index}]'


def four_nearest_points_in_rectangular_grid(x0, y0, x1, y1, dx, dy, nx, ny, x_is_major, x, y):
    if not (dx > 0 and x0 <= x <= x1) and not (dx < 0 and x0 >= x >= x1):
        raise ValueError(f'latitude {x} out of range [{x0}, {x1}]')
    # TODO: similar check for longitudes, but be careful
    x_upper = max(x0, x1)
    x_lower = min(x0, x1)
    x = min(x, x_upper - abs(dx) / 2.)
    x = max(x, x_lower + abs(dx) / 2.)
    i = int((x - x0) // dx)
    j = int(((y - y0) % (360. if dy > 0 else -360.)) // dy)
    grid_lat = x0 + i * dx
    grid_lon = (y0 + j * dy) % 360.
    if grid_lon > 180.:
        grid_lon -= 360.
    if x_is_major:
        major_offset = ny * i
        return (NearestPoint(grid_lat, grid_lon, major_offset + j),
                NearestPoint(grid_lat, grid_lon + dy, major_offset + (j + 1) % ny)), \
               (NearestPoint(grid_lat + dx, grid_lon, major_offset + ny + j),
                NearestPoint(grid_lat + dx, grid_lon + dy, major_offset + ny + (j + 1) % ny))
    else:
        major_offset = nx * j
        major_offset_plus_1 = nx * ((j + 1) % ny)
        return (NearestPoint(grid_lat, grid_lon, major_offset + i),
                NearestPoint(grid_lat, grid_lon + dy, major_offset_plus_1 + i)), \
               (NearestPoint(grid_lat + dx, grid_lon, major_offset + i + 1),
                NearestPoint(grid_lat + dx, grid_lon + dy, major_offset_plus_1 + i + 1))


def check_ENfilename(filename, prefix='EN'):
    if not isinstance(filename, str):
        return False
    if len(filename) != 10:
        return False
    if prefix is not None and not filename.startswith(prefix):
        return False
    try:
        get_timestamp_for_ENfilename(filename)
    except ValueError:
        return False
    return True


def get_timestamp_for_ENfilename(filename):
    year = int(filename[-8:-6])
    if year > 70:
        year += 1900
    else:
        year += 2000
    month = int(filename[-6:-4])
    day = int(filename[-4:-2])
    hour = int(filename[-2:])
    return pd.Timestamp(year, month, day, hour)


def get_timestamp_for_flexpart_output_hdf5_filename(yearmonth):
    yearmonthday = yearmonth + '01'
    return pd.Timestamp(yearmonthday)


def get_timestamp(date_in_yyyymmdd, time_in_hhmmss):
    if not isinstance(date_in_yyyymmdd, str) or len(date_in_yyyymmdd) != 8:
        raise ValueError(f'Invalid argument date_in_yyyymmdd={date_in_yyyymmdd}')
    if not isinstance(time_in_hhmmss, str) or len(time_in_hhmmss) > 6 or len(time_in_hhmmss) == 0:
        raise ValueError(f'Invalid argument time_in_hhmmss={time_in_hhmmss}')
    if len(time_in_hhmmss) < 6:
        time_in_hhmmss = '0'*(6 - len(time_in_hhmmss)) + time_in_hhmmss
    return pd.Timestamp(date_in_yyyymmdd + time_in_hhmmss)


def timestamp_now_formatted(format='%Y-%m-%d_%H:%M:%S', tz='UTC'):
    return pd.Timestamp.now(tz=tz).strftime(format)


class FixedSizeCacheDict:
    def __init__(self, size, close_action=None):
        self.size = size
        self._dict = OrderedDict()
        self._close_action = close_action

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        popped_items = []
        if key in self._dict:
            popped_items.append(self._dict.pop(key))
        while len(self._dict) >= self.size:
            popped_items.append(self._dict.popitem(last=False))
        if self._close_action:
            for item in popped_items:
                self._close_action(item)
        self._dict[key] = value


def piecewise_constant_intervals(series, limit=0):
    import numpy.testing
    if limit > 0:
        data = series.fillna(method='ffill', limit=limit)
    else:
        data = series
    data = data.fillna(0)
    (start_index, ) = np.nonzero(np.array(data.diff()))
    value = np.array(series.iloc[start_index])
    end_index = np.concatenate([start_index[1:], [len(data)]]) - 1
    for i in range(limit):
        offset = np.where((~np.isnan(value)) & np.array(series.iloc[end_index].isna()), 1, 0)
        end_index -= offset
        start_index[1:] -= offset[:-1]
    start = series.index[start_index]
    end = series.index[end_index]
    numpy.testing.assert_array_equal(value, np.array(series.loc[end]))
    return pd.DataFrame({'no_levels': value, 'start': start, 'end': end, 'period_length': end - start + np.timedelta64(3, 'h')})


def piecewise_constant_intervals_old(data, fill_value=None, limit=None):
    if fill_value is None:
        if limit is None:
            data = data.dropna()
        else:
            data = data.fillna(method='ffill', limit=limit).dropna()
    else:
        if limit is None:
            data = data.fillna(fill_value)
        else:
            data = data.fillna(method='ffill', limit=limit).fillna(fill_value)
    (start_index, ) = np.nonzero(np.array(data.diff()))
    end_index = np.concatenate([start_index[1:], [len(data)]]) - 1
    start = data.index[start_index]
    end = data.index[end_index]
    value = data.iloc[start_index]
    return pd.DataFrame({'value': np.array(value), 'start': start, 'end': end})


def head(filename, count=1):
    """
    This one is fairly trivial to implement but it is here for completeness.
    """
    with open(filename, 'r') as f:
        lines = [f.readline() for line in range(1, count+1)]
        return filter(len, lines)


def tail(filename, count=1, offset=1024):
    """
    A more efficent way of getting the last few lines of a file.
    Depending on the length of your lines, you will want to modify offset
    to get better performance.
    """
    offset = max(offset, 1)
    f_size = os.stat(filename).st_size
    with open(filename, 'r') as f:
        while True:
            seek_to = max(f_size - offset, 0)
            f.seek(seek_to)
            lines = f.readlines()
            # Standard case
            if len(lines) > count:
                return lines[count * -1:]
            if seek_to == 0 and len(lines) <= count:
                return lines
            offset *= 2


def list_to_file(url, data, mode='w'):
    with open(url, mode=mode) as f:
        f.writelines([f'{item}\n' for item in data])


def list_from_file(url, header=1, ignore_blank_lines=False):
    with open(url, mode='r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    if ignore_blank_lines:
        lines = [l for l in lines if l]
    return lines[header:]


# barometric formula: https://en.wikipedia.org/wiki/Atmospheric_pressure#Altitude_variation
p_0 = 101325 # Pa
g = 9.80665 # m s-2
c_p = 1004.68506 # J/(kg K)
T_0 = 288.16 # K
M = 0.02896968 # kg/mol
R_0 = 8.314462618 # J/(mol K)


def pressure_by_hasl(h):
    return p_0 * (1 - g * h / (c_p * T_0)) ** (c_p * M / R_0)


def hasl_by_pressure(p):
    return (1 - (p / p_0) ** (R_0 / (c_p * M))) * c_p * T_0 / g


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    See: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlon/2.0)**2 * np.cos(lat2) * np.cos(lat1) + np.sin(dlat/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def int_with_magnitude_suffix(num):
    """
    Change integer to string with k, M, G, T or P suffix
    :param num: int
    :return: str; formatted size as llll.dd M, for example
    """
    suffix = ''
    if num >= 1000:
        for suffix in 'kMGTP':
            num /= 1000
            if num < 1000:
                break
    if num >= 100 or num == int(num):
        ret = f'{num:.0f}'
    elif num >= 10:
        ret = f'{num:.1f} '
    else:
        ret = f'{num:.2f} '
    return f'{ret}{suffix}'


class StaticBST:
    def __init__(self, iterable, is_sorted=None, sort_func=None):
        iterable = list(iterable)
        n = len(iterable)
        if n == 0:
            self.empty = True
        else:
            self.empty = False
            if not is_sorted:
                def sort_function(item):
                    k, _ = item
                    return k if sort_func is None else sort_func(k)

                iterable = sorted(iterable, key=sort_function)
            middle = n // 2
            self.key, self.value = iterable[middle]
            self.left = StaticBST.__new__(type(self))
            self.left.__init__(iterable[:middle], is_sorted=True)
            self.right = StaticBST.__new__(type(self))
            self.right.__init__(iterable[(middle + 1):], is_sorted=True)


class StaticIntervalTree(StaticBST):
    def __init__(self, iterable, is_sorted=None):
        def sort_func(key):
            a, b = key
            return a
        super().__init__(iterable, is_sorted=is_sorted, sort_func=sort_func)
        self._max_b()

    def __repr__(self):
        if self.empty:
            return ''
        else:
            a, b = self.key
            return f'({self.left.__repr__()}) [{a},{b}; {self.max_b}];{self.value} ({self.right.__repr__()})'

    def _max_b(self):
        if self.empty:
            return None
        else:
            _, b = self.key
            bs = (b, self.left._max_b(), self.right._max_b())
            self.max_b = max(filter(lambda b: b is not None, bs))
            return self.max_b

    def _overlap(self, a, b):
        if self.empty:
            return False
        else:
            low, high = self.key
            return low <= b and high >= a

    def find(self, a, b):
        res = []
        if not self.empty:
            if not self.left.empty and self.left.max_b >= a:
                res.extend(self.left.find(a, b))
            if self._overlap(a, b):
                res.append((self.key, self.value))
            if not self.right.empty:
                low, _ = self.key
                high = self.right.max_b
                if low <= b and high >= a:
                    res.extend(self.right.find(a, b))
        return res


def make_backup(url, suffix=''):
    current_url = pathlib.Path(str(url) + suffix)
    if current_url.exists():
        if suffix == '':
            backup_suffix = '.bak'
        elif suffix == '.bak':
            backup_suffix = '.bak2'
        else:
            backup_suffix = '.bak' + str(int(suffix[4:]) + 1)
        make_backup(url, suffix=backup_suffix)
        backup_url = str(url) + backup_suffix
        current_url.rename(backup_url)
