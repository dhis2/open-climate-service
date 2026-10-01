import json
from typing import Any

import geopandas as gpd
import xarray as xr
from exactextract import exact_extract, writer
from exactextract.feature import JSONFeature

from open_climate_service.process import process


@process(
    summary="Weighted aggregation of spatial raster data to vector geometries using exact pixel-polygon overlap",
    parameters={
        "data": {"description": "A raster data cube."},
        "geometries": {"description": "GeoJSON FeatureCollection, Feature, or geometry (polygons or points)."},
        "reducer": {"description": "A reducer to apply on the pixel values."},
        "context": {"description": "Optional context passed to the reducer."},
    },
)
def aggregate_spatial_weighted(
    data: Any,
    geometries: Any,
    reducer: str,
    context: Any = None,
) -> xr.Dataset:
    """Weighted aggregation of raster values to each input geometry."""

    # Ensure correct input data format
    if isinstance(data, xr.DataArray):
        # The variable keeps the DataArray's attributes.
        data = data.to_dataset(name=data.name or "data")
    assert isinstance(data, xr.Dataset)

    # Convert geometries to geopandas
    if isinstance(geometries, dict):
        geometries = json.dumps(geometries)
    if isinstance(geometries, str):
        gdf = gpd.read_file(geometries)  # parses geojson string into GeoDataFrame
    else:
        raise TypeError(f"Unsupported type for geometries: {geometries}")

    # Compute aggregations using exact extract
    agg = exact_extract(
        data,
        gdf,
        [reducer],
        #include_geom=False,
        #include_cols=['id'],
        output=XArrayWriter(),
        strategy='raster-sequential',
    )

    # HACK: Add geopandas wkt to geometry dim (not supported by xarray writer yet)
    shapes = gdf.geometry.iloc[agg.feature.to_numpy() - 1].to_numpy()
    geoms = [shp.wkt for shp in shapes]
    agg = agg.assign_coords(
        geometry=("feature", geoms)
    )

    # Post-process: For raster cubes with a time (t) dimension,
    # map "band" index from aggregation results to the "t" values from raster cube
    if "t" in data and "band" in agg:
        time_values = data.t.values
        agg = (
            agg.assign_coords(
                time=("band", time_values)
            )
            .swap_dims({"band": "time"})
            .drop_vars("band")
        )

    return agg


class XArrayWriter(writer.Writer):
    """
    Writer that returns an :py:class:`xarray.Dataset`, with one or more data variables
    for each of the computed statistics. 
    Returned dimensions depend on the structure of the input raster: ``(feature)`` for single variable and single band 
    raster, ``(feature, band)`` for single variable and multi band raster, 
    ``(feature, var)`` for multi variable and single band raster, and ``(feature, var, band)``
    for multi variable and multi band raster. 
    If the input raster has multiple dimensions (e.g. time x level), band numbering follows 
    the same logic as other Writers, bands are enumerated in C-order (last dimension varies 
    fastest), matching the order returned by ``rasterio.count``.
    """

    def __init__(self):
        super().__init__()

        self.ops = []
        self.extra_cols = {}
        self.records = []
        self.feature_count = 0

    def add_operation(self, op):
        self.ops.append(op)

    def add_column(self, col_name):
        self.extra_cols[col_name] = []

    def write(self, feature):
        f = JSONFeature()
        feature.copy_to(f)
        props = f.feature["properties"]

        # get feature index
        self.feature_count += 1
        feature_index = int(self.feature_count)

        # add any extra column values
        for col in self.extra_cols:
            if col == 'id' and 'id' in f.feature:
                value = f.feature["id"]
            else:
                value = props[col]
            self.extra_cols[col].append(value)

        # all we have is a list of operations corresponding to properties per feature
        # each operation/property name encodes dimensions: statistic, band, and variable
        # and its data value and dimension values should be added as a dict to .records
        for op in self.ops:
            # extract misc dims from operation property name
            # possible operation property templates:
            # - statistics: mean, sum, etc
            # - bands and statistics: band_1_mean, band_1_sum, etc
            # - variables and statistics: var1_mean, var1_sum, etc
            # - variables and bands and statistics: var1_band_1_mean, var1_band_1_sum, etc
            prop = op.name
            value = props.get(prop, None)  # sometimes statistic is missing from props
            row = {"feature": feature_index, "value": value}
            
            # if prop starts with band_, then that should be used to split into band and stat
            if prop.startswith('band_'):
                parts = prop.split('_')
                band = parts[1]
                stat = '_'.join(parts[2:])
                row['band'] = int(band)
                row['stat'] = stat

            # if prop contains _band_ then that should be used to split into varname, band and stat
            elif '_band_' in prop:
                varname, band_plus_stat = prop.split('_band_')
                parts = band_plus_stat.split('_')
                band = parts[0]
                stat = '_'.join(parts[1:])
                row['var'] = varname
                row['band'] = int(band)
                row['stat'] = stat

            # prop should be stat and optionally varname
            else:
                # search for first string instance of stat
                stat_pos = prop.find(op.stat)
                if stat_pos == -1:
                    raise ValueError(f'Unable to parse {op.stat} statistic from field name {prop}')
                
                # extract full stat name starting at first string instance
                # eg stat may include additional parts based on kwargs, eg quantile_25
                stat = prop[stat_pos:]
                row['stat'] = stat

                # if stat starts in middle of string, then first part is varname
                if stat_pos > 0:
                    varname = prop[:stat_pos].strip('_')
                    row['var'] = varname

            self.records.append(row)

    def features(self):
        # make pandas df from dict records
        import pandas as pd
        df = pd.DataFrame(self.records)
        
        # get which xarray dims to keep
        dim_cols = [c for c in df.columns if c not in ("value", "stat")]

        # pivot stat into columns
        df = df.pivot_table(
            index=dim_cols,
            columns="stat",
            values="value",
            dropna=False,
        )

        # drop var from index if only one unique value
        if "var" in df.index.names and df.index.get_level_values("var").nunique() == 1:
            df = df.droplevel("var")

        # convert to xarray
        ds = df.to_xarray()

        # assign extra columns as coords
        # all extra columns are based on the feature dimension
        if self.extra_cols:
            col_dim = 'feature'
            coords = {
                col: (col_dim, col_values)
                for col, col_values
                in self.extra_cols.items()
            }
            ds = ds.assign_coords(**coords)

        return ds
