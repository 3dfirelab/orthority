#!/usr/bin/env python3
"""Convert a time-dependent navigation NetCDF to a point GeoPackage."""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("netcdf", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", default="trajectory")
    a = p.parse_args()

    with xr.open_dataset(a.netcdf) as ds:
        if "time" not in ds or "LATITUDE" not in ds or "LONGITUDE" not in ds:
            raise ValueError("NetCDF must contain time, LATITUDE, and LONGITUDE")
        data = {}
        for name, var in ds.variables.items():
            if var.dims == ("time",):
                values = var.values
                if name == "time":
                    continue
                data[name.lower()] = values
        data["time_utc"] = pd.to_datetime(ds.time.values, utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        frame = pd.DataFrame(data)

    # GeoPackage geometry cannot be made from missing coordinates.
    valid = frame["latitude"].notna() & frame["longitude"].notna()
    frame = frame.loc[valid].copy()
    out = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame.longitude, frame.latitude),
        crs="EPSG:4326",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(a.output, layer=a.layer, driver="GPKG")
    print(f"Wrote {len(out)} points to {a.output} layer={a.layer}")


if __name__ == "__main__":
    main()
