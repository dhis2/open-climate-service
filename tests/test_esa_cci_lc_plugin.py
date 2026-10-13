import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.rasters.esa_cci_lc import CLASS_GROUPS, ESACCILandCoverFractionPlugin, class_fractions


def _classes(values: list[list[int]]) -> xr.DataArray:
    data = np.array(values, dtype="uint8")
    return xr.DataArray(data, dims=("y", "x"), coords={"y": np.arange(data.shape[0]), "x": np.arange(data.shape[1])})


def test_every_lccs_class_belongs_to_exactly_one_group() -> None:
    codes = [code for group in CLASS_GROUPS.values() for code in group]
    assert len(codes) == len(set(codes))
    assert 0 not in codes


def test_fractions_are_shares_of_each_block_and_sum_to_one() -> None:
    # One 2x2 block: two cropland pixels, one tree, one bare.
    fractions = class_fractions(_classes([[10, 30], [60, 200]]), block=2)
    shares = dict(zip(fractions["land_cover_class"].values.tolist(), fractions.values[:, 0, 0].tolist()))
    assert shares["cropland"] == pytest.approx(0.5)
    assert shares["tree_cover"] == pytest.approx(0.25)
    assert shares["bare"] == pytest.approx(0.25)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_no_data_pixels_are_left_out_and_an_all_no_data_block_is_nan() -> None:
    fractions = class_fractions(_classes([[10, 0, 0, 0], [0, 0, 0, 0]]), block=2)
    assert float(fractions.sel(land_cover_class="cropland").values[0, 0]) == pytest.approx(1.0)
    assert np.isnan(fractions.values[:, 0, 1]).all()


def test_rejects_a_non_positive_block() -> None:
    with pytest.raises(ValueError, match="block"):
        ESACCILandCoverFractionPlugin(block=0)
