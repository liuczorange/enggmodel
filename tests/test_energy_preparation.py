import numpy as np
import pandas as pd

from prepare_energy_data import customer_frame, customer_split


def test_preparation_preserves_teammate_sample_and_disjoint_seeded_groups():
    ids = [str(value) for value in range(1000, 3000)]
    splits = customer_split(ids, 1500)
    assert splits == customer_split(ids, 1500)
    assert [len(splits[name]) for name in ("learn", "finetune", "test")] == [1050, 225, 225]
    combined = sum(splits.values(), [])
    assert len(set(combined)) == 1500
    expected_sample = np.random.RandomState(123).choice(ids, 1500, replace=False)
    assert set(combined) == set(expected_sample)


def test_preparation_uses_real_timestamp_lags_and_drops_gap_affected_rows():
    complete = pd.date_range("2013-01-01", periods=500, freq="30min")
    times = complete.delete(400)
    readings = np.delete(np.arange(500, dtype=float), 400)
    market = pd.DataFrame({"DATETIME_30MIN": complete, "PRICE_AUD_MWH": 50., "DEMAND_MW": 8000.})
    frame = customer_frame(times, readings, "12345", market)
    assert complete[402] not in set(frame["DATETIME_30MIN"])
    assert complete[403] in set(frame["DATETIME_30MIN"])
    assert (frame["KWH_READING"] - frame["KWH_LAG_1HR"]).eq(2).all()
    assert (frame["KWH_READING"] - frame["KWH_LAG_3DAYS"]).eq(144).all()
    assert (frame["KWH_READING"] - frame["KWH_LAG_1WEEK"]).eq(336).all()
