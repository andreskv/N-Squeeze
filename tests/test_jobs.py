"""Pausable, resumable archive jobs: a stop loses at most the chunk in progress."""

import datetime as dt

import pandas as pd
import pytest

from nsqueeze import jobs


class _FakeSource:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def archive(self, store, a, b):
        if a == self.fail_on:
            raise RuntimeError("429 Too Many Requests")
        self.calls.append(a)
        return 10


@pytest.fixture(autouse=True)
def _tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))


def test_plan_covers_range_once():
    c = jobs.plan_chunks(dt.date(2020, 1, 1), dt.date(2020, 3, 15), 31)
    days = sum((pd.Timestamp(b) - pd.Timestamp(a)).days + 1 for a, b in c)
    assert c[0][0] == "2020-01-01" and c[-1][1] == "2020-03-15" and days == 75


def test_fail_then_resume_skips_done_chunks():
    j = jobs.Job.new("t", "entsog", dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    src = _FakeSource(fail_on=dt.date(2020, 2, 1))
    assert j.run(store=object(), source=src) == "failed"
    j = jobs.Job.load("t")
    assert j.done[0] and not j.done[1] and "429" in j.last_error
    src.fail_on = None
    assert jobs.Job.load("t").run(store=object(), source=src) == "done"
    assert src.calls.count(dt.date(2020, 1, 1)) == 1  # the finished chunk did not run twice


def test_pause_flag_stops_before_next_chunk():
    j = jobs.Job.new("p", "entsog", dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    j.pause_flag.touch()
    assert j.run(store=object(), source=_FakeSource()) == "paused"
    assert not any(jobs.Job.load("p").done)
