from dataclasses import replace
import pytest
from src.components.models import Detection
from src.components.vehicle_tracking import VehicleTracker
from src.components.vehicle_counting import TrafficConfig, TrafficMonitor


def det(x=20, y=20, cls=2):
    return Detection((x - 5, y - 5, 10, 10), cls, 0.95)


def test_tracking_persists_id_and_reappears_after_short_gap():
    tracker = VehicleTracker(max_distance=40, max_missing=2)
    first = tracker.update([det()])
    assert list(first) == [1]
    assert tracker.update([]) == {}
    recovered = tracker.update([det(24, 24)])
    assert list(recovered) == [1]
    assert recovered[1].previous == (20, 20)
    assert tracker.update([det(30, 30, 3)])[2].class_id == 3


def test_tracker_expires_and_starts_new_id():
    t = VehicleTracker(max_missing=1)
    t.update([det()])
    t.update([])
    t.update([])
    assert list(t.update([det()])) == [2]


def test_counts_crossings_and_occupancy_independently():
    tracker = VehicleTracker(max_distance=100)
    monitor = TrafficMonitor(TrafficConfig(occupancy_threshold=2, persistence_seconds=2,
                                           recovery_seconds=1, line_y_fraction=.5))
    frame = (100, 100, 3)
    one = monitor.update(0, tracker.update([det(20, 20), det(40, 30, 3)]), frame)
    assert one['occupancy'] == 2 and one['total_line_crossings'] == 0
    two = monitor.update(1, tracker.update([det(20, 70), det(40, 30, 3)]), frame)
    assert two['total_line_crossings'] == 1
    assert two['flow_by_class']['car'] == 1
    three = monitor.update(2, tracker.update([det(20, 20), det(40, 30, 3)]), frame)
    assert three['event'] == 'CONGESTION_STARTED'
    assert three['total_line_crossings'] == 1  # no double-count on opposite crossing
    assert monitor.update(3, tracker.update([]), frame)['event'] is None
    cleared = monitor.update(4, tracker.update([]), frame)
    assert cleared['event'] == 'CONGESTION_CLEARED'
    assert cleared['congested'] is False


def test_persistence_resets_when_occupancy_falls():
    t = VehicleTracker()
    m = TrafficMonitor(TrafficConfig(occupancy_threshold=1, persistence_seconds=3))
    frame = (100,100,3)
    m.update(0, t.update([det()]), frame)
    m.update(2, t.update([]), frame)
    assert m.update(4, t.update([det()]), frame)['event'] is None
    assert m.update(7, t.update([det()]), frame)['event'] == 'CONGESTION_STARTED'


def test_fractional_roi_and_invalid_parameters():
    m = TrafficMonitor(TrafficConfig(occupancy_threshold=1, roi=(.5,.5,1,1)))
    t = VehicleTracker()
    result = m.update(0, t.update([det(10,10),det(90,90)]), (100,100,3))
    assert result['occupancy'] == 1
    for config in ({'roi':(0,.5,1,.4)}, {'occupancy_threshold':0}, {'line_y_fraction':1.0}):
        with pytest.raises(ValueError): TrafficConfig(**config)
    with pytest.raises(ValueError): m.update(-1, {}, (100,100,3))
