import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.alerts import AlertDispatcher
from src.components.models import Detection
from src.components.vehicle_counting import TrafficConfig
from src.pipeline.predict_pipeline import ProcessingOptions, process_video


class FakeDetector:
    """Deterministic detections to test the COMPLETE video/artifact path without 237MB weights."""
    def __init__(self): self.n = 0
    def detect(self, frame):
        i = self.n
        self.n += 1
        if i >= 14: return []
        first_y = 15 if i < 4 else 75
        return [Detection((40, first_y, 22, 22), 2, .91),
                Detection((95, 38, 22, 22), 3, .88)]


def write_video(path, fps=10, nframes=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (240, 120))
    assert writer.isOpened()
    for i in range(nframes): writer.write(np.full((120,240,3), i*3, dtype=np.uint8))
    writer.release()


def test_end_to_end_recorded_video_and_alert_artifacts(tmp_path):
    source=tmp_path/'in.mp4'; write_video(source)
    opt=ProcessingOptions(traffic=TrafficConfig(occupancy_threshold=2, persistence_seconds=.5,
                                                recovery_seconds=.4, line_y_fraction=.5))
    progress=[]
    summary=process_video(source,tmp_path/'out',opt,detector=FakeDetector(),
                          dispatcher=AlertDispatcher(),progress=lambda n,total:progress.append((n,total)))
    assert summary['processed_frames']==25
    assert summary['cumulative_crossings']==1
    assert summary['flow_by_class']['car']==1
    assert summary['congestion_events']==1
    assert summary['sms_mode']=='DRY_RUN'
    assert summary['alerts'][0]['status']=='preview_only'
    assert len(summary['events'])==2  # started and cleared
    assert summary['events'][0]['timestamp_seconds'] >= .5
    assert summary['events'][0]['screenshot']=='event_001.jpg'
    out=tmp_path/'out'
    for file in ('annotated.mp4','preview.jpg','summary.json','events.json','alerts.json','counts.csv','event_001.jpg'):
        assert (out/file).stat().st_size > 0
    assert json.loads((out/'summary.json').read_text())['congestion_events']==1
    assert progress[-1]==(25,25)
    v=cv2.VideoCapture(str(out/'annotated.mp4'))
    assert v.isOpened() and int(v.get(cv2.CAP_PROP_FRAME_COUNT))==25
    v.release()


def test_missing_model_informative(tmp_path):
    from src.components.video_network_setup import load_network
    with pytest.raises(FileNotFoundError,match='python -m src.download_models'):
        load_network(tmp_path)
