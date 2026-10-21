from pathlib import Path
from fastapi.testclient import TestClient
from src.web.app import create_app


def test_dashboard_and_video_upload_integration(tmp_path):
    def fake_processor(input_path, output_dir, options, *, progress):
        assert input_path.read_bytes()==b'fake video test bytes'
        assert options.traffic.occupancy_threshold==3
        assert options.live_sms is False
        progress(10,10)
        (output_dir/'counts.csv').write_text('timestamp_seconds,occupancy\n0,3\n')
        (output_dir/'preview.jpg').write_bytes(b'fake jpeg')
        return {'cumulative_crossings':1, 'congestion_events':0, 'video':'annotated.mp4'}
    client=TestClient(create_app(output_root=tmp_path,processor=fake_processor))
    assert client.get('/').status_code==200
    response=client.post('/api/jobs',data={'threshold':'3','duration':'0.5'},
                         files={'video':('road.mp4',b'fake video test bytes','video/mp4')})
    assert response.status_code==202, response.text
    jobid=response.json()['job_id']
    job=client.get('/api/jobs/'+jobid).json()
    assert job['status']=='completed'
    assert job['summary']['cumulative_crossings']==1
    assert client.get(f'/api/jobs/{jobid}/files/counts.csv').status_code==200
    preview = client.get(f'/api/jobs/{jobid}/preview')
    assert preview.status_code == 200
    assert preview.content == b'fake jpeg'
    assert 'no-store' in preview.headers['cache-control']
    assert client.get('/api/jobs/missing/preview').status_code == 404
    assert client.get(f'/api/jobs/{jobid}/files/../../.env').status_code in (400,404)


def test_invalid_parameters_and_upload_rejected(tmp_path, monkeypatch):
    # Simulate missing Twilio credentials independently of local .env.
    monkeypatch.setattr("src.web.app.configured", lambda: False)
    client=TestClient(create_app(output_root=tmp_path))
    assert client.post('/api/jobs',data={'threshold':'0'},files={'video':('r.mp4',b'xx')}).status_code==422
    assert client.post('/api/jobs',files={'video':('bad.txt',b'xx')}).status_code==422
    assert client.post('/api/jobs',data={'live_sms':'true'},files={'video':('r.mp4',b'xx')}).status_code==422
    assert client.get('/api/jobs/bad').status_code==404
