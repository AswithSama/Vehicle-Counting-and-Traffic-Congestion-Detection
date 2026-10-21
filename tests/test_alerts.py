import pytest
from src.alerts import AlertDispatcher, TwilioSettings


def sample(t=12.0):
    return {"event":"CONGESTION_STARTED", "timestamp_seconds":t, "occupancy":27, "threshold":25}


def fake_settings():
    return TwilioSettings('AC' + '1'*32, 'token', '+15551234567', '+15557654321')


def test_dry_run_never_calls_network():
    called=[]
    alerts=AlertDispatcher(live=False, send_function=lambda txt:called.append(txt))
    record=alerts.on_event('Road A', sample(), 180)
    assert record['status']=='preview_only' and not called
    assert '27' in record['message']


def test_live_sms_single_event_and_limits():
    sent=[]
    def fake_send(text):
        sent.append(text)
        return 'SM-test'
    alerts=AlertDispatcher(live=True, settings=fake_settings(), send_function=fake_send,
                            cooldown_seconds=30, max_live_messages=2)
    assert alerts.on_event('A', sample(0), 1)['sid']=='SM-test'
    assert alerts.on_event('A', sample(12), 1)['status']=='suppressed_cooldown_or_limit'
    assert alerts.on_event('A', sample(31), 1)['status']=='accepted_by_api'
    assert alerts.on_event('A', sample(100), 1)['status']=='suppressed_cooldown_or_limit'
    assert len(sent)==2


def test_failed_send_is_recorded_without_secret_leaks():
    def failure(text): raise RuntimeError('secret auth header')
    dispatcher=AlertDispatcher(live=True, settings=fake_settings(), send_function=failure)
    record=dispatcher.on_event('A',sample(),180)
    assert record['status']=='send_failed'
    assert 'secret auth' not in str(record)


def test_live_requires_valid_credentials():
    with pytest.raises(ValueError): AlertDispatcher(live=True, settings=TwilioSettings('','','',''))
