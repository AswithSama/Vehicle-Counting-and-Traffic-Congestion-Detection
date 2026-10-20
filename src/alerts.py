"""Preview-first Twilio SMS dispatcher. Never transmits without explicit live mode."""
from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class TwilioSettings:
    account_sid: str
    auth_token: str
    from_number: str
    to_number: str

    @classmethod
    def from_environment(cls):
        load_dotenv(override=False)
        return cls(
            os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
            os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
            os.getenv("TWILIO_FROM_NUMBER", "").strip(),
            os.getenv("TWILIO_TO_NUMBER", "").strip(),
        )

    def validate(self):
        if not (self.account_sid.startswith("AC") and len(self.account_sid) == 34):
            raise ValueError("TWILIO_ACCOUNT_SID is missing or malformed")
        if not self.auth_token:
            raise ValueError("TWILIO_AUTH_TOKEN is missing")
        for name, number in (("TWILIO_FROM_NUMBER", self.from_number), ("TWILIO_TO_NUMBER", self.to_number)):
            if not (number.startswith("+") and number[1:].isdigit() and 8 <= len(number) <= 16):
                raise ValueError(f"{name} must be an E.164 phone number, e.g. +1...")


def configured() -> bool:
    try:
        TwilioSettings.from_environment().validate()
        return True
    except ValueError:
        return False


def build_message(road_name: str, result: dict, persistence: float) -> str:
    return (
        "TRAFFIC CONGESTION ALERT\n"
        f"Location: {road_name[:80]}\n"
        f"Vehicles currently in ROI: {result['occupancy']}\n"
        f"Threshold: {result['threshold']} for {persistence:g}s\n"
        f"Video timestamp: {result['timestamp_seconds']:.1f}s\n"
        "Please review the road camera and investigate."
    )


class AlertDispatcher:
    def __init__(self, live: bool = False, settings: TwilioSettings | None = None,
                 send_function=None, cooldown_seconds: float = 600.0, max_live_messages: int = 3):
        self.live = live
        self.settings = settings or TwilioSettings.from_environment()
        if self.live:
            self.settings.validate()
        self.send_function = send_function
        self.cooldown_seconds = cooldown_seconds
        self.max_live_messages = max_live_messages
        self.last_live_at: float | None = None
        self.live_attempts = 0
        self.alerts: list[dict] = []

    def on_event(self, road_name: str, result: dict, persistence: float) -> dict | None:
        if result["event"] != "CONGESTION_STARTED":
            return None
        timestamp = result["timestamp_seconds"]
        text = build_message(road_name, result, persistence)
        record = {"at_seconds": timestamp, "mode": "LIVE" if self.live else "DRY_RUN", "message": text}
        if not self.live:
            record["status"] = "preview_only"
        elif (self.live_attempts >= self.max_live_messages or
              (self.last_live_at is not None and timestamp - self.last_live_at < self.cooldown_seconds)):
            record["status"] = "suppressed_cooldown_or_limit"
        else:
            self.live_attempts += 1
            self.last_live_at = timestamp
            try:
                if self.send_function is not None:
                    sid = self.send_function(text)
                else:
                    from twilio.rest import Client
                    client = Client(self.settings.account_sid, self.settings.auth_token)
                    sent = client.messages.create(body=text, from_=self.settings.from_number, to=self.settings.to_number)
                    sid = sent.sid
                record["status"] = "accepted_by_api"
                record["sid"] = sid
            except Exception:
                # Do not leak phone numbers, auth headers or raw provider errors into saved artifacts.
                record["status"] = "send_failed"
                record["error"] = "Twilio request failed; verify account, sender, recipient and provider logs"
        self.alerts.append(record)
        return record
