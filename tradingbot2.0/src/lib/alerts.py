"""
Alert System Module for Trading Bot Notifications.

Provides multi-channel alerting for critical trading events including:
- Email notifications via SMTP
- Slack webhook integration
- Generic webhook support (Discord, PagerDuty, etc.)
- Console/logging output

The alert system integrates with the recovery handler's on_alert callback
to provide real-time notifications for risk events, connection issues,
and critical errors.

Reference: specs/live-trading-execution.md (Logging, Error Handling sections)
"""

import asyncio
import aiohttp
import logging
import smtplib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from collections import deque

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Available alert channels."""
    CONSOLE = "console"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DISCORD = "discord"


class AlertPriority(Enum):
    """Alert priority levels with numeric values for comparison."""
    LOW = 1         # Informational, can be batched
    MEDIUM = 2      # Important, send within minutes
    HIGH = 3        # Urgent, send immediately
    CRITICAL = 4    # Emergency, send via all channels

    @property
    def label(self) -> str:
        """Get string label for display."""
        return self.name.lower()


@dataclass
class Alert:
    """Represents an alert to be sent."""
    timestamp: datetime
    priority: AlertPriority
    title: str
    message: str
    category: str = "system"  # risk, connection, order, position, etc.
    details: Optional[Dict[str, Any]] = None
    source: str = "trading_bot"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.label,
            "title": self.title,
            "message": self.message,
            "category": self.category,
            "details": self.details,
            "source": self.source,
        }

    def format_text(self) -> str:
        """Format as plain text."""
        lines = [
            f"[{self.priority.label.upper()}] {self.title}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Category: {self.category}",
            f"Message: {self.message}",
        ]
        if self.details:
            lines.append(f"Details: {json.dumps(self.details, indent=2)}")
        return "\n".join(lines)

    def format_html(self) -> str:
        """Format as HTML for email."""
        priority_colors = {
            AlertPriority.LOW: "#6c757d",
            AlertPriority.MEDIUM: "#ffc107",
            AlertPriority.HIGH: "#fd7e14",
            AlertPriority.CRITICAL: "#dc3545",
        }
        color = priority_colors.get(self.priority, "#6c757d")

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; padding: 10px 15px; border-radius: 5px 5px 0 0;">
                <strong>[{self.priority.label.upper()}]</strong> {self.title}
            </div>
            <div style="border: 1px solid #ddd; border-top: none; padding: 15px; border-radius: 0 0 5px 5px;">
                <p><strong>Time:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Category:</strong> {self.category}</p>
                <p><strong>Message:</strong> {self.message}</p>
        """
        if self.details:
            html += f"<p><strong>Details:</strong></p><pre style='background: #f5f5f5; padding: 10px; border-radius: 3px;'>{json.dumps(self.details, indent=2)}</pre>"
        html += "</div></div>"
        return html


@dataclass
class AlertConfig:
    """Configuration for the alert system."""
    # Enable/disable channels
    console_enabled: bool = True
    email_enabled: bool = False
    slack_enabled: bool = False
    webhook_enabled: bool = False
    discord_enabled: bool = False

    # Email settings
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # Slack settings
    slack_webhook_url: str = ""
    slack_channel: str = "#trading-alerts"

    # Generic webhook settings
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Discord settings
    discord_webhook_url: str = ""

    # Throttling settings
    throttle_window_seconds: float = 60.0  # Time window for throttling
    max_alerts_per_window: int = 10  # Max alerts in window
    cooldown_seconds: float = 5.0  # Min time between same alert

    # Priority-based routing
    min_priority_for_email: AlertPriority = AlertPriority.HIGH
    min_priority_for_slack: AlertPriority = AlertPriority.MEDIUM
    min_priority_for_webhook: AlertPriority = AlertPriority.MEDIUM


class AlertSender(ABC):
    """Abstract base class for alert senders."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send an alert. Returns True if successful."""
        pass

    @property
    @abstractmethod
    def channel(self) -> AlertChannel:
        """Return the channel type."""
        pass


class ConsoleAlertSender(AlertSender):
    """Send alerts to console/logger."""

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.CONSOLE

    async def send(self, alert: Alert) -> bool:
        """Log alert to console."""
        log_level = {
            AlertPriority.LOW: logging.INFO,
            AlertPriority.MEDIUM: logging.WARNING,
            AlertPriority.HIGH: logging.ERROR,
            AlertPriority.CRITICAL: logging.CRITICAL,
        }.get(alert.priority, logging.WARNING)

        logger.log(log_level, f"ALERT [{alert.category}]: {alert.title} - {alert.message}")
        return True


class EmailAlertSender(AlertSender):
    """Send alerts via email using SMTP."""

    def __init__(self, config: AlertConfig):
        self.config = config

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.EMAIL

    async def send(self, alert: Alert) -> bool:
        """Send email alert."""
        if not self.config.email_to:
            logger.warning("No email recipients configured")
            return False

        try:
            # Run SMTP in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._send_sync, alert)
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _send_sync(self, alert: Alert) -> bool:
        """Synchronous email sending."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.priority.label.upper()}] Trading Alert: {alert.title}"
            msg["From"] = self.config.email_from or self.config.smtp_username
            msg["To"] = ", ".join(self.config.email_to)

            # Add plain text and HTML parts
            text_part = MIMEText(alert.format_text(), "plain")
            html_part = MIMEText(alert.format_html(), "html")
            msg.attach(text_part)
            msg.attach(html_part)

            # Send via SMTP
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.sendmail(
                    msg["From"],
                    self.config.email_to,
                    msg.as_string()
                )

            logger.debug(f"Email alert sent to {self.config.email_to}")
            return True

        except Exception as e:
            logger.error(f"SMTP error: {e}")
            return False


class SlackAlertSender(AlertSender):
    """Send alerts to Slack via webhook."""

    def __init__(self, config: AlertConfig):
        self.config = config

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.SLACK

    async def send(self, alert: Alert) -> bool:
        """Send Slack webhook alert."""
        if not self.config.slack_webhook_url:
            logger.warning("No Slack webhook URL configured")
            return False

        try:
            # Build Slack message payload
            emoji = {
                AlertPriority.LOW: ":information_source:",
                AlertPriority.MEDIUM: ":warning:",
                AlertPriority.HIGH: ":rotating_light:",
                AlertPriority.CRITICAL: ":fire:",
            }.get(alert.priority, ":bell:")

            color = {
                AlertPriority.LOW: "#6c757d",
                AlertPriority.MEDIUM: "#ffc107",
                AlertPriority.HIGH: "#fd7e14",
                AlertPriority.CRITICAL: "#dc3545",
            }.get(alert.priority, "#6c757d")

            payload = {
                "channel": self.config.slack_channel,
                "username": "Trading Bot",
                "icon_emoji": emoji,
                "attachments": [{
                    "color": color,
                    "title": f"{emoji} {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Priority", "value": alert.priority.label.upper(), "short": True},
                        {"title": "Category", "value": alert.category, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True},
                    ],
                    "footer": alert.source,
                    "ts": int(alert.timestamp.timestamp()),
                }]
            }

            if alert.details:
                payload["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": f"```{json.dumps(alert.details, indent=2)}```",
                    "short": False,
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.debug("Slack alert sent successfully")
                        return True
                    else:
                        logger.error(f"Slack webhook failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class WebhookAlertSender(AlertSender):
    """Send alerts to generic webhook endpoint."""

    def __init__(self, config: AlertConfig):
        self.config = config

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.WEBHOOK

    async def send(self, alert: Alert) -> bool:
        """Send webhook alert."""
        if not self.config.webhook_url:
            logger.warning("No webhook URL configured")
            return False

        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self.config.webhook_headers)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=alert.to_dict(),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in (200, 201, 202, 204):
                        logger.debug("Webhook alert sent successfully")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class DiscordAlertSender(AlertSender):
    """Send alerts to Discord via webhook."""

    def __init__(self, config: AlertConfig):
        self.config = config

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.DISCORD

    async def send(self, alert: Alert) -> bool:
        """Send Discord webhook alert."""
        if not self.config.discord_webhook_url:
            logger.warning("No Discord webhook URL configured")
            return False

        try:
            # Build Discord embed
            color = {
                AlertPriority.LOW: 0x6c757d,
                AlertPriority.MEDIUM: 0xffc107,
                AlertPriority.HIGH: 0xfd7e14,
                AlertPriority.CRITICAL: 0xdc3545,
            }.get(alert.priority, 0x6c757d)

            payload = {
                "username": "Trading Bot",
                "embeds": [{
                    "title": f"[{alert.priority.label.upper()}] {alert.title}",
                    "description": alert.message,
                    "color": color,
                    "fields": [
                        {"name": "Category", "value": alert.category, "inline": True},
                        {"name": "Priority", "value": alert.priority.label, "inline": True},
                    ],
                    "timestamp": alert.timestamp.isoformat(),
                    "footer": {"text": alert.source},
                }]
            }

            if alert.details:
                details_str = json.dumps(alert.details, indent=2)
                if len(details_str) > 1000:
                    details_str = details_str[:997] + "..."
                payload["embeds"][0]["fields"].append({
                    "name": "Details",
                    "value": f"```json\n{details_str}\n```",
                    "inline": False,
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in (200, 204):
                        logger.debug("Discord alert sent successfully")
                        return True
                    else:
                        logger.error(f"Discord webhook failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False


class AlertManager:
    """
    Central alert management system.

    Handles alert routing, throttling, and delivery across multiple channels.
    Integrates with the recovery handler's on_alert callback.
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._senders: Dict[AlertChannel, AlertSender] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._recent_alerts: Dict[str, datetime] = {}  # For deduplication
        self._alerts_in_window: deque = deque()
        self._lock = asyncio.Lock()

        self._initialize_senders()

    def _initialize_senders(self) -> None:
        """Initialize enabled alert senders."""
        if self.config.console_enabled:
            self._senders[AlertChannel.CONSOLE] = ConsoleAlertSender()

        if self.config.email_enabled:
            self._senders[AlertChannel.EMAIL] = EmailAlertSender(self.config)

        if self.config.slack_enabled:
            self._senders[AlertChannel.SLACK] = SlackAlertSender(self.config)

        if self.config.webhook_enabled:
            self._senders[AlertChannel.WEBHOOK] = WebhookAlertSender(self.config)

        if self.config.discord_enabled:
            self._senders[AlertChannel.DISCORD] = DiscordAlertSender(self.config)

    def _should_throttle(self) -> bool:
        """Check if alerts should be throttled."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.throttle_window_seconds)

        # Remove old alerts from window
        while self._alerts_in_window and self._alerts_in_window[0] < window_start:
            self._alerts_in_window.popleft()

        return len(self._alerts_in_window) >= self.config.max_alerts_per_window

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if this is a duplicate alert within cooldown period."""
        # Create a key based on title and category
        key = f"{alert.category}:{alert.title}"
        now = datetime.now()

        if key in self._recent_alerts:
            last_sent = self._recent_alerts[key]
            if (now - last_sent).total_seconds() < self.config.cooldown_seconds:
                return True

        return False

    def _mark_sent(self, alert: Alert) -> None:
        """Mark an alert as sent for deduplication."""
        key = f"{alert.category}:{alert.title}"
        self._recent_alerts[key] = datetime.now()
        self._alerts_in_window.append(datetime.now())

    def _get_channels_for_priority(self, priority: AlertPriority) -> List[AlertChannel]:
        """Get the appropriate channels for an alert priority."""
        channels = []

        # Console always receives if enabled
        if AlertChannel.CONSOLE in self._senders:
            channels.append(AlertChannel.CONSOLE)

        # Email for high priority and above
        if (AlertChannel.EMAIL in self._senders and
            priority.value >= self.config.min_priority_for_email.value):
            channels.append(AlertChannel.EMAIL)

        # Slack for medium priority and above
        if (AlertChannel.SLACK in self._senders and
            priority.value >= self.config.min_priority_for_slack.value):
            channels.append(AlertChannel.SLACK)

        # Webhook for medium priority and above
        if (AlertChannel.WEBHOOK in self._senders and
            priority.value >= self.config.min_priority_for_webhook.value):
            channels.append(AlertChannel.WEBHOOK)

        # Discord follows webhook settings
        if (AlertChannel.DISCORD in self._senders and
            priority.value >= self.config.min_priority_for_webhook.value):
            channels.append(AlertChannel.DISCORD)

        # Critical alerts go to ALL enabled channels
        if priority == AlertPriority.CRITICAL:
            channels = list(self._senders.keys())

        return channels

    async def send_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        category: str = "system",
        details: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Send an alert through appropriate channels.

        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority level
            category: Alert category (risk, connection, order, etc.)
            details: Additional details dictionary
            force: If True, skip throttling and deduplication

        Returns:
            True if alert was sent to at least one channel
        """
        async with self._lock:
            alert = Alert(
                timestamp=datetime.now(),
                priority=priority,
                title=title,
                message=message,
                category=category,
                details=details,
            )

            # Check throttling (unless forced or critical)
            if not force and priority != AlertPriority.CRITICAL:
                if self._should_throttle():
                    logger.warning(f"Alert throttled: {title}")
                    return False

                if self._is_duplicate(alert):
                    logger.debug(f"Duplicate alert suppressed: {title}")
                    return False

            # Get channels for this priority
            channels = self._get_channels_for_priority(priority)

            if not channels:
                logger.warning(f"No channels configured for alert: {title}")
                return False

            # Send to all applicable channels
            results = await asyncio.gather(
                *[self._senders[ch].send(alert) for ch in channels],
                return_exceptions=True
            )

            # Track success
            success = any(r is True for r in results)

            if success:
                self._mark_sent(alert)
                self._alert_history.append(alert)

            return success

    async def send_risk_alert(
        self,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        priority: AlertPriority = AlertPriority.HIGH,
    ) -> bool:
        """Send a risk-related alert."""
        return await self.send_alert(
            title=title,
            message=message,
            priority=priority,
            category="risk",
            details=details,
        )

    async def send_connection_alert(
        self,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a connection-related alert."""
        return await self.send_alert(
            title=title,
            message=message,
            priority=AlertPriority.MEDIUM,
            category="connection",
            details=details,
        )

    async def send_order_alert(
        self,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send an order-related alert."""
        return await self.send_alert(
            title=title,
            message=message,
            priority=AlertPriority.MEDIUM,
            category="order",
            details=details,
        )

    async def send_critical_alert(
        self,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a critical alert to all channels (force)."""
        return await self.send_alert(
            title=title,
            message=message,
            priority=AlertPriority.CRITICAL,
            category="critical",
            details=details,
            force=True,
        )

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return list(self._alert_history)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.throttle_window_seconds)

        alerts_in_window = sum(
            1 for a in self._alert_history
            if a.timestamp > window_start
        )

        by_priority = {}
        by_category = {}
        for alert in self._alert_history:
            by_priority[alert.priority.label] = by_priority.get(alert.priority.label, 0) + 1
            by_category[alert.category] = by_category.get(alert.category, 0) + 1

        return {
            "total_alerts": len(self._alert_history),
            "alerts_in_window": alerts_in_window,
            "throttle_limit": self.config.max_alerts_per_window,
            "enabled_channels": [ch.value for ch in self._senders.keys()],
            "by_priority": by_priority,
            "by_category": by_category,
        }


def create_error_event_handler(alert_manager: AlertManager) -> Callable:
    """
    Create an error event handler compatible with RecoveryHandler.on_alert callback.

    This bridges the ErrorEvent from recovery.py to the AlertManager.

    Usage:
        alert_manager = AlertManager(config)
        handler = create_error_event_handler(alert_manager)
        recovery_handler = RecoveryHandler(..., on_alert=handler)
    """
    from src.trading.recovery import ErrorEvent, ErrorSeverity, ErrorCategory

    def handler(error: ErrorEvent) -> None:
        """Handle ErrorEvent from RecoveryHandler."""
        # Map ErrorSeverity to AlertPriority
        priority_map = {
            ErrorSeverity.DEBUG: AlertPriority.LOW,
            ErrorSeverity.WARNING: AlertPriority.MEDIUM,
            ErrorSeverity.ERROR: AlertPriority.HIGH,
            ErrorSeverity.CRITICAL: AlertPriority.CRITICAL,
        }
        priority = priority_map.get(error.severity, AlertPriority.MEDIUM)

        # Create alert from error event
        asyncio.create_task(alert_manager.send_alert(
            title=f"{error.category.value.title()} Error",
            message=error.message,
            priority=priority,
            category=error.category.value,
            details={
                "recoverable": error.recoverable,
                "recovery_action": error.recovery_action,
                "exception": str(error.exception) if error.exception else None,
                **(error.details or {}),
            },
        ))

    return handler


# Convenience function to create AlertManager from environment variables
def create_alert_manager_from_env() -> AlertManager:
    """
    Create AlertManager with configuration from environment variables.

    Environment variables:
        ALERT_EMAIL_ENABLED: "true" to enable email
        ALERT_SMTP_HOST: SMTP server host
        ALERT_SMTP_PORT: SMTP server port
        ALERT_SMTP_USERNAME: SMTP username
        ALERT_SMTP_PASSWORD: SMTP password
        ALERT_EMAIL_TO: Comma-separated list of recipients
        ALERT_SLACK_ENABLED: "true" to enable Slack
        ALERT_SLACK_WEBHOOK_URL: Slack webhook URL
        ALERT_SLACK_CHANNEL: Slack channel (default: #trading-alerts)
        ALERT_DISCORD_ENABLED: "true" to enable Discord
        ALERT_DISCORD_WEBHOOK_URL: Discord webhook URL
        ALERT_WEBHOOK_ENABLED: "true" to enable generic webhook
        ALERT_WEBHOOK_URL: Generic webhook URL
    """
    import os

    config = AlertConfig(
        console_enabled=True,

        # Email settings
        email_enabled=os.getenv("ALERT_EMAIL_ENABLED", "").lower() == "true",
        smtp_host=os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.getenv("ALERT_SMTP_PORT", "587")),
        smtp_username=os.getenv("ALERT_SMTP_USERNAME", ""),
        smtp_password=os.getenv("ALERT_SMTP_PASSWORD", ""),
        email_to=[e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(",") if e.strip()],

        # Slack settings
        slack_enabled=os.getenv("ALERT_SLACK_ENABLED", "").lower() == "true",
        slack_webhook_url=os.getenv("ALERT_SLACK_WEBHOOK_URL", ""),
        slack_channel=os.getenv("ALERT_SLACK_CHANNEL", "#trading-alerts"),

        # Discord settings
        discord_enabled=os.getenv("ALERT_DISCORD_ENABLED", "").lower() == "true",
        discord_webhook_url=os.getenv("ALERT_DISCORD_WEBHOOK_URL", ""),

        # Generic webhook settings
        webhook_enabled=os.getenv("ALERT_WEBHOOK_ENABLED", "").lower() == "true",
        webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
    )

    return AlertManager(config)
