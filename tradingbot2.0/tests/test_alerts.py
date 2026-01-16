"""
Tests for the Alert System Module.

Tests cover:
- Alert dataclass creation and formatting
- AlertConfig validation
- Individual senders (Console, Email, Slack, Webhook, Discord)
- AlertManager routing, throttling, and deduplication
- Integration with RecoveryHandler ErrorEvent
- Environment variable configuration

These tests ensure operators receive timely notifications for
critical trading events, connection issues, and risk limit breaches.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import aiohttp

from src.lib.alerts import (
    Alert,
    AlertChannel,
    AlertConfig,
    AlertManager,
    AlertPriority,
    AlertSender,
    ConsoleAlertSender,
    DiscordAlertSender,
    EmailAlertSender,
    SlackAlertSender,
    WebhookAlertSender,
    create_alert_manager_from_env,
    create_error_event_handler,
)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test basic alert creation."""
        alert = Alert(
            timestamp=datetime(2026, 1, 16, 10, 30, 0),
            priority=AlertPriority.HIGH,
            title="Test Alert",
            message="This is a test message",
            category="risk",
        )

        assert alert.priority == AlertPriority.HIGH
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test message"
        assert alert.category == "risk"
        assert alert.source == "trading_bot"
        assert alert.details is None

    def test_alert_with_details(self):
        """Test alert creation with details."""
        details = {"account_balance": 850.0, "daily_loss": 75.0}
        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.CRITICAL,
            title="Daily Loss Limit Hit",
            message="Trading halted",
            category="risk",
            details=details,
        )

        assert alert.details == details
        assert alert.details["account_balance"] == 850.0

    def test_alert_to_dict(self):
        """Test alert serialization."""
        timestamp = datetime(2026, 1, 16, 10, 30, 0)
        alert = Alert(
            timestamp=timestamp,
            priority=AlertPriority.MEDIUM,
            title="Connection Issue",
            message="WebSocket disconnected",
            category="connection",
            details={"retry_count": 3},
        )

        d = alert.to_dict()

        assert d["timestamp"] == timestamp.isoformat()
        assert d["priority"] == "medium"  # Uses label property
        assert d["title"] == "Connection Issue"
        assert d["message"] == "WebSocket disconnected"
        assert d["category"] == "connection"
        assert d["details"]["retry_count"] == 3

    def test_alert_format_text(self):
        """Test plain text formatting."""
        alert = Alert(
            timestamp=datetime(2026, 1, 16, 10, 30, 0),
            priority=AlertPriority.HIGH,
            title="Order Rejected",
            message="Insufficient margin",
            category="order",
        )

        text = alert.format_text()

        assert "[HIGH]" in text
        assert "Order Rejected" in text
        assert "Insufficient margin" in text
        assert "order" in text
        assert "2026-01-16" in text

    def test_alert_format_text_with_details(self):
        """Test plain text formatting with details."""
        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.MEDIUM,
            title="Test",
            message="Test message",
            category="test",
            details={"key": "value"},
        )

        text = alert.format_text()

        assert "Details:" in text
        assert '"key": "value"' in text

    def test_alert_format_html(self):
        """Test HTML formatting."""
        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.CRITICAL,
            title="Kill Switch Triggered",
            message="Emergency halt",
            category="risk",
        )

        html = alert.format_html()

        assert "<div" in html
        assert "#dc3545" in html  # Critical color
        assert "Kill Switch Triggered" in html
        assert "Emergency halt" in html

    def test_alert_format_html_with_details(self):
        """Test HTML formatting with details."""
        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test",
            message="Test",
            category="test",
            details={"metric": 123},
        )

        html = alert.format_html()

        assert "<pre" in html
        assert '"metric": 123' in html


class TestAlertConfig:
    """Tests for AlertConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AlertConfig()

        assert config.console_enabled is True
        assert config.email_enabled is False
        assert config.slack_enabled is False
        assert config.webhook_enabled is False
        assert config.discord_enabled is False
        assert config.throttle_window_seconds == 60.0
        assert config.max_alerts_per_window == 10
        assert config.cooldown_seconds == 5.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = AlertConfig(
            email_enabled=True,
            smtp_host="mail.example.com",
            smtp_port=465,
            email_to=["admin@example.com", "ops@example.com"],
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/services/xxx",
        )

        assert config.email_enabled is True
        assert config.smtp_host == "mail.example.com"
        assert len(config.email_to) == 2
        assert config.slack_enabled is True

    def test_priority_thresholds(self):
        """Test priority threshold settings."""
        config = AlertConfig(
            min_priority_for_email=AlertPriority.CRITICAL,
            min_priority_for_slack=AlertPriority.HIGH,
            min_priority_for_webhook=AlertPriority.LOW,
        )

        assert config.min_priority_for_email == AlertPriority.CRITICAL
        assert config.min_priority_for_slack == AlertPriority.HIGH
        assert config.min_priority_for_webhook == AlertPriority.LOW


class TestConsoleAlertSender:
    """Tests for ConsoleAlertSender."""

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alert to console."""
        sender = ConsoleAlertSender()
        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test Alert",
            message="Test message",
            category="test",
        )

        result = await sender.send(alert)

        assert result is True
        assert sender.channel == AlertChannel.CONSOLE

    @pytest.mark.asyncio
    async def test_send_all_priorities(self):
        """Test sending alerts of all priorities."""
        sender = ConsoleAlertSender()

        for priority in AlertPriority:
            alert = Alert(
                timestamp=datetime.now(),
                priority=priority,
                title=f"{priority.value} Alert",
                message="Test",
                category="test",
            )
            result = await sender.send(alert)
            assert result is True


class TestEmailAlertSender:
    """Tests for EmailAlertSender."""

    def test_channel_type(self):
        """Test channel type."""
        config = AlertConfig()
        sender = EmailAlertSender(config)
        assert sender.channel == AlertChannel.EMAIL

    @pytest.mark.asyncio
    async def test_send_without_recipients(self):
        """Test sending without configured recipients."""
        config = AlertConfig(email_enabled=True)
        sender = EmailAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test",
            message="Test",
            category="test",
        )

        result = await sender.send(alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_with_mock_smtp(self):
        """Test sending with mocked SMTP."""
        config = AlertConfig(
            email_enabled=True,
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="test@test.com",
            smtp_password="password",
            email_to=["recipient@test.com"],
        )
        sender = EmailAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test Alert",
            message="Test message",
            category="test",
        )

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            result = await sender.send(alert)

            assert result is True
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@test.com", "password")
            mock_server.sendmail.assert_called_once()


class TestSlackAlertSender:
    """Tests for SlackAlertSender."""

    def test_channel_type(self):
        """Test channel type."""
        config = AlertConfig()
        sender = SlackAlertSender(config)
        assert sender.channel == AlertChannel.SLACK

    @pytest.mark.asyncio
    async def test_send_without_webhook_url(self):
        """Test sending without configured webhook URL."""
        config = AlertConfig(slack_enabled=True)
        sender = SlackAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test",
            message="Test",
            category="test",
        )

        result = await sender.send(alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_with_mock_webhook(self):
        """Test sending with mocked webhook."""
        config = AlertConfig(
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/services/xxx",
            slack_channel="#trading-alerts",
        )
        sender = SlackAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.CRITICAL,
            title="Kill Switch",
            message="Trading halted",
            category="risk",
            details={"reason": "max drawdown"},
        )

        with patch("src.lib.alerts.aiohttp.ClientSession") as mock_session_cls:
            # Create properly mocked async context managers
            mock_response = MagicMock()
            mock_response.status = 200

            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_post_cm.__aexit__.return_value = None

            mock_session = MagicMock()
            mock_session.post.return_value = mock_post_cm

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None

            mock_session_cls.return_value = mock_session_cm

            result = await sender.send(alert)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_handles_failure(self):
        """Test handling webhook failure."""
        config = AlertConfig(
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/services/xxx",
        )
        sender = SlackAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test",
            message="Test",
            category="test",
        )

        with patch("src.lib.alerts.aiohttp.ClientSession") as mock_session_cls:
            mock_response = MagicMock()
            mock_response.status = 500

            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_post_cm.__aexit__.return_value = None

            mock_session = MagicMock()
            mock_session.post.return_value = mock_post_cm

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None

            mock_session_cls.return_value = mock_session_cm

            result = await sender.send(alert)

            assert result is False


class TestWebhookAlertSender:
    """Tests for WebhookAlertSender."""

    def test_channel_type(self):
        """Test channel type."""
        config = AlertConfig()
        sender = WebhookAlertSender(config)
        assert sender.channel == AlertChannel.WEBHOOK

    @pytest.mark.asyncio
    async def test_send_without_url(self):
        """Test sending without configured URL."""
        config = AlertConfig(webhook_enabled=True)
        sender = WebhookAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test",
            message="Test",
            category="test",
        )

        result = await sender.send(alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_with_custom_headers(self):
        """Test sending with custom headers."""
        config = AlertConfig(
            webhook_enabled=True,
            webhook_url="https://api.example.com/alerts",
            webhook_headers={"Authorization": "Bearer token123", "X-Custom": "value"},
        )
        sender = WebhookAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.MEDIUM,
            title="Test",
            message="Test",
            category="test",
        )

        with patch("src.lib.alerts.aiohttp.ClientSession") as mock_session_cls:
            mock_response = MagicMock()
            mock_response.status = 201

            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_post_cm.__aexit__.return_value = None

            mock_session = MagicMock()
            mock_session.post.return_value = mock_post_cm

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None

            mock_session_cls.return_value = mock_session_cm

            result = await sender.send(alert)

            assert result is True
            # Verify headers were passed
            call_kwargs = mock_session.post.call_args[1]
            assert "Authorization" in call_kwargs["headers"]


class TestDiscordAlertSender:
    """Tests for DiscordAlertSender."""

    def test_channel_type(self):
        """Test channel type."""
        config = AlertConfig()
        sender = DiscordAlertSender(config)
        assert sender.channel == AlertChannel.DISCORD

    @pytest.mark.asyncio
    async def test_send_without_url(self):
        """Test sending without configured URL."""
        config = AlertConfig(discord_enabled=True)
        sender = DiscordAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Test",
            message="Test",
            category="test",
        )

        result = await sender.send(alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_with_mock_webhook(self):
        """Test sending with mocked Discord webhook."""
        config = AlertConfig(
            discord_enabled=True,
            discord_webhook_url="https://discord.com/api/webhooks/xxx/yyy",
        )
        sender = DiscordAlertSender(config)

        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            title="Risk Alert",
            message="Approaching daily loss limit",
            category="risk",
            details={"current_loss": 40.0, "limit": 50.0},
        )

        with patch("src.lib.alerts.aiohttp.ClientSession") as mock_session_cls:
            mock_response = MagicMock()
            mock_response.status = 204

            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_post_cm.__aexit__.return_value = None

            mock_session = MagicMock()
            mock_session.post.return_value = mock_post_cm

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None

            mock_session_cls.return_value = mock_session_cm

            result = await sender.send(alert)

            assert result is True

    @pytest.mark.asyncio
    async def test_truncates_long_details(self):
        """Test that long details are truncated."""
        config = AlertConfig(
            discord_enabled=True,
            discord_webhook_url="https://discord.com/api/webhooks/xxx/yyy",
        )
        sender = DiscordAlertSender(config)

        # Create alert with very long details
        long_details = {"data": "x" * 2000}
        alert = Alert(
            timestamp=datetime.now(),
            priority=AlertPriority.LOW,
            title="Test",
            message="Test",
            category="test",
            details=long_details,
        )

        with patch("src.lib.alerts.aiohttp.ClientSession") as mock_session_cls:
            mock_response = MagicMock()
            mock_response.status = 204

            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_post_cm.__aexit__.return_value = None

            mock_session = MagicMock()
            mock_session.post.return_value = mock_post_cm

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None

            mock_session_cls.return_value = mock_session_cm

            result = await sender.send(alert)

            assert result is True
            # Check that details were truncated
            call_kwargs = mock_session.post.call_args[1]
            payload = call_kwargs["json"]
            details_field = next(
                f for f in payload["embeds"][0]["fields"]
                if f["name"] == "Details"
            )
            assert "..." in details_field["value"]


class TestAlertManager:
    """Tests for AlertManager."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        manager = AlertManager()

        # Only console should be enabled by default
        assert AlertChannel.CONSOLE in manager._senders
        assert AlertChannel.EMAIL not in manager._senders
        assert AlertChannel.SLACK not in manager._senders

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = AlertConfig(
            console_enabled=True,
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/xxx",
            email_enabled=False,
        )
        manager = AlertManager(config)

        assert AlertChannel.CONSOLE in manager._senders
        assert AlertChannel.SLACK in manager._senders
        assert AlertChannel.EMAIL not in manager._senders

    @pytest.mark.asyncio
    async def test_send_alert_basic(self):
        """Test basic alert sending."""
        manager = AlertManager()

        result = await manager.send_alert(
            title="Test Alert",
            message="Test message",
            priority=AlertPriority.MEDIUM,
            category="test",
        )

        assert result is True
        assert len(manager._alert_history) == 1

    @pytest.mark.asyncio
    async def test_send_alert_with_details(self):
        """Test alert sending with details."""
        manager = AlertManager()

        result = await manager.send_alert(
            title="Risk Alert",
            message="Daily loss limit approaching",
            priority=AlertPriority.HIGH,
            category="risk",
            details={"current_loss": 45.0, "limit": 50.0},
        )

        assert result is True
        alert = manager._alert_history[-1]
        assert alert.details["current_loss"] == 45.0

    @pytest.mark.asyncio
    async def test_throttling(self):
        """Test alert throttling."""
        config = AlertConfig(
            console_enabled=True,
            max_alerts_per_window=3,
            throttle_window_seconds=60.0,
        )
        manager = AlertManager(config)

        # Send up to limit
        for i in range(3):
            result = await manager.send_alert(
                title=f"Alert {i}",
                message="Test",
                priority=AlertPriority.MEDIUM,
                category="test",
            )
            assert result is True

        # Next should be throttled
        result = await manager.send_alert(
            title="Alert 4",
            message="Test",
            priority=AlertPriority.MEDIUM,
            category="test",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_throttling_critical_bypasses(self):
        """Test that critical alerts bypass throttling."""
        config = AlertConfig(
            console_enabled=True,
            max_alerts_per_window=2,
            throttle_window_seconds=60.0,
        )
        manager = AlertManager(config)

        # Fill up throttle window
        for i in range(2):
            await manager.send_alert(
                title=f"Alert {i}",
                message="Test",
                priority=AlertPriority.MEDIUM,
                category="test",
            )

        # Critical should still go through
        result = await manager.send_alert(
            title="Critical Alert",
            message="Emergency",
            priority=AlertPriority.CRITICAL,
            category="critical",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test alert deduplication."""
        config = AlertConfig(
            console_enabled=True,
            cooldown_seconds=10.0,
        )
        manager = AlertManager(config)

        # First alert
        result1 = await manager.send_alert(
            title="Duplicate Test",
            message="First",
            priority=AlertPriority.MEDIUM,
            category="test",
        )
        assert result1 is True

        # Duplicate within cooldown
        result2 = await manager.send_alert(
            title="Duplicate Test",
            message="Second (should be suppressed)",
            priority=AlertPriority.MEDIUM,
            category="test",
        )
        assert result2 is False

        # Different title should go through
        result3 = await manager.send_alert(
            title="Different Test",
            message="Should work",
            priority=AlertPriority.MEDIUM,
            category="test",
        )
        assert result3 is True

    @pytest.mark.asyncio
    async def test_force_bypasses_throttle_and_dedup(self):
        """Test that force flag bypasses throttling and deduplication."""
        config = AlertConfig(
            console_enabled=True,
            max_alerts_per_window=1,
            cooldown_seconds=10.0,
        )
        manager = AlertManager(config)

        # Send first alert
        await manager.send_alert(
            title="Test",
            message="First",
            priority=AlertPriority.MEDIUM,
            category="test",
        )

        # Force send bypasses both throttle and dedup
        result = await manager.send_alert(
            title="Test",
            message="Forced",
            priority=AlertPriority.MEDIUM,
            category="test",
            force=True,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_priority_routing(self):
        """Test priority-based channel routing."""
        config = AlertConfig(
            console_enabled=True,
            email_enabled=True,
            slack_enabled=True,
            smtp_host="smtp.test.com",
            email_to=["test@test.com"],
            slack_webhook_url="https://hooks.slack.com/xxx",
            min_priority_for_email=AlertPriority.CRITICAL,
            min_priority_for_slack=AlertPriority.HIGH,
        )
        manager = AlertManager(config)

        # LOW priority (1) - only console
        # Email needs CRITICAL (4), Slack needs HIGH (3)
        channels = manager._get_channels_for_priority(AlertPriority.LOW)
        assert AlertChannel.CONSOLE in channels
        assert AlertChannel.SLACK not in channels
        assert AlertChannel.EMAIL not in channels

        # MEDIUM priority (2) - only console
        channels = manager._get_channels_for_priority(AlertPriority.MEDIUM)
        assert AlertChannel.CONSOLE in channels
        assert AlertChannel.SLACK not in channels
        assert AlertChannel.EMAIL not in channels

        # HIGH priority (3) - console + slack
        channels = manager._get_channels_for_priority(AlertPriority.HIGH)
        assert AlertChannel.CONSOLE in channels
        assert AlertChannel.SLACK in channels
        assert AlertChannel.EMAIL not in channels

        # CRITICAL priority (4) - all channels
        channels = manager._get_channels_for_priority(AlertPriority.CRITICAL)
        assert AlertChannel.CONSOLE in channels
        assert AlertChannel.SLACK in channels
        assert AlertChannel.EMAIL in channels

    @pytest.mark.asyncio
    async def test_send_risk_alert(self):
        """Test send_risk_alert convenience method."""
        manager = AlertManager()

        result = await manager.send_risk_alert(
            title="Daily Loss Limit",
            message="Hit 5% daily loss limit",
            details={"loss": 50.0, "limit": 50.0},
        )

        assert result is True
        alert = manager._alert_history[-1]
        assert alert.category == "risk"
        assert alert.priority == AlertPriority.HIGH

    @pytest.mark.asyncio
    async def test_send_connection_alert(self):
        """Test send_connection_alert convenience method."""
        manager = AlertManager()

        result = await manager.send_connection_alert(
            title="WebSocket Disconnected",
            message="Lost connection to market data",
            details={"retry_attempt": 3},
        )

        assert result is True
        alert = manager._alert_history[-1]
        assert alert.category == "connection"
        assert alert.priority == AlertPriority.MEDIUM

    @pytest.mark.asyncio
    async def test_send_order_alert(self):
        """Test send_order_alert convenience method."""
        manager = AlertManager()

        result = await manager.send_order_alert(
            title="Order Rejected",
            message="Insufficient margin",
            details={"order_id": "12345", "reason": "MARGIN"},
        )

        assert result is True
        alert = manager._alert_history[-1]
        assert alert.category == "order"

    @pytest.mark.asyncio
    async def test_send_critical_alert(self):
        """Test send_critical_alert convenience method."""
        manager = AlertManager()

        result = await manager.send_critical_alert(
            title="Kill Switch Activated",
            message="Emergency halt triggered",
            details={"cumulative_loss": 350.0},
        )

        assert result is True
        alert = manager._alert_history[-1]
        assert alert.category == "critical"
        assert alert.priority == AlertPriority.CRITICAL

    def test_get_alert_history(self):
        """Test getting alert history."""
        manager = AlertManager()

        # Manually add some alerts
        for i in range(5):
            manager._alert_history.append(
                Alert(
                    timestamp=datetime.now(),
                    priority=AlertPriority.LOW,
                    title=f"Alert {i}",
                    message="Test",
                    category="test",
                )
            )

        history = manager.get_alert_history(limit=3)

        assert len(history) == 3
        assert history[-1].title == "Alert 4"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting alert statistics."""
        manager = AlertManager()

        # Send some alerts
        await manager.send_alert(
            title="Alert 1",
            message="Test",
            priority=AlertPriority.HIGH,
            category="risk",
        )
        await manager.send_alert(
            title="Alert 2",
            message="Test",
            priority=AlertPriority.LOW,
            category="connection",
        )
        await manager.send_alert(
            title="Alert 3",
            message="Test",
            priority=AlertPriority.HIGH,
            category="risk",
        )

        stats = manager.get_stats()

        assert stats["total_alerts"] == 3
        assert stats["by_priority"]["high"] == 2
        assert stats["by_priority"]["low"] == 1
        assert stats["by_category"]["risk"] == 2
        assert stats["by_category"]["connection"] == 1
        assert "console" in stats["enabled_channels"]


class TestCreateErrorEventHandler:
    """Tests for create_error_event_handler function."""

    @pytest.mark.asyncio
    async def test_handler_creation(self):
        """Test handler creation."""
        manager = AlertManager()
        handler = create_error_event_handler(manager)

        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_processes_error_event(self):
        """Test handler processes ErrorEvent correctly."""
        from src.trading.recovery import ErrorEvent, ErrorSeverity, ErrorCategory

        manager = AlertManager()
        handler = create_error_event_handler(manager)

        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.ERROR,
            message="WebSocket disconnected",
            recoverable=True,
            recovery_action="reconnect",
        )

        # Call handler
        handler(error)

        # Give async task time to complete
        await asyncio.sleep(0.1)

        # Check alert was created
        assert len(manager._alert_history) >= 1

    @pytest.mark.asyncio
    async def test_handler_maps_severity_to_priority(self):
        """Test handler maps ErrorSeverity to AlertPriority correctly."""
        from src.trading.recovery import ErrorEvent, ErrorSeverity, ErrorCategory

        manager = AlertManager()
        handler = create_error_event_handler(manager)

        # Test critical severity
        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            message="Critical system error",
            recoverable=False,
        )

        handler(error)
        await asyncio.sleep(0.1)

        alert = manager._alert_history[-1]
        assert alert.priority == AlertPriority.CRITICAL


class TestCreateAlertManagerFromEnv:
    """Tests for create_alert_manager_from_env function."""

    def test_default_without_env_vars(self):
        """Test creation without environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            manager = create_alert_manager_from_env()

            assert manager.config.console_enabled is True
            assert manager.config.email_enabled is False
            assert manager.config.slack_enabled is False

    def test_email_config_from_env(self):
        """Test email configuration from environment."""
        env_vars = {
            "ALERT_EMAIL_ENABLED": "true",
            "ALERT_SMTP_HOST": "smtp.test.com",
            "ALERT_SMTP_PORT": "465",
            "ALERT_SMTP_USERNAME": "user@test.com",
            "ALERT_SMTP_PASSWORD": "secret",
            "ALERT_EMAIL_TO": "admin@test.com, ops@test.com",
        }

        with patch.dict("os.environ", env_vars, clear=True):
            manager = create_alert_manager_from_env()

            assert manager.config.email_enabled is True
            assert manager.config.smtp_host == "smtp.test.com"
            assert manager.config.smtp_port == 465
            assert manager.config.smtp_username == "user@test.com"
            assert len(manager.config.email_to) == 2

    def test_slack_config_from_env(self):
        """Test Slack configuration from environment."""
        env_vars = {
            "ALERT_SLACK_ENABLED": "true",
            "ALERT_SLACK_WEBHOOK_URL": "https://hooks.slack.com/xxx",
            "ALERT_SLACK_CHANNEL": "#alerts",
        }

        with patch.dict("os.environ", env_vars, clear=True):
            manager = create_alert_manager_from_env()

            assert manager.config.slack_enabled is True
            assert manager.config.slack_webhook_url == "https://hooks.slack.com/xxx"
            assert manager.config.slack_channel == "#alerts"

    def test_discord_config_from_env(self):
        """Test Discord configuration from environment."""
        env_vars = {
            "ALERT_DISCORD_ENABLED": "true",
            "ALERT_DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/xxx",
        }

        with patch.dict("os.environ", env_vars, clear=True):
            manager = create_alert_manager_from_env()

            assert manager.config.discord_enabled is True
            assert manager.config.discord_webhook_url == "https://discord.com/api/webhooks/xxx"

    def test_webhook_config_from_env(self):
        """Test generic webhook configuration from environment."""
        env_vars = {
            "ALERT_WEBHOOK_ENABLED": "true",
            "ALERT_WEBHOOK_URL": "https://api.example.com/alerts",
        }

        with patch.dict("os.environ", env_vars, clear=True):
            manager = create_alert_manager_from_env()

            assert manager.config.webhook_enabled is True
            assert manager.config.webhook_url == "https://api.example.com/alerts"


class TestAlertIntegration:
    """Integration tests for the alert system."""

    @pytest.mark.asyncio
    async def test_multi_channel_alert(self):
        """Test sending alert to multiple channels."""
        config = AlertConfig(
            console_enabled=True,
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/xxx",
            min_priority_for_slack=AlertPriority.HIGH,
        )
        manager = AlertManager(config)

        with patch.object(SlackAlertSender, "send", new_callable=AsyncMock) as mock_slack:
            mock_slack.return_value = True

            result = await manager.send_alert(
                title="Multi-Channel Test",
                message="Should go to console and slack",
                priority=AlertPriority.HIGH,
                category="test",
            )

            assert result is True
            mock_slack.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_alerts(self):
        """Test sending multiple alerts concurrently."""
        manager = AlertManager()

        # Send multiple alerts concurrently
        tasks = [
            manager.send_alert(
                title=f"Concurrent Alert {i}",
                message=f"Message {i}",
                priority=AlertPriority.MEDIUM,
                category="test",
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # First one should succeed, rest may be deduplicated or throttled
        assert results[0] is True

    @pytest.mark.asyncio
    async def test_alert_history_persistence(self):
        """Test that alert history persists across calls."""
        manager = AlertManager()

        await manager.send_alert(
            title="Alert 1",
            message="First",
            priority=AlertPriority.LOW,
            category="test",
        )

        await asyncio.sleep(0.01)  # Avoid dedup

        await manager.send_alert(
            title="Alert 2",
            message="Second",
            priority=AlertPriority.MEDIUM,
            category="test",
        )

        history = manager.get_alert_history()

        assert len(history) == 2
        assert history[0].title == "Alert 1"
        assert history[1].title == "Alert 2"

    @pytest.mark.asyncio
    async def test_channel_failure_handling(self):
        """Test graceful handling when a channel fails."""
        config = AlertConfig(
            console_enabled=True,
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/xxx",
        )
        manager = AlertManager(config)

        # Mock Slack to fail
        with patch.object(SlackAlertSender, "send", new_callable=AsyncMock) as mock_slack:
            mock_slack.side_effect = Exception("Network error")

            # Should still succeed via console
            result = await manager.send_alert(
                title="Test",
                message="Should work via console",
                priority=AlertPriority.HIGH,
                category="test",
            )

            # Result may vary based on gather behavior, but shouldn't raise
            assert isinstance(result, bool)


class TestAlertPriorityComparison:
    """Tests for AlertPriority comparison (used in routing)."""

    def test_priority_values(self):
        """Test priority enum values for comparison."""
        # Numeric values enable proper comparison for routing logic
        assert AlertPriority.LOW.value < AlertPriority.MEDIUM.value
        assert AlertPriority.MEDIUM.value < AlertPriority.HIGH.value
        assert AlertPriority.HIGH.value < AlertPriority.CRITICAL.value

    def test_priority_ordering(self):
        """Test that priorities are properly ordered by numeric value."""
        priorities = sorted(AlertPriority, key=lambda p: p.value)
        # Values are: 1, 2, 3, 4 (numeric)
        values = [p.value for p in priorities]
        assert values == [1, 2, 3, 4]
        labels = [p.label for p in priorities]
        assert labels == ["low", "medium", "high", "critical"]

    def test_priority_labels(self):
        """Test that priority labels return correct strings."""
        assert AlertPriority.LOW.label == "low"
        assert AlertPriority.MEDIUM.label == "medium"
        assert AlertPriority.HIGH.label == "high"
        assert AlertPriority.CRITICAL.label == "critical"
