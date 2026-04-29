"""Enterprise messaging and webhook integrations for openvibe.

Each integration listens for inbound messages on an external platform
(Slack, Discord, Telegram, …) and routes them through the openvibe agent.

Usage::

    vibe gateway          # start all enabled integrations
    vibe gateway --help   # see options

Configuration (openvibe.json)::

    {
      "integrations": {
        "slack": {
          "enabled": true,
          "bot_token": "${SLACK_BOT_TOKEN}",
          "app_token": "${SLACK_APP_TOKEN}"
        }
      }
    }
"""

from openvibe.integrations.gateway import GatewayManager

__all__ = ["GatewayManager"]
