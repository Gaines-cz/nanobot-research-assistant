"""Async message queue for decoupled channel-agent communication."""

import asyncio

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self, maxsize: int = 1000):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=maxsize)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        try:
            await asyncio.wait_for(self.inbound.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Inbound queue full ({} items), message dropped: {}",
                self.inbound.qsize(),
                msg.sender_id
            )

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        try:
            await asyncio.wait_for(self.outbound.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Outbound queue full ({} items), message dropped: {}",
                self.outbound.qsize(),
                msg.chat_id
            )

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
