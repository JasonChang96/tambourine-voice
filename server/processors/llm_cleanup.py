"""LLM-based text cleanup processor for dictation using idiomatic Pipecat patterns."""

from typing import Any

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputTransportMessageFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from utils.logger import logger

# System prompt for text cleanup
CLEANUP_SYSTEM_PROMPT = """You are a dictation cleanup assistant. Your task is to clean up transcribed speech.

Rules:
- Remove filler words (um, uh, like, you know, basically, actually, literally, sort of, kind of)
- Fix grammar and punctuation
- Capitalize sentences properly
- Keep the original meaning and tone intact
- Do NOT add any new information or change the intent
- Output ONLY the cleaned text, nothing else - no explanations, no quotes, no prefixes

Example:
Input: "um so basically I was like thinking we should uh you know update the readme file"
Output: I was thinking we should update the readme file."""


class TranscriptionToLLMConverter(FrameProcessor):
    """Converts TranscriptionFrame to OpenAILLMContextFrame for LLM cleanup.

    This processor receives accumulated transcription text and converts it
    to an LLM context with the cleanup system prompt, triggering the LLM
    service to generate cleaned text.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the converter."""
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Convert transcription frames to LLM context frames.

        Args:
            frame: The frame to process
            direction: The direction of frame flow
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text
            if text and text.strip():
                logger.debug(f"Converting transcription to LLM context: {text[:50]}...")

                # Create OpenAI-compatible context with cleanup prompt
                context = OpenAILLMContext(
                    messages=[
                        {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ]
                )

                # Push context frame to trigger LLM processing
                await self.push_frame(OpenAILLMContextFrame(context=context), direction)
            return

        # Pass through all other frames unchanged
        await self.push_frame(frame, direction)


class LLMResponseToRTVIConverter(FrameProcessor):
    """Aggregates LLM response and converts to RTVI message for client.

    This processor collects streamed TextFrames between LLMFullResponseStartFrame
    and LLMFullResponseEndFrame, then sends the complete cleaned text as an
    RTVI server message to the client.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the response converter."""
        super().__init__(**kwargs)
        self._accumulator: str = ""
        self._is_accumulating: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Accumulate LLM response and convert to RTVI message.

        Args:
            frame: The frame to process
            direction: The direction of frame flow
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Start accumulating LLM response
            self._accumulator = ""
            self._is_accumulating = True
            return

        if isinstance(frame, TextFrame) and self._is_accumulating:
            # Accumulate text chunks from LLM
            self._accumulator += frame.text
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            # LLM response complete - send cleaned text to client
            self._is_accumulating = False
            cleaned_text = self._accumulator.strip()

            if cleaned_text:
                logger.info(f"Cleaned text: '{cleaned_text}'")

                # Create RTVI message for client
                rtvi_message = {
                    "label": "rtvi-ai",
                    "type": "server-message",
                    "data": {"type": "transcript", "text": cleaned_text},
                }
                await self.push_frame(OutputTransportMessageFrame(message=rtvi_message), direction)

            self._accumulator = ""
            return

        # Pass through all other frames unchanged
        await self.push_frame(frame, direction)
