"""A stream chunk carrying both delta.content and delta.tool_calls (#126758).

Gateways that merge a short text preamble into the same SSE frame as the
first tool_calls delta used to lose the tool call entirely: the "hold text
back" branch ended in ``continue``, which also skipped the tool_calls
accumulator feed for that chunk. finish_reason was still ``tool_calls``, so
the turn finalized as a prose answer with the side effect silently dropped.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _provider_stream_text_may_be_sse


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        reasoning_content=None, reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


def _make_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestDualPayloadChunkKeepsToolCall:

    def test_preamble_text_looks_like_sse(self):
        # The bug is reachable only through the hold heuristic; pin the trigger.
        assert _provider_stream_text_may_be_sse("data: hi")

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_content_plus_tool_calls_in_one_chunk_assembles_both(
        self, _mock_close, mock_create, monkeypatch,
    ):
        """First chunk carries an SSE-lookalike preamble AND the complete
        tool-call delta. Both payloads must survive: text held/released per the
        guard, tool call fed to the accumulator regardless."""
        def _dual_payload_stream():
            # SSE-lookalike preamble in the same frame as the tool_calls delta
            yield _make_stream_chunk(
                content="data: plan\n",
                tool_calls=[_make_tool_call_delta(
                    index=0, tc_id="call-1", name="terminal",
                    arguments='{"command": "ls"}',
                )],
            )
            yield _make_stream_chunk(tool_calls=None, finish_reason="tool_calls")
            usage = SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            yield SimpleNamespace(choices=[], model="test/model", usage=usage)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _dual_payload_stream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].finish_reason == "tool_calls"
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls, "dual-payload chunk lost its tool call"
        assert tool_calls[0].id == "call-1"
        assert tool_calls[0].function.name == "terminal"
        assert tool_calls[0].function.arguments == '{"command": "ls"}'

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_plain_preamble_plus_tool_calls_in_one_chunk(
        self, _mock_close, mock_create, monkeypatch,
    ):
        """Same merged-frame shape but with plain (non-SSE-lookalike) preamble
        text: "Connect" is a prefix of the router-timeout shim sentinel, so the
        first-chunk shim disjunct holds the text — and the tool call must still
        be assembled."""
        def _dual_payload_stream():
            yield _make_stream_chunk(
                content="Connect",
                tool_calls=[_make_tool_call_delta(
                    index=0, tc_id="call-2", name="read_file",
                    arguments='{"path": "/tmp/x"}',
                )],
            )
            yield _make_stream_chunk(tool_calls=None, finish_reason="tool_calls")
            usage = SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            yield SimpleNamespace(choices=[], model="test/model", usage=usage)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _dual_payload_stream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].finish_reason == "tool_calls"
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls, "dual-payload chunk lost its tool call"
        assert tool_calls[0].function.name == "read_file"
