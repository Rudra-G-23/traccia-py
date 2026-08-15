"""Tests for TracciaCallbackHandler chain/tool/LLM span nesting."""

from __future__ import annotations

import unittest
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from traccia import get_tracer, set_tracer_provider
from traccia.integrations.langchain import TracciaCallbackHandler
from traccia.tracer import TracerProvider
from traccia.tracer.span import SpanStatus


def _chat_result(prompt_tokens: int, completion_tokens: int) -> ChatResult:
    message = AIMessage(content="hello there")
    generation = ChatGeneration(message=message)
    return ChatResult(
        generations=[generation],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        },
    )


class TestLangChainCallbackSpanNesting(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = TracerProvider()
        set_tracer_provider(self.provider)
        self.tracer = get_tracer("test")
        self.handler = TracciaCallbackHandler()

    def test_chain_spans_nest_under_graph_and_llm_nests_under_chain(self) -> None:
        graph_run_id = uuid.uuid4()
        node_run_id = uuid.uuid4()
        llm_run_id = uuid.uuid4()

        # Simulate LangGraph -> node -> chat model
        self.handler.on_chain_start(
            {"id": ["langgraph", "graph", "StateGraph"]},
            {"topic": "plants"},
            run_id=graph_run_id,
            parent_run_id=None,
        )
        self.handler.on_chain_start(
            {"id": ["langchain", "schema", "runnable", "RunnableSequence"]},
            {"topic": "plants"},
            run_id=node_run_id,
            parent_run_id=graph_run_id,
            metadata={"langgraph_node": "ideator"},
        )
        self.handler.on_chat_model_start(
            {"id": ["langchain_groq", "chat_models", "ChatGroq"]},
            [[HumanMessage(content="hi")]],
            run_id=llm_run_id,
            parent_run_id=node_run_id,
        )

        graph_span = self.handler._spans[graph_run_id]
        node_span = self.handler._spans[node_run_id]
        llm_span = self.handler._spans[llm_run_id]

        self.assertEqual(node_span.parent_span_id, graph_span.context.span_id)
        self.assertEqual(llm_span.parent_span_id, node_span.context.span_id)
        self.assertEqual(node_span.attributes["chain.name"], "ideator")
        self.assertEqual(node_span.attributes["langchain.type"], "chain")

        self.handler.on_llm_end(_chat_result(72, 153), run_id=llm_run_id, parent_run_id=node_run_id)
        self.handler.on_chain_end({"ideas": "..."}, run_id=node_run_id, parent_run_id=graph_run_id)
        self.handler.on_chain_end({"story": "..."}, run_id=graph_run_id, parent_run_id=None)

        self.assertNotIn(llm_run_id, self.handler._spans)
        self.assertNotIn(node_run_id, self.handler._spans)
        self.assertNotIn(graph_run_id, self.handler._spans)
        self.assertEqual(llm_span.attributes["llm.usage.prompt_tokens"], 72)
        self.assertEqual(llm_span.attributes["llm.usage.completion_tokens"], 153)

    def test_chain_error_sets_error_status_and_cleans_up(self) -> None:
        run_id = uuid.uuid4()
        self.handler.on_chain_start(
            {"id": ["langchain", "schema", "runnable", "RunnableSequence"]},
            {},
            run_id=run_id,
            parent_run_id=None,
        )
        span = self.handler._spans[run_id]

        self.handler.on_chain_error(ValueError("boom"), run_id=run_id, parent_run_id=None)

        self.assertEqual(span.status, SpanStatus.ERROR)
        self.assertNotIn(run_id, self.handler._spans)

    def test_tool_spans_nest_and_record_input_output(self) -> None:
        parent_run_id = uuid.uuid4()
        tool_run_id = uuid.uuid4()

        self.handler.on_chain_start(
            {"id": ["langchain", "agents", "AgentExecutor"]},
            {},
            run_id=parent_run_id,
            parent_run_id=None,
        )
        parent_span = self.handler._spans[parent_run_id]

        self.handler.on_tool_start(
            {"name": "web_search"},
            "query=plants",
            run_id=tool_run_id,
            parent_run_id=parent_run_id,
        )
        tool_span = self.handler._spans[tool_run_id]

        self.assertEqual(tool_span.parent_span_id, parent_span.context.span_id)
        self.assertEqual(tool_span.attributes["tool.name"], "web_search")
        self.assertEqual(tool_span.attributes["span.type"], "TOOL")

        self.handler.on_tool_end("3 results found", run_id=tool_run_id, parent_run_id=parent_run_id)

        self.assertNotIn(tool_run_id, self.handler._spans)
        self.assertEqual(tool_span.attributes["tool.output"], "3 results found")

    def test_tool_error_sets_error_status(self) -> None:
        run_id = uuid.uuid4()
        self.handler.on_tool_start(
            {"name": "web_search"}, "query", run_id=run_id, parent_run_id=None
        )
        span = self.handler._spans[run_id]

        self.handler.on_tool_error(RuntimeError("tool failed"), run_id=run_id, parent_run_id=None)

        self.assertEqual(span.status, SpanStatus.ERROR)
        self.assertNotIn(run_id, self.handler._spans)


if __name__ == "__main__":
    unittest.main()
