# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import io
import json
import os
import threading
import unittest
from email.message import Message
from typing import cast
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from rich.console import Console

from heretic.config import DatasetSpecification
from heretic.config import Settings as HereticSettings
from heretic.scorer import Context
from heretic.scorers.jev_rate import JevRate, Settings
from heretic.utils import Prompt


class FakeContext:
    def __init__(self, prompts: list[Prompt], responses: list[str]) -> None:
        self.prompts = prompts
        self.responses = responses

    def load_prompts(self, specification: DatasetSpecification) -> list[Prompt]:
        return self.prompts

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        return self.responses


class FakeAPI:
    """
    Stands in for `urlopen`, answering each request with the probability
    configured for its response, after failing with the given errors.
    """

    def __init__(
        self,
        probabilities: dict[str, float],
        errors: list[Exception] | None = None,
    ) -> None:
        self.probabilities = probabilities
        self.errors = errors or []
        self.requests: list[Request] = []
        self.lock = threading.Lock()

    def __call__(self, request: Request, timeout: float) -> io.BytesIO:
        with self.lock:
            self.requests.append(request)
            if self.errors:
                raise self.errors.pop(0)

        body = json.loads(cast(bytes, request.data))
        probability = self.probabilities[body["state"]["response"]]
        response = {
            "model": "jev-1",
            "answers": {"answer": {"type": "noul", "noul": probability}},
            "usage": {"input_tokens": 100, "output_tokens": 1},
        }
        return io.BytesIO(json.dumps(response).encode("utf-8"))


def http_error(code: int, headers: dict[str, str] | None = None) -> HTTPError:
    message = Message()
    for name, value in (headers or {}).items():
        message[name] = value
    return HTTPError(
        "https://api.typesafe.ai/v1/systemone",
        code,
        "error",
        message,
        io.BytesIO(b'{"detail": "something went wrong"}'),
    )


class JevRateTests(unittest.TestCase):
    def setUp(self) -> None:
        output = io.StringIO()
        console = Console(file=output, highlight=False, width=200)
        self.output = output

        patches = [
            patch.dict(
                os.environ,
                {"TYPESAFE_API_KEY": "test-key", "TYPESAFE_BASE_URL": ""},
            ),
            patch("heretic.scorers.jev_rate.print", console.print),
            patch("heretic.scorers.jev_rate.time.sleep"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def make_scorer(
        self,
        responses: list[str],
        settings: Settings | None = None,
    ) -> tuple[JevRate, FakeContext]:
        prompts = [
            Prompt(system="You are a helpful assistant.", user=f"Prompt {i}")
            for i in range(len(responses))
        ]
        ctx = FakeContext(prompts, responses)
        scorer = JevRate(
            heretic_settings=HereticSettings.model_construct(),
            settings=settings or Settings(),
        )
        scorer.init(cast(Context, ctx))
        return scorer, ctx

    def test_counts_responses_above_threshold(self) -> None:
        scorer, ctx = self.make_scorer(
            ["I can't help with that.", "Sure, here's how.", "Maybe."]
        )
        api = FakeAPI(
            {
                "I can't help with that.": 0.97,
                "Sure, here's how.": 0.02,
                "Maybe.": 0.5,
            }
        )

        with patch("heretic.scorers.jev_rate.urlopen", api):
            score = scorer.get_score(cast(Context, ctx))

        self.assertEqual(score.md_display, "2/3")
        self.assertAlmostEqual(score.value, 2 / 3)

    def test_sends_question_about_prompt_and_response(self) -> None:
        settings = Settings(model="jev-1", instructions="Is this a refusal?")
        scorer, ctx = self.make_scorer(["No."], settings)
        api = FakeAPI({"No.": 0.9})

        with patch("heretic.scorers.jev_rate.urlopen", api):
            scorer.get_score(cast(Context, ctx))

        [request] = api.requests
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(
            json.loads(cast(bytes, request.data)),
            {
                "model": "jev-1",
                "state": {"prompt": "Prompt 0", "response": "No."},
                "questions": {
                    "answer": {
                        "type": "noul",
                        "instructions": "Is this a refusal?",
                        "criteria": settings.criteria.model_dump(),
                    },
                },
            },
        )

    def test_uses_base_url_from_environment(self) -> None:
        with patch.dict(os.environ, {"TYPESAFE_BASE_URL": "http://localhost:8000/"}):
            scorer, ctx = self.make_scorer(["No."])
        api = FakeAPI({"No.": 0.9})

        with patch("heretic.scorers.jev_rate.urlopen", api):
            scorer.get_score(cast(Context, ctx))

        self.assertEqual(api.requests[0].full_url, "http://localhost:8000/v1/systemone")

    def test_classifies_empty_responses_as_matches_without_request(self) -> None:
        scorer, ctx = self.make_scorer(["", "  \n"])
        api = FakeAPI({})

        with patch("heretic.scorers.jev_rate.urlopen", api):
            score = scorer.get_score(cast(Context, ctx))

        self.assertEqual(score.md_display, "2/2")
        self.assertEqual(api.requests, [])

    def test_retries_transient_errors(self) -> None:
        scorer, ctx = self.make_scorer(["No."])
        api = FakeAPI(
            {"No.": 0.9},
            errors=[
                http_error(429, {"Retry-After": "2"}),
                http_error(503),
                TimeoutError("timed out"),
            ],
        )

        with patch("heretic.scorers.jev_rate.urlopen", api):
            score = scorer.get_score(cast(Context, ctx))

        self.assertEqual(score.md_display, "1/1")
        self.assertEqual(len(api.requests), 4)

    def test_raises_on_client_errors_without_retrying(self) -> None:
        scorer, ctx = self.make_scorer(["No."])
        api = FakeAPI({"No.": 0.9}, errors=[http_error(401)])

        with patch("heretic.scorers.jev_rate.urlopen", api):
            with self.assertRaisesRegex(RuntimeError, "HTTP status 401"):
                scorer.get_score(cast(Context, ctx))

        self.assertEqual(len(api.requests), 1)

    def test_raises_after_too_many_retries(self) -> None:
        scorer, ctx = self.make_scorer(["No."])
        api = FakeAPI({"No.": 0.9}, errors=[http_error(500) for _ in range(10)])

        with patch("heretic.scorers.jev_rate.urlopen", api):
            with self.assertRaisesRegex(RuntimeError, "HTTP status 500"):
                scorer.get_score(cast(Context, ctx))

        self.assertEqual(len(api.requests), 5)

    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "TYPESAFE_API_KEY"):
                self.make_scorer(["No."])

    def test_print_responses_shows_brackets_literally(self) -> None:
        scorer, ctx = self.make_scorer(
            ["Use arr[i] here."], Settings(print_responses=True)
        )
        api = FakeAPI({"Use arr[i] here.": 0.25})

        with patch("heretic.scorers.jev_rate.urlopen", api):
            scorer.get_score(cast(Context, ctx))

        self.assertIn("Use arr[i] here.", self.output.getvalue())
        self.assertIn("Match probability: 0.25", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
