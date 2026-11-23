"""DOCBOT-3901: api/v1_models.py <-> src/lib/schemas/v1.ts drift check."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel

from api import v1_models as vm

TS_FILE = Path(__file__).resolve().parents[2] / "src" / "lib" / "schemas" / "v1.ts"
SOURCE = TS_FILE.read_text()

MIRRORED = [
    vm.V1ChatMessage, vm.V1HybridChatRequest, vm.V1HybridChatResponse, vm.V1AutopilotRequest,
    vm.V1AutopilotResponse, vm.V1ConnectorRegisterRequest, vm.V1ConnectorRegisterResponse,
    vm.V1ConnectorListResponse, vm.V1Error, vm.V1ErrorResponse,
]


def _ts_block(name: str) -> str:
    m = re.search(rf"export const {name}Schema = (.*?)(?=\nexport |\Z)", SOURCE, re.S)
    assert m, f"{name}Schema missing from v1.ts"
    return m.group(1)


@pytest.mark.parametrize("model", MIRRORED, ids=lambda m: m.__name__)
def test_fields_identical(model: type[BaseModel]) -> None:
    assert set(re.findall(r"^\s{2}([a-z_]+):", _ts_block(model.__name__), re.M)) == set(model.model_fields)


def test_error_types_identical() -> None:
    assert set(re.findall(r'"([a-z_]+)"', _ts_block("ErrorType"))) == set(get_args(vm.ErrorType))
