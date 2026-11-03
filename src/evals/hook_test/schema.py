"""Response schema for the hook test (plan/02-hook-test.md section 5.3).

``HookTestResult`` doubles as the Gemini ``response_schema`` (the SDK converts
Pydantic models to JSON Schema) and the validator for raw responses.
``validate_result`` returns an ``invalid`` verdict rather than clamping
out-of-range values silently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError


# note: no extra="forbid" — it emits additionalProperties:false into the
# response_schema, which the Gemini API rejects (400 INVALID_ARGUMENT).
# Extra keys are rejected manually in validate_result instead.
class Moment(BaseModel):
    t_seconds: float
    kind: Literal["hook", "peak", "dead_zone", "drop_risk"]
    note: str


class HookTestResult(BaseModel):
    """Field order matters: the SDK emits ``propertyOrdering`` in declaration
    order (verified in google-genai 2.32.0 _transformers.process_schema, which
    sets property_ordering = list(properties.keys()) on model_json_schema()
    output), and Gemini generates JSON in that order. The two scratch-pad
    fields come first so the model writes its analysis before the ratings."""

    video_description: str
    analysis: str
    hook_verdict: Literal["strong", "ok", "weak"]
    dropoff_timestamp_s: float = Field(ge=0.0)
    p_strong: float = Field(ge=0.0, le=1.0)
    group: Literal["strong", "weak"]
    confidence: Literal["low", "medium", "high"]
    # note: ge (not gt) — the SDK Schema type has no exclusiveMinimum field
    video_length_s_estimate: float = Field(ge=0.0)
    moments: list[Moment] = Field(default_factory=list, max_length=6)
    reasoning: str


class HookTestResultV2(BaseModel):
    """v2 "split" design: the model predicts an absolute average view
    percentage; length detrending happens in our code (score.py)."""

    video_description: str
    analysis: str
    hook_verdict: Literal["strong", "ok", "weak"]
    dropoff_timestamp_s: float = Field(ge=0.0)
    predicted_avg_view_pct: float = Field(ge=0.0, le=400.0)
    confidence: Literal["low", "medium", "high"]
    video_length_s_estimate: float = Field(ge=0.0)
    moments: list[Moment] = Field(default_factory=list, max_length=6)
    reasoning: str


SCHEMA_CLASSES = {"v1": HookTestResult, "v2": HookTestResultV2}


def schema_class(schema_version: str):
    return SCHEMA_CLASSES.get(schema_version, HookTestResult)


def validate_result(data, duration_s: float,
                    schema_version: str = "v1"):
    """Validate a parsed JSON object against the schema + duration bounds.

    Returns (result, invalid_reason, quality_flags). ``result`` is None when
    invalid_reason is set. A large video_length_s_estimate miss is a quality
    flag, not an invalidation.
    """
    cls = schema_class(schema_version)
    if isinstance(data, dict):
        extra = set(data) - set(cls.model_fields)
        if extra:
            return None, f"schema: extra field {sorted(extra)[0]}", []
        for m in data.get("moments") or []:
            if isinstance(m, dict):
                mextra = set(m) - set(Moment.model_fields)
                if mextra:
                    return None, f"schema: extra field {sorted(mextra)[0]}", []
    try:
        r = cls.model_validate(data)
    except ValidationError as e:
        return None, f"schema: {e.errors()[0]['loc']} {e.errors()[0]['msg']}", []
    if r.dropoff_timestamp_s > duration_s + 0.5:
        return None, (
            f"dropoff_timestamp_s {r.dropoff_timestamp_s} > duration {duration_s} + 0.5"
        ), []
    flags = []
    if abs(r.video_length_s_estimate - duration_s) > 3.0:
        flags.append(
            f"length_est_off:{r.video_length_s_estimate:.1f} vs {duration_s:.1f}"
        )
    return r, None, flags
