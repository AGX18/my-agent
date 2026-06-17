from typing import ClassVar

import pytest

from agent import (
    ASSISTANT_INSTRUCTIONS,
    CallAnalysis,
    PropertySearchResult,
    RoomMetadata,
    build_call_analysis,
    build_call_payload,
    duration_secs_from_session_report,
    extract_participant_metadata,
    extract_room_metadata,
    format_property_search_results,
    merge_metadata,
    metadata_from_log_context,
    persist_call_analysis,
    resolve_tenant_id,
    search_property_embeddings,
    transcript_from_session_report,
)


class FakeEmbedding:
    def __init__(self) -> None:
        self.embedding = [0.1, 0.2, 0.3]


class FakeEmbeddingResponse:
    def __init__(self) -> None:
        self.data = [FakeEmbedding()]


class FakeEmbeddings:
    async def create(self, *, model: str, input: str):  # noqa: A002
        self.model = model
        self.input = input
        return FakeEmbeddingResponse()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


class FakeRunContext:
    def __init__(self, userdata) -> None:
        self.userdata = userdata


class FakeConnection:
    def __init__(self) -> None:
        self.fetch_args = None
        self.closed = False

    async def fetch(self, *args):
        self.fetch_args = args
        return [
            {
                "property_id": 42,
                "content": "شقة غرفتين قريبة من المترو.",
                "similarity": 0.91,
            }
        ]

    async def close(self):
        self.closed = True


class FakeHttpResponse:
    status = 201

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def text(self) -> str:
        return '{"id":100,"lead_id":55}'

    async def json(self):
        return {"id": 100, "lead_id": 55}


class FakeHttpSession:
    posted: ClassVar[dict] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    def post(self, url, *, json, headers):
        self.__class__.posted = {
            "url": url,
            "json": json,
            "headers": headers,
        }
        return FakeHttpResponse()


def test_assistant_instructions_do_not_overpromise_unsupported_actions() -> None:
    assert "do not currently have tools" in ASSISTANT_INSTRUCTIONS
    assert "create appointments" in ASSISTANT_INSTRUCTIONS
    assert "send WhatsApp messages" in ASSISTANT_INSTRUCTIONS
    assert "share map pins" in ASSISTANT_INSTRUCTIONS
    assert "Never say that you booked" in ASSISTANT_INSTRUCTIONS
    assert "one of the brokers will contact them" in ASSISTANT_INSTRUCTIONS
    assert (
        "Do not say that you personally booked or scheduled it"
        in ASSISTANT_INSTRUCTIONS
    )


def test_transcript_from_session_report_extracts_messages() -> None:
    transcript = transcript_from_session_report(
        {
            "history": {
                "items": [
                    {
                        "role": "user",
                        "content": ["عايز شقة في التجمع بميزانية خمسة مليون"],
                    },
                    {
                        "role": "assistant",
                        "content": ["تمام، هدورلك على اختيارات مناسبة."],
                    },
                ]
            }
        }
    )

    assert "user: عايز شقة في التجمع بميزانية خمسة مليون" in transcript
    assert "assistant: تمام، هدورلك على اختيارات مناسبة." in transcript


def test_duration_secs_from_session_report_reads_top_level_duration() -> None:
    assert duration_secs_from_session_report({"duration_secs": 42.8}) == 42


def test_metadata_from_log_context_reads_tenant_and_phone() -> None:
    metadata = metadata_from_log_context(
        {
            "tenant_id": "d600715c-4ba8-4e94-be2f-9db73abd7654",
            "phone_number": "+201012345678",
        }
    )

    assert metadata == RoomMetadata(
        tenant_id="d600715c-4ba8-4e94-be2f-9db73abd7654",
        phone_number="+201012345678",
    )


def test_build_call_analysis_is_todo_stub() -> None:
    analysis = build_call_analysis(
        "user: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف",
        RoomMetadata(
            tenant_id="d600715c-4ba8-4e94-be2f-9db73abd7654",
            phone_number="+201012345678",
        ),
        duration_secs=60,
    )

    assert analysis.transcript == (
        "user: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف"
    )
    assert analysis.details == "TODO: implement call details"
    assert analysis.summary == "TODO: implement call summary"
    assert analysis.sentiment == "neutral"
    assert analysis.outcome == "follow_up"
    assert analysis.lead_status == "Follow_Up"
    assert analysis.duration_secs == 60


@pytest.mark.asyncio
async def test_persist_call_analysis_posts_to_backend_endpoint() -> None:
    analysis = CallAnalysis(
        transcript="user: عايز شقة في التجمع",
        details="Phone number: +201012345678\nCall outcome: qualified",
        summary="Call with +201012345678. Outcome: qualified. Sentiment: positive.",
        sentiment="positive",
        outcome="qualified",
        lead_status="qualified",
        duration_secs=90,
    )

    call_id, lead_id = await persist_call_analysis(
        "d600715c-4ba8-4e94-be2f-9db73abd7654",
        "+201012345678",
        analysis,
        backend_base_url="https://backend.example.com",
        session_factory=FakeHttpSession,
    )

    assert (call_id, lead_id) == (100, 55)
    assert (
        FakeHttpSession.posted["url"]
        == "https://backend.example.com/tenants/d600715c-4ba8-4e94-be2f-9db73abd7654/calls"
    )
    assert FakeHttpSession.posted["headers"] == {"Content-Type": "application/json"}
    assert FakeHttpSession.posted["json"] == {
        "phone_number": "+201012345678",
        "status": "qualified",
        "lead_status": "qualified",
        "transcript": analysis.transcript,
        "details": analysis.details,
        "summary": analysis.summary,
        "sentiment": "positive",
        "outcome": "qualified",
        "duration_secs": 90,
    }


def test_build_call_payload_includes_phone_number_and_status() -> None:
    analysis = CallAnalysis(
        transcript="transcript",
        details="details",
        summary="summary",
        sentiment="neutral",
        outcome="follow_up",
        lead_status="Follow_Up",
    )

    payload = build_call_payload("+201012345678", analysis)

    assert payload["phone_number"] == "+201012345678"
    assert "phone" not in payload
    assert payload["status"] == "Follow_Up"
    assert payload["lead_status"] == "Follow_Up"


@pytest.mark.asyncio
async def test_search_property_embeddings_queries_tenant_scoped_pgvector() -> None:
    fake_connection = FakeConnection()

    async def fake_connect(database_url: str):
        assert database_url == "postgres://example"
        return fake_connection

    results = await search_property_embeddings(
        "عايز شقة قريبة من المترو",
        "550e8400-e29b-41d4-a716-446655440000",
        database_url="postgres://example",
        openai_client=FakeOpenAIClient(),
        connect=fake_connect,
    )

    assert results == [
        PropertySearchResult(
            property_id=42,
            content="شقة غرفتين قريبة من المترو.",
            similarity=0.91,
        )
    ]
    assert fake_connection.closed is True

    sql, tenant_id, vector, limit = fake_connection.fetch_args
    assert "FROM property_embeddings" in sql
    assert "tenant_id = $1::uuid" in sql
    assert "ORDER BY embedding <=> $2::vector" in sql
    assert tenant_id == "550e8400-e29b-41d4-a716-446655440000"
    assert vector == "[0.1,0.2,0.3]"
    assert limit == 4


def test_extract_room_metadata_reads_snake_case_values() -> None:
    metadata = extract_room_metadata(
        '{"tenant_id":"550e8400-e29b-41d4-a716-446655440000","phone_number":"+201001112222"}'
    )

    assert metadata == RoomMetadata(
        tenant_id="550e8400-e29b-41d4-a716-446655440000",
        phone_number="+201001112222",
    )


def test_extract_participant_metadata_reads_snake_case_values() -> None:
    metadata = extract_participant_metadata(
        '{"tenant_id":"d600715c-4ba8-4e94-be2f-9db73abd7654","phone_number":"+201012345678"}'
    )

    assert metadata == RoomMetadata(
        tenant_id="d600715c-4ba8-4e94-be2f-9db73abd7654",
        phone_number="+201012345678",
    )


def test_extract_room_metadata_reads_camel_case_tenant_only() -> None:
    metadata = extract_room_metadata(
        '{"tenantId":"550e8400-e29b-41d4-a716-446655440000","phoneNumber":"+201001112222"}'
    )

    assert metadata == RoomMetadata(
        tenant_id="550e8400-e29b-41d4-a716-446655440000",
    )


def test_extract_room_metadata_handles_missing_or_invalid_values() -> None:
    assert extract_room_metadata(None) == RoomMetadata()
    assert extract_room_metadata("not-json") == RoomMetadata()
    assert extract_room_metadata("[]") == RoomMetadata()
    assert (
        extract_room_metadata('{"tenant_id":123,"phone_number":false}')
        == RoomMetadata()
    )


def test_merge_metadata_prefers_participant_metadata() -> None:
    merged = merge_metadata(
        RoomMetadata(
            tenant_id="participant-tenant",
            phone_number="+201012345678",
        ),
        RoomMetadata(
            tenant_id="room-tenant",
            phone_number="+201099999999",
        ),
    )

    assert merged == RoomMetadata(
        tenant_id="participant-tenant",
        phone_number="+201012345678",
    )


def test_merge_metadata_falls_back_to_room_metadata() -> None:
    merged = merge_metadata(
        RoomMetadata(tenant_id="participant-tenant"),
        RoomMetadata(
            tenant_id="room-tenant",
            phone_number="+201099999999",
        ),
    )

    assert merged == RoomMetadata(
        tenant_id="participant-tenant",
        phone_number="+201099999999",
    )


def test_resolve_tenant_id_prefers_explicit_argument() -> None:
    tenant_id = resolve_tenant_id(
        FakeRunContext(RoomMetadata(tenant_id="metadata-tenant")),
        "explicit-tenant",
    )

    assert tenant_id == "explicit-tenant"


def test_resolve_tenant_id_reads_room_metadata() -> None:
    tenant_id = resolve_tenant_id(
        FakeRunContext(RoomMetadata(tenant_id="metadata-tenant"))
    )

    assert tenant_id == "metadata-tenant"


def test_format_property_search_results_handles_no_matches() -> None:
    assert (
        format_property_search_results([])
        == "No matching property information was found in the property database."
    )


def test_format_property_search_results_includes_content_and_similarity() -> None:
    result = format_property_search_results(
        [
            PropertySearchResult(
                property_id=7,
                content="فيلا بحديقة خاصة في التجمع.",
                similarity=0.876,
            )
        ]
    )

    assert "Property 7" in result
    assert "فيلا بحديقة خاصة في التجمع." in result
    assert "similarity 0.88" in result
