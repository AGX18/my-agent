import json
from typing import ClassVar

import pytest

from agent import (
    ASSISTANT_INSTRUCTIONS,
    CallAnalysis,
    PropertySearchResult,
    RoomMetadata,
    build_call_analysis,
    build_call_analysis_messages,
    build_call_payload,
    build_llm_call_analysis,
    duration_secs_from_session_report,
    extract_participant_metadata,
    extract_room_metadata,
    format_property_search_results,
    merge_metadata,
    metadata_from_log_context,
    persist_call_analysis,
    resolve_metadata_tenant_id,
    resolve_tenant_id,
    resolve_tenant_id_from_name,
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


class FakeChatCompletionMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatCompletionChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeChatCompletionMessage(content)


class FakeChatCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChatCompletionChoice(content)]


class FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.create_kwargs = None

    async def create(self, **kwargs):
        self.create_kwargs = kwargs
        return FakeChatCompletionResponse(self.content)


class FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = FakeChatCompletions(content)


class FakeAnalysisOpenAIClient:
    def __init__(self, content: str) -> None:
        self.chat = FakeChat(content)


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


class FakeTenantConnection:
    def __init__(self, row=None) -> None:
        self.fetchrow_args = None
        self.closed = False
        self.row = row or {"id": "d600715c-4ba8-4e94-be2f-9db73abd7654"}

    async def fetchrow(self, *args):
        self.fetchrow_args = args
        return self.row

    async def close(self):
        self.closed = True


class FakePersistenceConnection:
    pass


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


def test_metadata_from_log_context_reads_tenant_name() -> None:
    metadata = metadata_from_log_context(
        {
            "tenant_name": "Acme Realty",
            "phone_number": "+201012345678",
        }
    )

    assert metadata == RoomMetadata(
        tenant_name="Acme Realty",
        phone_number="+201012345678",
    )


def test_build_call_analysis_includes_client_wants_phone_status_and_labels() -> None:
    analysis = build_call_analysis(
        "user: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف",
        RoomMetadata(
            tenant_id="d600715c-4ba8-4e94-be2f-9db73abd7654",
            phone_number="+201012345678",
        ),
        duration_secs=60,
    )

    assert analysis.sentiment == "positive"
    assert analysis.outcome == "qualified"
    assert analysis.lead_status == "qualified"
    assert analysis.duration_secs == 60
    assert analysis.intent == "buy"
    assert "Lead wants" in analysis.lead_summary
    assert "+201012345678" in analysis.details
    assert "Budget:" in analysis.details
    assert "Rooms:" in analysis.details
    assert "Location:" in analysis.details
    assert "Call outcome: qualified" in analysis.details
    assert "Sentiment: positive" in analysis.summary


def test_build_call_analysis_ignores_assistant_property_results() -> None:
    analysis = build_call_analysis(
        "\n".join(
            [
                "assistant: تمام، عايز تشتري فين؟ قولي المنطقة أو المدينة اللي في بالك.",
                "assistant: تمام، لقيت دوبلكس للبيع في مدينة الرحاب المرحلة السابعة.",
                "assistant: المساحة حوالي مية وتسعتاشر متر، تلات غرف نوم وتلات حمام، والسعر حوالي ستة مليون وستمية ألف جنيه.",
                "user: شكرا، هفكر وارد عليك.",
            ]
        ),
        RoomMetadata(
            tenant_id="d600715c-4ba8-4e94-be2f-9db73abd7654",
            phone_number="+201012345678",
        ),
    )

    assert "Budget: Not captured" in analysis.details
    assert "Rooms: Not captured" in analysis.details
    assert "Location: Not captured" in analysis.details
    assert "Property type: Not captured" in analysis.details
    assert analysis.sentiment == "neutral"
    assert analysis.outcome == "follow_up"
    assert analysis.intent == "buy"


def test_build_call_analysis_uses_only_client_preferences() -> None:
    analysis = build_call_analysis(
        "\n".join(
            [
                "assistant: السعر حوالي ستة مليون وستمية ألف جنيه وفيه تلات غرف.",
                "user: ميزانيتي خمسة مليون وعايز شقة في التجمع من تلات غرف.",
            ]
        ),
        RoomMetadata(phone_number="+201012345678"),
    )

    assert (
        "Budget: ميزانيتي خمسة مليون وعايز شقة في التجمع من تلات غرف."
        in analysis.details
    )
    assert (
        "Rooms: ميزانيتي خمسة مليون وعايز شقة في التجمع من تلات غرف."
        in analysis.details
    )
    assert (
        "Location: ميزانيتي خمسة مليون وعايز شقة في التجمع من تلات غرف."
        in analysis.details
    )
    assert (
        "Property type: ميزانيتي خمسة مليون وعايز شقة في التجمع من تلات غرف."
        in analysis.details
    )
    assert "ستة مليون وستمية" not in analysis.details


def test_build_call_analysis_detects_rent_intent() -> None:
    analysis = build_call_analysis(
        "user: عايز أأجر شقة في التجمع غرفتين",
        RoomMetadata(phone_number="+201012345678"),
    )

    assert analysis.intent == "rent"
    assert "Intent: rent" in analysis.details
    assert "Lead wants to rent" in analysis.lead_summary


def test_build_call_analysis_messages_request_strict_payload_shape() -> None:
    messages = build_call_analysis_messages(
        "assistant: Hello\nuser: I am not interested",
        RoomMetadata(phone_number="+201012345678"),
        duration_secs=90,
    )

    assert messages[0]["role"] == "system"
    assert "Return only one JSON object" in messages[0]["content"]
    assert "phone_number" in messages[0]["content"]
    assert "call_summary" in messages[0]["content"]
    assert "lead_summary" in messages[0]["content"]
    assert 'intent: one of "buy", "rent"' in messages[0]["content"]
    assert (
        '"follow_up", "qualified", "closed", "unqualified", "no_answer"'
        in messages[0]["content"]
    )
    assert '"Follow_Up", "qualified", "closed", "unqualified"' in messages[0]["content"]
    assert "+201012345678" in messages[1]["content"]
    assert "assistant: Hello\nuser: I am not interested" in messages[1]["content"]


@pytest.mark.asyncio
async def test_build_llm_call_analysis_requests_json_payload() -> None:
    llm_payload = json.dumps(
        {
            "phone_number": "+201099999999",
            "lead_status": "qualified",
            "outcome": "unqualified",
            "sentiment": "negative",
            "duration_secs": 90,
            "transcript": "assistant: Hello\nuser: I am not interested",
            "details": "Phone number: +201099999999\nIntent: wants to rent\nCall outcome: unqualified",
            "call_summary": "Unqualified call with +201099999999.",
            "lead_summary": "Lead wants to rent a two-bedroom apartment and is price sensitive.",
            "intent": "rent",
        }
    )
    client = FakeAnalysisOpenAIClient(llm_payload)

    analysis = await build_llm_call_analysis(
        "assistant: Hello\nuser: I am not interested",
        RoomMetadata(phone_number="+201012345678"),
        duration_secs=90,
        openai_client=client,
    )

    create_kwargs = client.chat.completions.create_kwargs
    assert create_kwargs["response_format"]["type"] == "json_object"
    assert create_kwargs["temperature"] == 0
    assert analysis == CallAnalysis(
        transcript="assistant: Hello\nuser: I am not interested",
        details="Phone number: +201099999999\nIntent: rent\nCall outcome: unqualified",
        summary="Unqualified call with +201099999999.",
        sentiment="negative",
        outcome="unqualified",
        lead_status="unqualified",
        lead_summary="Lead wants to rent a two-bedroom apartment and is price sensitive.",
        intent="rent",
        duration_secs=90,
    )


@pytest.mark.asyncio
async def test_persist_call_analysis_posts_to_backend_endpoint() -> None:
    analysis = CallAnalysis(
        transcript="user: عايز شقة في التجمع",
        details="Phone number: +201012345678\nCall outcome: qualified",
        summary="Call with +201012345678. Outcome: qualified. Sentiment: positive.",
        sentiment="positive",
        outcome="qualified",
        lead_status="qualified",
        lead_summary="Lead wants to buy an apartment in New Cairo.",
        intent="buy",
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
        == "https://backend.example.com/v2/tenants/d600715c-4ba8-4e94-be2f-9db73abd7654/calls"
    )
    assert FakeHttpSession.posted["headers"] == {"Content-Type": "application/json"}
    assert FakeHttpSession.posted["json"] == {
        "phone_number": "+201012345678",
        "lead_status": "qualified",
        "outcome": "qualified",
        "sentiment": "positive",
        "duration_secs": 90,
        "transcript": analysis.transcript,
        "details": analysis.details,
        "call_summary": analysis.summary,
    }


def test_build_call_payload_includes_phone_number_and_status() -> None:
    analysis = CallAnalysis(
        transcript="transcript",
        details="Phone number: +201012345678\nIntent: rent",
        summary="summary",
        sentiment="neutral",
        outcome="follow_up",
        lead_status="Follow_Up",
        lead_summary="Lead wants to rent a studio.",
        intent="rent",
    )

    payload = build_call_payload("+201012345678", analysis)

    assert payload["phone_number"] == "+201012345678"
    assert "phone" not in payload
    assert "status" not in payload
    assert payload["lead_status"] == "Follow_Up"
    assert payload["call_summary"] == "summary"
    assert "Intent: rent" in payload["details"]
    assert "lead_summary" not in payload
    assert "intent" not in payload
    assert "summary" not in payload


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


def test_extract_participant_metadata_reads_tenant_name() -> None:
    metadata = extract_participant_metadata(
        '{"tenant_name":"Acme Realty","phone_number":"+201012345678"}'
    )

    assert metadata == RoomMetadata(
        tenant_name="Acme Realty",
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
            tenant_name="Participant Tenant",
            phone_number="+201012345678",
        ),
        RoomMetadata(
            tenant_id="room-tenant",
            tenant_name="Room Tenant",
            phone_number="+201099999999",
        ),
    )

    assert merged == RoomMetadata(
        tenant_id="participant-tenant",
        tenant_name="Participant Tenant",
        phone_number="+201012345678",
    )


def test_merge_metadata_falls_back_to_room_metadata() -> None:
    merged = merge_metadata(
        RoomMetadata(tenant_id="participant-tenant"),
        RoomMetadata(
            tenant_id="room-tenant",
            tenant_name="Room Tenant",
            phone_number="+201099999999",
        ),
    )

    assert merged == RoomMetadata(
        tenant_id="participant-tenant",
        tenant_name="Room Tenant",
        phone_number="+201099999999",
    )


@pytest.mark.asyncio
async def test_resolve_tenant_id_from_name_queries_tenants_table() -> None:
    fake_connection = FakeTenantConnection()

    async def fake_connect(database_url: str):
        assert database_url == "postgres://example"
        return fake_connection

    tenant_id = await resolve_tenant_id_from_name(
        "Acme Realty",
        database_url="postgres://example",
        connect=fake_connect,
    )

    assert tenant_id == "d600715c-4ba8-4e94-be2f-9db73abd7654"
    assert fake_connection.closed is True
    sql, tenant_name = fake_connection.fetchrow_args
    assert "FROM tenants" in sql
    assert "lower(name) = lower($1)" in sql
    assert tenant_name == "Acme Realty"


@pytest.mark.asyncio
async def test_resolve_metadata_tenant_id_uses_tenant_name_when_id_missing() -> None:
    fake_connection = FakeTenantConnection(
        {"id": "550e8400-e29b-41d4-a716-446655440000"}
    )

    async def fake_connect(database_url: str):
        assert database_url == "postgres://example"
        return fake_connection

    metadata = await resolve_metadata_tenant_id(
        RoomMetadata(tenant_name="Acme Realty", phone_number="+201012345678"),
        database_url="postgres://example",
        connect=fake_connect,
    )

    assert metadata == RoomMetadata(
        tenant_id="550e8400-e29b-41d4-a716-446655440000",
        tenant_name="Acme Realty",
        phone_number="+201012345678",
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
