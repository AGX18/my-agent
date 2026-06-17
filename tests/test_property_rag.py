import pytest

from agent import (
    PropertySearchResult,
    RoomMetadata,
    extract_participant_metadata,
    extract_room_metadata,
    format_property_search_results,
    merge_metadata,
    resolve_tenant_id,
    search_property_embeddings,
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


def test_extract_room_metadata_reads_camel_case_values() -> None:
    metadata = extract_room_metadata(
        '{"tenantId":"550e8400-e29b-41d4-a716-446655440000","phoneNumber":"+201001112222"}'
    )

    assert metadata == RoomMetadata(
        tenant_id="550e8400-e29b-41d4-a716-446655440000",
        phone_number="+201001112222",
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
