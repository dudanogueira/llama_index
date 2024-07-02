from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from weaviate.util import generate_uuid5
import pytest

try:
    import weaviate

    # Start an embedded Weaviate
    # client = weaviate.connect_to_embedded(version="latest")
    # this tests will assume there is a default Weaviate available
    client = weaviate.connect_to_local()
    weaviate_available = client.is_connected()
    client.close()
except Exception:
    weaviate_available = False


def test_class():
    names_of_base_classes = [b.__name__ for b in WeaviateVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.mark.skipif(
    not weaviate_available, reason="Weaviate Embedded or Docker not available"
)
def test_multitenant_collection_creation():
    client.connect()
    # client = weaviate.connect_to_embedded(version="latest")
    client.collections.delete("LlamaIndexMT")
    # create the collection, specifying the tenant
    WeaviateVectorStore(weaviate_client=client, index_name="LlamaIndexMT", tenant="T1")
    collection = client.collections.get("LlamaIndexMT")
    assert collection.config.get().multi_tenancy_config.enabled == True


@pytest.mark.skipif(
    not weaviate_available, reason="Weaviate Embedded or Docker not available"
)
def test_multitenant_data_insertion():
    client.connect()
    # client = weaviate.connect_to_embedded(version="latest")
    client.collections.delete("LlamaIndexMT")
    # T1
    vector_store_t1 = WeaviateVectorStore(
        weaviate_client=client, index_name="LlamaIndexMT", tenant="T1"
    )
    storage_context_t1 = StorageContext.from_defaults(vector_store=vector_store_t1)
    index_t1 = VectorStoreIndex.from_documents(
        [Document(text="Content for T1", doc_id=generate_uuid5("tenant1"))],
        storage_context=storage_context_t1,
    )
    # T2
    vector_store_t2 = WeaviateVectorStore(
        weaviate_client=client, index_name="LlamaIndexMT", tenant="T2"
    )
    storage_context_t2 = StorageContext.from_defaults(vector_store=vector_store_t2)
    index_t2 = VectorStoreIndex.from_documents(
        [Document(text="Content for T2", doc_id=generate_uuid5("tenant2"))],
        storage_context=storage_context_t2,
    )
    # check TOTAL for T1
    collection = client.collections.get("LlamaIndexMT").with_tenant("T1")
    total = collection.aggregate.over_all(total_count=True)
    assert total.total_count == 1
    # check content for object in T1 directly
    assert (
        collection.query.fetch_objects(limit=1).objects[0].properties.get("text")
        == "Content for T1"
    )
    # check content for object in T1 directly
    assert (
        collection.with_tenant("T2")
        .query.fetch_objects(limit=1)
        .objects[0]
        .properties.get("text")
        == "Content for T2"
    )
    # finally, test retriever from llama-index on different tenants
    retriever = index_t1.as_retriever()
    response = retriever.retrieve("Content")
    assert response[0].text == "Content for T1"
    retriever = index_t2.as_retriever()
    response = retriever.retrieve("Content")
    assert response[0].text == "Content for T2"
