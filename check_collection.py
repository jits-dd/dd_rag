from pymilvus import connections, Collection
from pymilvus import utility
#
connections.connect(host='localhost', port='19530', user='root', password='Milvus')
collection = Collection('conversational_rag')
collection.load()
#
# print(f"\nCollection stats:")
# print(f"Entities: {collection.num_entities}")
# print(f"Schema: {collection.schema}")
# print(f"Indexes: {collection.indexes}")

#
# Query with all fields specified
# results = collection.query(
#     expr="",  # Empty expression to get all
#     output_fields=["id", "doc_id", "text","metadata"],
#     limit=10
# )

# dynamic_results = collection.query(
#     expr="",
#     output_fields=["*"],  # Get all fields including dynamic ones
#     limit=10
# )

# print(f"\nQuery results: {results}")
#
# print(f"\nQuery results: {dynamic_results}")

# print("\nSample documents:")
# for i, doc in enumerate(results):
#     print(f"{i+1}. ID: {doc['id']}")
#     print(f"   doc_id: {doc['doc_id'][:50]}...")
#
# # ---------------------------------------


# Delete existing collection if it exists
# if "conversational_rag" in utility.list_collections():
#     utility.drop_collection("conversational_rag")


# -- Milvus databse data

"""
from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Load your collection
collection = Collection("conversational_rag")
collection.load()

# Check basic stats
print(f"Number of entities: {collection.num_entities}")

# Query some sample records
results = collection.query(
    expr="",
    output_fields=["text", "metadata"],
    limit=3
)

print("\nSample documents:")
for i, doc in enumerate(results):
    print(f"\nDocument {i+1}:")
    print(f"Text: {doc['text'][:200]}...")
    print(f"Metadata: {doc['metadata']}")
"""

# -- verify retrieval

col = Collection("conversational_rag")
col.load()
print("Entity count:", col.num_entities)

# Sample query
results = col.query(expr='metadata["company"] == "Fusion Food and Beverage Dynamics"',
                    output_fields=["text"])
print("Fusion Food documents:", results)