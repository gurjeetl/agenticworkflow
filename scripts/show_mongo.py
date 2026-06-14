"""Pretty-print the agent_memory MongoDB contents from the command line.

Usage (from the project root):
    .venv\\Scripts\\python.exe scripts\\show_mongo.py                  # overview: every collection + counts
    .venv\\Scripts\\python.exe scripts\\show_mongo.py agent_registry   # dump one collection
    .venv\\Scripts\\python.exe scripts\\show_mongo.py agent_facts 5    # dump first 5 docs of a collection

Connection + DB name come from the same env vars the app uses, falling back to
local defaults so it works even without a .env loaded.
"""
import os
import sys

from bson.json_util import dumps
from pymongo import MongoClient

uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
dbname = os.getenv("MONGODB_DB", "agent_memory")
db = MongoClient(uri)[dbname]

args = sys.argv[1:]
print(f"# {uri} / {dbname}\n")

if not args:
    names = db.list_collection_names()
    if not names:
        print("(no collections)")
    for c in sorted(names):
        print(f"{c:24} {db[c].count_documents({}):>6} docs")
    print("\nTip: pass a collection name to dump its documents, e.g. "
          "scripts\\show_mongo.py agent_registry")
else:
    coll = args[0]
    limit = int(args[1]) if len(args) > 1 else 0
    cursor = db[coll].find()
    if limit:
        cursor = cursor.limit(limit)
    docs = list(cursor)
    print(f"{coll}: {len(docs)} document(s) shown\n")
    print(dumps(docs, indent=2))
