import sys
from elasticsearch import Elasticsearch
from elasticsearch import helpers

doccount = 0
actions = []
es = Elasticsearch()
with open("../data/wikidata_translation_v1.tsv") as infile:
    for line in infile:
        items = line.strip().split('\t')
        key = items[0]
        vector = items[1:]
        if '/Q' in key or '/P' in key or '@en' in key:
            action = { "_index": "wikidataembedsindex01", "_type": "records", "_source": { "key": key, "embedding": vector } }
            actions.append(action)
        if len(actions) == 100000:
            print("indexing 100k docs ....")
            helpers.bulk(es, actions)
            doccount += 100000
            print("%d done"%(doccount))
            actions = []
helpers.bulk(es, actions)
print("All %d done"%(doccount + len(actions)))
