import sys,os,json,re


gold = []
f = open('./LC-QuAD2.0/dataset/test.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: int(x['uid']))

for item in d:
    wikisparql = item['sparql_wikidata']
    unit = {}
    unit['uid'] = item['uid']
    unit['question'] = item['question']
    _ents = re.findall( r'wd:(.*?) ', wikisparql)
    _rels = re.findall( r'wdt:(.*?) ',wikisparql)
    unit['entities'] = [ent for ent in _ents]
    unit['relations'] = [rel for rel in _rels]
    gold.append(unit)

f = open('lcqout.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: int(x[0]))

tpentity = 0
fpentity = 0
fnentity = 0
tprelation = 0
fprelation = 0
fnrelation = 0
totalentchunks = 0
totalrelchunks = 0
mrrent = 0
mrrrel = 0
chunkingerror = 0
for queryitem,golditem in zip(d,gold):
    if queryitem[0] != golditem['uid']:
        print('uid mismatch')
        sys.exit(1)
    queryentities = []
    if 'entities' in queryitem[1]:
        if len(queryitem[1]['entities']) > 0:
            for k,v in queryitem[1]['entities'].iteritems():
                queryentities.append(v[0][0])
    print(set(golditem['entities']),set(queryentities), golditem['question'])
    for goldentity in set(golditem['entities']):
        totalentchunks += 1
        if goldentity in queryentities:
            tpentity += 1
        else:
            fnentity += 1
    for queryentity in set(queryentities):
        if queryentity not in golditem['entities']:
            fpentity += 1

precisionentity = tpentity/float(tpentity+fpentity)
recallentity = tpentity/float(tpentity+fnentity)
f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
print("precision entity = ",precisionentity)
print("recall entity = ",recallentity)
print("f1 entity = ",f1entity)
