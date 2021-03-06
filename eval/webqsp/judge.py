import sys,os,json,re


gold = []
f = open('input/webqsp.test.entities.with_classes.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: x['question_id'])

for item in d:
    unit = {}
    unit['uid'] = item['question_id']
    unit['question'] = item['utterance']
    unit['entities'] = item['entities']
    unit['all'] = item
    gold.append(unit)

f = open('webqtestout.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: x[0])

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
        print(queryitem[0], golditem['uid'])
        print('uid mismatch')
        sys.exit(1)
    queryentities = []
    if 'entities' in queryitem[1]:
        if len(queryitem[1]['entities']) > 0:
            for k,v in queryitem[1]['entities'].iteritems():
                queryentities.append(v[0][0])
    print(set(golditem['entities']),set(queryentities), golditem['question'])
    if None in set(golditem['entities']):
        print('skip none')
        continue
    for goldentity in set(golditem['entities']):
        if goldentity == None:
            print("skip none")
            continue
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
