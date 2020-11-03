import sys,os,json,re


gold = []
f = open('annotated_wd_data_test.txt')
for line in f.readlines():
    line = line.strip()
    s,p,o,q = line.split('\t')
    gold.append(s)


f = open('simpleqtestout.json')
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
    if len(queryitem[1]) == 0:
        continue
#    if queryitem[0] != golditem['uid']:
#        print('uid mismatch')
#        sys.exit(1)
    queryentities = []
    if len(queryitem[1]['entities']) > 0:
        for k,v in queryitem[1]['entities'].iteritems():
            queryentities.append(v[0][0])
    print(golditem,set(queryentities))
    for goldentity in [golditem]:
        totalentchunks += 1
        if goldentity in queryentities:
            tpentity += 1
        else:
            fnentity += 1
    for queryentity in set(queryentities):
        if queryentity not in [golditem]:
            fpentity += 1
precisionentity = tpentity/float(tpentity+fpentity)
recallentity = tpentity/float(tpentity+fnentity)
f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
print("precision entity = ",precisionentity)
print("recall entity = ",recallentity)
print("f1 entity = ",f1entity)
