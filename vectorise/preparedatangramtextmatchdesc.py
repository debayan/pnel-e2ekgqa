from __future__ import division
import sys
import json
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz
import requests
from annoy import AnnoyIndex
import re,random
from nltk.util import ngrams
from textblob import TextBlob
import urllib2
from multiprocessing import Pool
import redis
import random

redisdb = redis.Redis()

def mean(a):
    return sum(a) / len(a)

postags = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
es = Elasticsearch()

writef = open(sys.argv[3], 'w') 

def getdescription(entid):
    res = es.search(index="wikidataentitydescriptionsindex01", body={"query":{"term":{"entityid.keyword":entid}}})
    try:
        description = res['hits']['hits'][0]['_source']['description']
        return description
    except Exception as e:
        return ''
cache = {}

def gettextembedding(text):
    if text in cache:
        return cache[text]
    try:
        req = urllib2.Request('http://localhost:8887/ftwv')
        req.add_header('Content-Type', 'application/json')
        inputjson = {'chunks':[text]}
        response = urllib2.urlopen(req, json.dumps(inputjson))
        labelembedding = json.loads(response.read().decode('utf8'))[0]
        cache[text] = labelembedding
        return labelembedding
    except Exception as e:
        #print("getdescriptionsembedding err: ",e)
        return [0]*300
    return [0]*300

def getembedding(enturl):
    entityurl = '<http://www.wikidata.org/entity/'+enturl[37:]+'>'
    res = es.search(index="wikidataembedsindex01", body={"query":{"term":{"key":{"value":entityurl}}}})
    try:
        embedding = [float(x) for x in res['hits']['hits'][0]['_source']['embedding']]
        return embedding
    except Exception as e:
        #print(enturl,' not found')
        return None
    return None

def gettextmatchmetric(label,word):
    return [fuzz.ratio(label,word)/100.0,fuzz.partial_ratio(label,word)/100.0,fuzz.token_sort_ratio(label,word)/100.0] 

fail = 0
def givewordvectors(inputtuple):#(id,question,entities):
    id = inputtuple[0]
    question = inputtuple[1]
    entities = inputtuple[2]
    if not question:
        return []
    q = question
    q = re.sub("\s*\?", "", q.strip())
    pos = TextBlob(q)
    chunks = pos.tags
    candidatevectors = []
    #questionembedding
    tokens = [token[0] for token in chunks]
    r = requests.post("http://localhost:8887/ftwv",json={'chunks': tokens})
    questionembeddings = r.json()
    print(question,len(questionembeddings))
    questionembedding = list(map(lambda x: sum(x)/len(x), zip(*questionembeddings)))
    true = []
    false = []
    for idx,chunk in enumerate(chunks):
        #n
        word = chunk[0]
        tokenembedding = questionembeddings[idx]
        posonehot = len(postags)*[0.0]
        posonehot[postags.index(chunk[1])] = 1
        esresult = es.search(index="wikidataentitylabelindex01", body={"query":{"multi_match":{"query":chunks[idx][0]}},"size":30})
        esresults = esresult['hits']['hits']
        if len(esresults) > 0:
            for entidx,esresult in enumerate(esresults):
                entityembedding = getembedding(esresult['_source']['uri'])
                desc = getdescription(esresult['_source']['uri'][37:])
                descembedding = gettextembedding(desc)
                label = esresult['_source']['wikidataLabel']
                textmatchmetric = gettextmatchmetric(label, word)
                if entityembedding and questionembedding :
                    if esresult['_source']['uri'][37:] in entities:
                        candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,1],esresult['_source']['uri'][37:],1.0])
                    else:
                        candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,1],esresult['_source']['uri'][37:],0.0])
        #n-1,n
        if idx > 0:
             word = chunks[idx-1][0]+' '+chunks[idx][0]
             esresult = es.search(index="wikidataentitylabelindex01", body={"query":{"multi_match":{"query":word}},"size":30})
             esresults = esresult['hits']['hits']
             if len(esresults) > 0:
                 for entidx,esresult in enumerate(esresults):
                     entityembedding = getembedding(esresult['_source']['uri'])
                     label = esresult['_source']['wikidataLabel']
                     desc = getdescription(esresult['_source']['uri'][37:])
                     descembedding = gettextembedding(desc)
                     textmatchmetric = gettextmatchmetric(label, word)
                     if entityembedding and questionembedding :
                         if esresult['_source']['uri'][37:] in entities:
                             candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,-2],esresult['_source']['uri'][37:],1.0])
                         else:
                             candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,-2],esresult['_source']['uri'][37:],0.0])
        #n,n+1
        if idx < len(chunks) - 1:
            word = chunks[idx][0]+' '+chunks[idx+1][0]
            esresult = es.search(index="wikidataentitylabelindex01", body={"query":{"multi_match":{"query":word}},"size":30})
            esresults = esresult['hits']['hits']
            if len(esresults) > 0:
                for entidx,esresult in enumerate(esresults):
                    entityembedding = getembedding(esresult['_source']['uri'])
                    label = esresult['_source']['wikidataLabel']
                    desc = getdescription(esresult['_source']['uri'][37:])
                    descembedding = gettextembedding(desc)
                    textmatchmetric = gettextmatchmetric(label, word)
                    if entityembedding and questionembedding :
                        if esresult['_source']['uri'][37:] in entities:
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,2],esresult['_source']['uri'][37:],1.0])
                        else:
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,2],esresult['_source']['uri'][37:],0.0])
        #n-1,n,n+1
        if idx < len(chunks) - 1 and idx > 0:
            word = chunks[idx-1][0]+' '+chunks[idx][0]+' '+chunks[idx+1][0]
            esresult = es.search(index="wikidataentitylabelindex01", body={"query":{"multi_match":{"query":word}},"size":30})
            esresults = esresult['hits']['hits']
            if len(esresults) > 0:
                for entidx,esresult in enumerate(esresults):
                    entityembedding = getembedding(esresult['_source']['uri'])
                    label = esresult['_source']['wikidataLabel']
                    desc = getdescription(esresult['_source']['uri'][37:])
                    descembedding = gettextembedding(desc)
                    textmatchmetric = gettextmatchmetric(label, word)
                    if entityembedding and questionembedding :
                        if esresult['_source']['uri'][37:] in entities:
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,3],esresult['_source']['uri'][37:],1.0])
                        else:
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,3],esresult['_source']['uri'][37:],0.0])
    print("len ",len(candidatevectors[0][0])) 
    writef.write(json.dumps([id,item['entities'],candidatevectors])+'\n')
    #return (id,entities,candidatevectors)

d = json.loads(open(sys.argv[1]).read())
random.shuffle(d)
labelledcandidates = []
inputcandidates = []
for idx,item in enumerate(d):
    if item['source'] != sys.argv[2]:
        continue
    print(idx,item['question'])
    inputcandidates.append((item['id'],item['question'],item['entities']))
    candidatevectors = givewordvectors(item['id'],item['question'],item['entities'])
    print(len(candidatevectors))
#pool = Pool(10)
#responses = pool.imap(givewordvectors,inputcandidates)
#count = 0
#redisdb.delete(sys.argv[3])
#for response in responses:
#    print("count = ",count)
#    count += 1
#    redisdb.rpush(sys.argv[3],json.dumps([response[0],response[1],response[2]])+'\n')

