#!/usr/bin/python

from __future__ import print_function
import sys,json,urllib, urllib2, requests
import requests,re
from multiprocessing import Pool
import time

def hiturl(questionserial):
    question = questionserial[0]
    serial = questionserial[1]['question_id']
    req = urllib2.Request('http://localhost:4444/processQuery')
    req.add_header('Content-Type', 'application/json')
    try:
        print(question)
        question = re.sub(r"[^a-zA-Z0-9]+", ' ', question)
        print(question)
        inputjson = {'nlquery': question}
        start = time.time()
        response = urllib2.urlopen(req, json.dumps(inputjson))
        end = time.time()
        response = response.read()
        return (serial,response,questionserial[1],end-start,len(question.split(' ')))
    except Exception,e:
        return(serial,'[]',questionserial[1],0,0)

f = open('input/webqsp.test.entities.with_classes.json')
s = f.read()
d = json.loads(s)
f.close()
questions = []

for item in d:
    questions.append((item['utterance'],item))

pool = Pool(1)
responses = pool.imap(hiturl,questions)

_results = []

count = 0
totalentchunks = 0
tpentity = 0
fpentity = 0
fnentity = 0
for response in responses:
    count += 1
    print(count)
    _results.append((response[0],json.loads(response[1]),response[3],response[4]))


results = []
for result in _results:
    results.append(result)

 
f1 = open('webqtestout.json','w')
print(json.dumps(results),file=f1)
f1.close()
