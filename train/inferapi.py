import tensorflow as tf
import copy
import numpy as np
import pointer_net
from flask import request, Response
from flask import Flask
from gevent.pywsgi import WSGIServer
import time
import os
import random
import sys
import json
import glob
from copy import deepcopy

tf.app.flags.DEFINE_integer("batch_size", 1,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 3000, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 100, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 512, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 1, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", True, "Forward Only.")
tf.app.flags.DEFINE_string("models_dir", "../models/model2/solid/", "Log directory")
tf.app.flags.DEFINE_string("data_path", "", "Training Data path.")
tf.app.flags.DEFINE_string("test_data_path", "", "Test Data path.")
tf.app.flags.DEFINE_string("pred_out_path", "", "Test Data path.")

FLAGS = tf.app.flags.FLAGS

class EntityLinker(object):
    def __init__(self, forward_only):
        self.forward_only = True
        self.epoch = 0
        self.bestf1 = 0
        self.testgraph = tf.Graph()
        with self.testgraph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.operation_timeout_in_ms=6000
            self.testsess = tf.Session(config=config)
            self.build_model()
  
    def build_model(self):
        self.testmodel = None
        with self.testgraph.as_default():
            self.testmodel = pointer_net.PointerNet(batch_size=1,
                    max_input_sequence_len=FLAGS.max_input_sequence_len,
                    max_output_sequence_len=FLAGS.max_output_sequence_len,
                    rnn_size=FLAGS.rnn_size,
                    attention_size=FLAGS.attention_size,
                    num_layers=FLAGS.num_layers,
                    beam_width=FLAGS.beam_width,
                    learning_rate=FLAGS.learning_rate,
                    max_gradient_norm=FLAGS.max_gradient_norm,
                    forward_only=True)
            self.testsess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(FLAGS.models_dir)
            print(ckpt, FLAGS.models_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Load model parameters from %s" % ckpt.model_checkpoint_path)
                self.testmodel.saver.restore(self.testsess, ckpt.model_checkpoint_path)

    def getvector(self,d):
        inputs = []
        enc_input_weights = []
        dec_input_weights = []
        maxlen = 0
        self.testoutputs = []
        for question in d:
            questioninputs = []
            enc_input_len = len(question)
            if enc_input_len > FLAGS.max_input_sequence_len:
                print("Length too long, skip")
                continue
            for idx,word in enumerate(question):
                questioninputs.append(word[0])
        for i in range(FLAGS.max_input_sequence_len-enc_input_len):
            questioninputs.append([0]*500)
        weight = np.zeros(FLAGS.max_input_sequence_len)
        weight[:enc_input_len]=1
        enc_input_weights.append(weight)
        inputs.append(questioninputs)
        self.test_inputs = np.stack(inputs)
        self.test_enc_input_weights = np.stack(enc_input_weights)

linker = EntityLinker(FLAGS.forward_only)
app = Flask(__name__)
print("serving api ...")

@app.route('/erlinker', methods=['POST'])
def erlinker():
    d = request.get_json(silent=True)
    citem = copy.deepcopy(d)
    inputvecs = []
    linkedvecs = []
    sepseen = False
    for cans,canv in zip(d['candidatestring'],d['candidatevectors']):
        inputvecs.append([canv,cans])
        if cans == '[SEP]':
            sepseen = True
            continue
        if not sepseen:
            linkedvecs.append([canv,cans])
    linkedvecs.append([500*[-1.0],'[SEP]']) # This for now only holds thhe question token + [SEP]. The linked ents and rels will be appended shortly.

    batch = [inputvecs]
    vector = linker.getvector(batch)
    predicted,_ = linker.testmodel.step(linker.testsess, linker.test_inputs, linker.test_enc_input_weights, update=False)
    predentrels = set()
    for entrelnum in list(predicted[0][0]):
        if entrelnum <= 0:
            continue
        predentrels.add(inputvecs[entrelnum-1][1]) 
        linkedvecs.append([inputvecs[entrelnum-1][0],inputvecs[entrelnum-1][1]])
    print(predentrels)
    citem['predentrels'] = list(predentrels)
    del citem['candidatevectors']
    citem['linkedentrelvecs'] = linkedvecs
    citem['linkedentrelstring'] = [x[1] for x in linkedvecs]
    return json.dumps(citem, indent=4, sort_keys=True)

if __name__ == '__main__':
    http_server = WSGIServer(('', 2223), app)
    http_server.serve_forever()
