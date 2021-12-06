import tensorflow as tf
import numpy as np
import pointer_net
import time
import os
import random
import sys
import json
import glob
from copy import deepcopy

tf.app.flags.DEFINE_integer("batch_size", 32,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 3000, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 100, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 512, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 1, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
tf.app.flags.DEFINE_string("models_dir", "", "Log directory")
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


  def run(self):
    self.build_model()
    self.testall()

  def getvector(self,d):
    inputs = []
    enc_input_weights = []
    dec_input_weights = []
    maxlen = 0
    self.testoutputs = []
    for question in d:
      questioninputs = []
      enc_input_len = len(question[2])
      #print(enc_input_len)
      if enc_input_len > FLAGS.max_input_sequence_len:
        print("Length too long, skip")
        continue
      for idx,word in enumerate(question[2]):
        questioninputs.append(word[0])
      for i in range(FLAGS.max_input_sequence_len-enc_input_len):
        questioninputs.append([0]*500)
    self.testoutputs.append(question[1])
    weight = np.zeros(FLAGS.max_input_sequence_len)
    weight[:enc_input_len]=1
    enc_input_weights.append(weight)
    inputs.append(questioninputs)
    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)

  def calculatef1(self, batchd, predictions, tp,fp,fn):
    citems = []
    for inputquestion,prediction,groundtruth in zip(batchd, predictions, self.testoutputs):
      idtoentity = {}
      predents = set()
      gtents = groundtruth
      #print(len(self.test_inputs))
      for entnum in list(prediction[0]):
        if entnum <= 0:
          continue
        predents.add(inputquestion[2][entnum-1][1])
      #print(inputquestion[3],gtents,predents)
      citem = deepcopy(inputquestion[3])
      citem['goldentrel'] = list(gtents)
      citem['predentrel'] = list(predents)
      citems.append(citem)
      for goldentity in gtents:
        #totalentchunks += 1
        if goldentity in predents:
          tp += 1
        else:
          fn += 1
      for queryentity in predents:
        if queryentity not in gtents:
          fp += 1
    try:
      precisionentity = tp/float(tp+fp)
      recallentity = tp/float(tp+fn)
      f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
      #print("precision entity = ",precisionentity)
      #print("recall entity = ",recallentity)
      #print("f1 entity = ",f1entity)
    except Exception as e:
      #print(e)
      pass
    return tp,fp,fn,citems
    #print(tp,fp,fn)

  def testall(self):
    print("Test set evaluation running ...")
#    ckpt = tf.train.get_checkpoint_state(FLAGS.models_dir)
#    print(ckpt, FLAGS.models_dir)
#    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#      print("Load model parameters from %s" % ckpt.model_checkpoint_path)
#      self.testmodel.saver.restore(self.testsess, ckpt.model_checkpoint_path)
    tp = 0
    fp = 0
    fn = 0
    linecount = 0
    batchd = []
    mastercitems = []
    with open(FLAGS.test_data_path) as rfp:
      for line in rfp:
        linecount += 1
        line = line.strip()
        print(linecount)
        d = json.loads(line)
        if len(d) > FLAGS.max_input_sequence_len:
          print("Skip question, too long")
          continue
  #      #print(len(d))
        batchd.append(d)
        #print(linecount)
        try:
          self.getvector(batchd)
          predicted,_ = self.testmodel.step(self.testsess, self.test_inputs, self.test_enc_input_weights, update=False)
          _tp,_fp,_fn,citems = self.calculatef1(batchd,predicted,tp,fp,fn)
          for citem in citems:
            mastercitems.append(citem)
          #print("mastercitems:",mastercitems)
          batchd = []
        except Exception as e:
          print(e)
          batchd = []
          continue
        tp = _tp
        fp = _fp
        fn = _fn
    precisionentity = tp/float(tp+fp+0.001)
    recallentity = tp/float(tp+fn+0.001)
    f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity+0.001)
    print("precision  %f  recall %f  f1 %f  "%(precisionentity, recallentity, f1entity))
    f = open(FLAGS.pred_out_path,'w')
    f.write(json.dumps(mastercitems,indent=4))
    f.close()

def main(_):
  entitylinker = EntityLinker(FLAGS.forward_only)
  entitylinker.run()

if __name__ == "__main__":
  tf.app.run()
