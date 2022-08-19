
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from nasbench import api

from nasbench.lib import model_builder
import time
import json

import datetime
import os


import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--test', action='store_true', default=False, help='test if it works')
parser.add_argument('--data', type=str, default='../nasbench_data/nasbench_full.tfrecord', help='location of the tfrecord')
parser.add_argument('--device', type=str, default='cuda110@2080ti', help='name of the device')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--ckpt', type=str, default=None, help='the checkpoint file already searched')


args = parser.parse_args()

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


if "cuda" in args.device:
    print("testing cuda device")
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

NASBENCH_TFRECORD = args.data
DEVICE = args.device
BATCH_SIZE = args.batch_size
LOOP_NUM = 10 # how many runs for one network




def get_latency(model_spec):
    config = api.config.build_config()
    net_fn = model_builder.build_simple_model_fn(spec=model_spec, config=config)

    latency = []

    with tf.Graph().as_default() as graph:

        # create graph from spec
        input = tf.compat.v1.placeholder(shape = [BATCH_SIZE ,32 ,32 ,3] ,dtype=tf.float32, name="input") # NHWC format
        net = net_fn(features = input, mode = tf.estimator.ModeKeys.PREDICT )

        data = np.random.uniform(0, 1, [BATCH_SIZE, 32, 32, 3])

        # print(net)
        with tf.compat.v1.Session() as session:
            init = tf.compat.v1.global_variables_initializer()
            session.run(init)
            # note that there is overhead to run the network
            # meaning the first run takes much more time
            res = session.run(net, feed_dict={input: data})
            for i in range(LOOP_NUM):
                start_time = time.time()
                # Get predict result of the graph.
                res = session.run(net, feed_dict={input : data})
                end_time = time.time()
                # print(res)
                duration = (end_time-start_time)*1000
                latency .append (duration) # at ms

    return latency


def main():
    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d-%H-%M-%S")

    os.mkdir("./{}".format(date))
    config = {}
    config["input_size"] = [BATCH_SIZE ,32 ,32 ,3]
    config["device"] = DEVICE
    config["runs"] = LOOP_NUM

    latency_result = {}
    if args.ckpt:
        config["check_point"] = args.ckpt
        with open(args.ckpt) as f:
            latency_result = json.load(f)


    with open('./{}/configs.json'.format(date), 'w') as f:
        json.dump(config, f)  # 编码JSON数据

    nasbench_api = api.NASBench(NASBENCH_TFRECORD)

    count = 0

    WARM_UP()
    for h in nasbench_api.hash_iterator():
        count += 1
        if h in latency_result:
            continue

        print("Running on #", count, "#")
        fixed, computed = nasbench_api.get_metrics_from_hash(h)

        model_spec = api.ModelSpec(
            # Adjacency matrix of the module
            matrix=fixed["module_adjacency"],
            # Operations at the vertices of the module, matches order of matrix
            ops=fixed["module_operations"])

        res = get_latency(model_spec)

        latency_result[h] = res
        print(h, "------>", res)
        if count % 5000 == 0:
            with open('./{}/hash_latency_{}.json'.format(date, str(count)), 'w') as f:
                json.dump(latency_result, f)  # 编码JSON数据

    with open('./{}/hash_latency.json'.format(date), 'w') as f:
        json.dump(latency_result, f)  # 编码JSON数据



def WARM_UP():
    model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=[[0, 1, 1, 1, 0, 1, 0],  # input layer
                [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
                [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
                [0, 0, 0, 0, 0, 0, 0]],  # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])
    config = api.config.build_config()
    net_fn = model_builder.build_simple_model_fn(spec=model_spec, config=config)

    data = np.random.uniform(0, 1, [64, 32, 32, 3])
    latency = 0

    with tf.Graph().as_default() as graph:

        # create graph from spec
        input = tf.compat.v1.placeholder(shape = [64 ,32 ,32 ,3] ,dtype=tf.float32, name="input") # NHWC format
        net = net_fn(features = input, mode = tf.estimator.ModeKeys.PREDICT )

        # print(net)
        with tf.compat.v1.Session() as session:
            init = tf.compat.v1.global_variables_initializer()
            session.run(init)

            for i in range(500):
                # Get predict result of the graph.
                res = session.run(net, feed_dict={input : data})


def test():
    WARM_UP()
    # Create an Inception-like module (5x5 convolution replaced with two 3x3
    # convolutions).
    model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=[[0, 1, 1, 1, 0, 1, 0],  # input layer
                [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
                [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
                [0, 0, 0, 0, 0, 0, 0]],  # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

    res = get_latency(model_spec)
    print(res)

if __name__ == '__main__':
    if args.test:
        test()
    else:
        main()




















