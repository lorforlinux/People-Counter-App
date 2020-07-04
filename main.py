"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def get_bounding_box(img, output, conf_level=0.35):
    height = img.shape[0]
    width = img.shape[1]
    box = output[0,0,:,3:7] * np.array([width, height, width, height])
    box = box.astype(np.int32)
    cls = output[0,0,:,1]
    conf = output[0,0,:,2]
    count=0
    p1 = None
    p2 = None
    for i in range(len(box)):
        if (not int(cls[i])==1) or conf[i]<conf_level:
            continue
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(img, p1, p2, (0,255,0))
        count+=1
    return img, count, (p1,p2)

def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    if args.input=="CAM":
        camera = cv2.VideoCapture(0)
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        infer_network.load_model(args.model, 1, args.device, args.cpu_extension)
        input_shape = infer_network.get_input_shape()
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        resized_frame = cv2.resize(img, (input_shape[3], input_shape[2]))
        frame_preproc = np.transpose(np.expand_dims(resized_frame.copy(), axis=0), (0,3,1,2))
        infer_network.exec_net(frame_preproc)
        if infer_network.wait()==0:
            outputs = infer_network.get_output()
            box_frame, count, bbox = get_bounding_box(img, outputs, prob_threshold)
            cv2.putText(box_frame, "Count:"+str(count), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imwrite('output.jpg', box_frame)
        return
    else:
        if not os.path.isfile(args.input):
            exit(1)
        camera = cv2.VideoCapture(args.input)
    if (camera.isOpened()== False): 
        exit(1)
    cur_req_id=0
    next_req_id=1
    num_requests=2
    infer_network.load_model(args.model, num_requests, args.device, args.cpu_extension)
    input_shape = infer_network.get_input_shape()
    ret, frame = camera.read()
    total_count=0
    pres_count = 0
    prev_count=0
    start_time=0 
    no_bbox=0
    duration=0
    prev_bbox_x = 0

    while camera.isOpened():
        ret, next_frame = camera.read()
        if not ret:
            break
        key = cv2.waitKey(60)
        resized_frame = cv2.resize(next_frame.copy(), (input_shape[3], input_shape[2]))
        frame_preproc = np.transpose(np.expand_dims(resized_frame.copy(), axis=0), (0,3,1,2))
        infer_network.exec_net(frame_preproc.copy(), req_id=next_req_id)
        if infer_network.wait(cur_req_id)==0:
            outputs = infer_network.get_output(cur_req_id)
            frame, pres_count, bbox = get_bounding_box(frame.copy(), outputs[0], prob_threshold)
            box_w = frame.shape[1]
            topleft, bottomright = bbox
        
            if pres_count>prev_count:
                start_time = time.time()
                total_count+=pres_count-prev_count
                no_bbox=0
                client.publish("person", json.dumps({"total":total_count}))
            elif pres_count<prev_count:
                if no_bbox<=20:
                    pres_count=prev_count
                    no_bbox+=1
                elif prev_bbox_x<box_w-200:
                    pres_count=prev_count
                    no_bbox=0
                else:
                    duration = int(time.time()-start_time)
                    client.publish("person/duration", json.dumps({"duration":duration}))
            if not (topleft==None and bottomright==None):
                prev_bbox_x=int((topleft[0]+bottomright[0])/2)
            prev_count=pres_count
                    
            client.publish("person", json.dumps({"count":pres_count}))
            if key==27:
                break

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        cur_req_id, next_req_id = next_req_id, cur_req_id
        frame = next_frame

    #output_video.release()
    camera.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()