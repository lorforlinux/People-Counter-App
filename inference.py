#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = IECore()
        self.exec_network = None
        self.net = None
        self.input_layer = None

    def load_model(self, xml_path, num_requests, device_name, cpu_extension):
        self.net = IENetwork(model=xml_path, weights=xml_path.split('.')[0]+'.bin')
        supported_layers = self.plugin.query_network(network = self.net, device_name=device_name)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0:
            if not cpu_extension==None:
                self.plugin.add_extension(cpu_extension, device_name)
                supported_layers = self.plugin.query_network(network = self.net, device_name=device_name)
                unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    exit(1)
            else:
                exit(1)
        self.exec_network = self.plugin.load_network(network=self.net, num_requests=num_requests, device_name=device_name)
        self.input_layer = next(iter(self.net.inputs))
        return self.exec_network, self.input_layer

    def get_input_shape(self):
        return self.net.inputs[self.input_layer].shape

    def exec_net(self, frame, req_id):
        self.exec_network.start_async(request_id=req_id, inputs={self.input_layer:frame})
        return self.exec_network

    def wait(self, reqest_id):
        wait_process = self.exec_network.requests[reqest_id].wait(-1)
        return wait_process

    def get_output(self, req_id):
        output = self.exec_network.requests[req_id].outputs
        return [output[i] for i in output.keys()]