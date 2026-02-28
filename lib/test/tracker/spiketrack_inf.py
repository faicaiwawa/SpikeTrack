from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.seqtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh

from lib.models.spiketrack.spiketrack_inf import build_spiketrack
from lib.test.tracker.seqtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.test.utils.hann import hann2d
import numpy as np
import torch
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SpikeTrack(BaseTracker):
    def __init__(self, params, dataset_name, checkpoint_path):
        super(SpikeTrack, self).__init__(params)
        network, encoder_temp = build_spiketrack(params.cfg)
        print("run at inference mode  !!!!!!!!!!!!!")
        res_1 = network.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['net'], strict=False)
        state_dict = torch.load(checkpoint_path, map_location='cpu')['net']
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_k = k.replace('encoder.', '', 1)  # remove encoder_
                encoder_state_dict[new_k] = v
        res_2 = encoder_temp.load_state_dict(encoder_state_dict, strict=False)

        self.cfg = params.cfg
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.network = network.cuda()
        self.encoder_temp = encoder_temp.cuda()
        self.network.eval()
        self.encoder_temp.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = params.debug
        self.frame_id = 0

        if 'lasot' in dataset_name:
            self.update_threshold = 0.8
            self.update_intervals = 40
        else:    
            self.update_threshold = 0.7
            self.update_intervals = 25
        print("Update threshold is: ", self.update_threshold)
        print("Update interval is: ", self.update_intervals)

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // 16
        self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()
        self.cache = None
        self.spike_rate_dict_temp = None
        self.window_penalty = True

    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        self.state = info['init_bbox']
        self.frame_id = 0
        template = torch.stack(self.template_list, dim=0)
        with torch.no_grad():
            self.cache, self.spike_rate_dict_temp = self.encoder_temp.forward(template)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        # run the encoder
        with torch.no_grad():
            xz = self.network.forward_encoder(search, self.cache)
        # run the decoder
        with torch.no_grad():
            out_dict, spike_rate_dict = self.network.inference_decoder(xz=xz)

        pred_score_map = out_dict['score_map']
        if self.window_penalty: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                                   out_dict['offset_map'],
                                                                   return_score=True)

        pred_boxes = pred_boxes.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result

        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):        

                    z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)
                    template = self.preprocessor.process(z_patch_arr)

                    if len(self.template_list) >= self.num_template:
                        self.template_list.pop(1)
                    self.template_list.append(template)

                    with torch.no_grad():
                        template = torch.stack(self.template_list, dim=0)
                        self.cache, self.spike_rate_dict_temp = self.encoder_temp.forward(template)

        # for debug
        if self.debug == 1:
            screen_width, screen_height = 1600, 800
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            img_height, img_width = image_BGR.shape[:2]
            scale_w = screen_width / img_width
            scale_h = screen_height / img_height
            scale = min(scale_w, scale_h, 1.0)

            if scale < 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                image_BGR = cv2.resize(image_BGR, (new_width, new_height), interpolation=cv2.INTER_AREA)

            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state}, spike_rate_dict, self.spike_rate_dict_temp

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return SpikeTrack

