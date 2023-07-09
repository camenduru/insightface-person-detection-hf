#!/usr/bin/env python

from __future__ import annotations

import functools
import pathlib

import cv2
import gradio as gr
import huggingface_hub
import insightface
import numpy as np
import onnxruntime as ort

TITLE = 'insightface Person Detection'
DESCRIPTION = 'https://github.com/deepinsight/insightface/tree/master/examples/person_detection'


def load_model():
    path = huggingface_hub.hf_hub_download('public-data/insightface',
                                           'models/scrfd_person_2.5g.onnx')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 8
    session = ort.InferenceSession(
        path,
        sess_options=options,
        providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    model = insightface.model_zoo.retinaface.RetinaFace(model_file=path,
                                                        session=session)
    return model


def detect_person(
    img: np.ndarray, detector: insightface.model_zoo.retinaface.RetinaFace
) -> tuple[np.ndarray, np.ndarray]:
    bboxes, kpss = detector.detect(img)
    bboxes = np.round(bboxes[:, :4]).astype(int)
    kpss = np.round(kpss).astype(int)
    kpss[:, :, 0] = np.clip(kpss[:, :, 0], 0, img.shape[1])
    kpss[:, :, 1] = np.clip(kpss[:, :, 1], 0, img.shape[0])
    vbboxes = bboxes.copy()
    vbboxes[:, 0] = kpss[:, 0, 0]
    vbboxes[:, 1] = kpss[:, 0, 1]
    vbboxes[:, 2] = kpss[:, 4, 0]
    vbboxes[:, 3] = kpss[:, 4, 1]
    return bboxes, vbboxes


def visualize(image: np.ndarray, bboxes: np.ndarray,
              vbboxes: np.ndarray) -> np.ndarray:
    res = image.copy()
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        vbbox = vbboxes[i]
        x1, y1, x2, y2 = bbox
        vx1, vy1, vx2, vy2 = vbbox
        cv2.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), 1)
        alpha = 0.8
        color = (255, 0, 0)
        for c in range(3):
            res[vy1:vy2, vx1:vx2,
                c] = res[vy1:vy2, vx1:vx2,
                         c] * alpha + color[c] * (1.0 - alpha)
        cv2.circle(res, (vx1, vy1), 1, color, 2)
        cv2.circle(res, (vx1, vy2), 1, color, 2)
        cv2.circle(res, (vx2, vy1), 1, color, 2)
        cv2.circle(res, (vx2, vy2), 1, color, 2)
    return res


def detect(image: np.ndarray, detector) -> np.ndarray:
    image = image[:, :, ::-1]  # RGB -> BGR
    bboxes, vbboxes = detect_person(image, detector)
    res = visualize(image, bboxes, vbboxes)
    return res[:, :, ::-1]  # BGR -> RGB


detector = load_model()
detector.prepare(-1, nms_thresh=0.5, input_size=(640, 640))
fn = functools.partial(detect, detector=detector)

image_dir = pathlib.Path('images')
examples = [[path.as_posix()] for path in sorted(image_dir.glob('*.jpg'))]

gr.Interface(
    fn=fn,
    inputs=gr.Image(label='Input', type='numpy'),
    outputs=gr.Image(label='Output', height=600),
    examples=examples,
    examples_per_page=30,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch()
