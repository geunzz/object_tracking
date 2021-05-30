import time
import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
from IOU_tracker import IOU_tracker

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'C:/projects/object_tracking/model/face_test.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/projects/object_tracking/model/face_test.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return (boxes, scores, classes, num_detections)

if __name__ == "__main__":

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)
    cap = cv2.VideoCapture(0)
    windowNotSet = True
    tracker = IOU_tracker(categories)

    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [height, width] = image.shape[:2]
        image = cv2.flip(image, 1)

        (boxes, scores, classes, num_detections) = tDetector.run(image)
        image = tracker.tracker_run(image, height, width, boxes, scores, classes)
    
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
