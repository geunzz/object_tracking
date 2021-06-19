import numpy as np
import cv2

class IOU_tracker():
    def __init__(self, categories, min_score_thres = 0.5, draw_length = 10, frame_counter_thd = 1):
        self.boxes_draw_bef = []
        self.color_draw_bef = []
        self.classes_draw_bef = []
        self.color_draw = []
        self.start_trigger = True
        self.min_score_thres = min_score_thres
        self.draw_length = draw_length + 1
        self.categories = categories
        self.frame_counter = 0
        self.frame_counter_thd = frame_counter_thd


    def tracker_run(self, image, height, width, boxes, scores, classes):
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        boxes_draw = []
        score_draw = np.array([])
        classes_draw = np.array([])
        
        for i in range(len(boxes)):
            #표출되는 box의 개수를 제한
            if (scores[i] > self.min_score_thres) and (len(boxes_draw) < self.draw_length):
                boxes_draw.append(boxes[i])
                score_draw = np.append(score_draw, scores[i])
                classes_draw = np.append(classes_draw, classes[i])
            else:
                break

        if self.start_trigger == True:
            self.boxes_draw_bef = boxes_draw
            self.classes_draw_bef = classes_draw
            self.start_trigger = False
            
        boxes_draw = np.array(boxes_draw)
        self.color_draw = []

        for c in range(len(boxes_draw)):
            color_add = np.random.choice(range(256), size=3)
            self.color_draw.append(color_add)
        self.color_draw = np.array(self.color_draw)
        if len(self.color_draw_bef) == 0:
            self.color_draw_bef = []
            for c in range(len(self.boxes_draw_bef)):
                color_add = np.random.choice(range(256), size=3)
                self.color_draw_bef.append(color_add)
            self.color_draw_bef = np.array(self.color_draw_bef)

        for j in range(len(boxes_draw)):

            IOU_max = 0
            IOU_now = 0
            for k in range(len(self.boxes_draw_bef)):
                IOU_now = (min(self.boxes_draw_bef[k][3],boxes_draw[j][3]) - max(self.boxes_draw_bef[k][1],boxes_draw[j][1]))*\
                (min(self.boxes_draw_bef[k][2],boxes_draw[j][2]) - max(self.boxes_draw_bef[k][0],boxes_draw[j][0]))/\
                ((self.boxes_draw_bef[k][2]-self.boxes_draw_bef[k][0])*(self.boxes_draw_bef[k][3]-self.boxes_draw_bef[k][1]))

                if (IOU_max < IOU_now) and (IOU_now > 0.5):
                    IOU_max = IOU_now
                    max_color = self.color_draw_bef[k]

            if IOU_max == 0:
                max_color = np.random.choice(range(256), size=3)
                for color in self.color_draw:
                    #랜덤으로 생성된 색깔이 하나라도 겹치면 색깔을 다시 생성
                    while any(color == max_color):
                            max_color = np.random.choice(range(256), size=3)

            self.color_draw[j] = max_color
            
            x_top = int(boxes_draw[j][1]*width)
            y_top = int(boxes_draw[j][0]*height)
            x_bottom = int(boxes_draw[j][3]*width)
            y_bottom = int(boxes_draw[j][2]*height)

            r = int(self.color_draw[j][0])
            g = int(self.color_draw[j][1])
            b = int(self.color_draw[j][2])

            image = cv2.rectangle(image, (x_top, y_top), (x_bottom, y_bottom), (r,g,b), 2)
            image = cv2.putText(image, str(self.categories[int(classes_draw[j])-1]['name']), (x_top, int(y_top-10)), cv2.FONT_HERSHEY_SIMPLEX, 1,(r,g,b),2)
        
        if self.frame_counter == 0:
            self.color_draw_bef = self.color_draw
            self.boxes_draw_bef = boxes_draw
            self.classes_draw_bef = classes_draw

        if self.frame_counter == self.frame_counter_thd:
            self.frame_counter = 0
        else:
            self.frame_counter = self.frame_counter + 1
        cv2.imshow("tracking", image)

        return image
