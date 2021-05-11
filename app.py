import time

import cv2
import edgeiq
import numpy as np 

difficulty = 'easy'

tracker = edgeiq.KalmanTracker(deregister_frames=4, max_distance=100)
# tracker = edgeiq.CentroidTracker(deregister_frames=4, max_distance=100)

def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/yolo_v3")
    obj_detect.load(engine=edgeiq.Engine.DNN)
 
    video_path = f"data/inputs/{difficulty}.mp4"
    stream_context = edgeiq.FileVideoStream(f"{video_path}", play_realtime=True)

    with stream_context as video_stream, edgeiq.Streamer() as streamer:
        while video_stream.more():
            
            image = video_stream.read()
            results = obj_detect.detect_objects(image, confidence_level=.5)
            specific_predictions = [r for r in results.predictions if r.label == 'person']
            
            res = tracker.update(specific_predictions)

            image = draw_tracked_boxes(image, res)
            # image = edgeiq.markup_image(image, people_predictions)

            streamer.send_data(image)


def draw_tracked_boxes(
    frame,
    objects,
    line_color=None,
    line_width=None,
    id_size=None,
    id_thickness=None,
    draw_box=True,
):
    frame_scale = frame.shape[0] / 100
    if line_width is None:
        line_width = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    color_is_None = line_color == None

    for obj_id, pred in objects.items():
        line_color = Color.random(obj_id)

        points = np.array([[pred.box.start_x, pred.box.start_y], 
                            [pred.box.end_x, pred.box.end_y]])
        points = points.astype(int)
        cv2.rectangle(
            frame,
            tuple(points[0, :]),
            tuple(points[1, :]),
            color=line_color,
            thickness=line_width,
        )

        if id_size > 0:
            id_draw_position = np.mean(points, axis=0)
            id_draw_position = id_draw_position.astype(int)
            cv2.putText(
                frame,
                str(obj_id),
                tuple(id_draw_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                id_size,
                line_color,
                id_thickness,
                cv2.LINE_AA,
            )
    return frame


class Color:
    green = (0, 128, 0)
    white = (255, 255, 255)
    olive = (0, 128, 128)
    black = (0, 0, 0)
    navy = (128, 0, 0)
    red = (0, 0, 255)
    maroon = (0, 0, 128)
    grey = (128, 128, 128)
    purple = (128, 0, 128)
    yellow = (0, 255, 255)
    lime = (0, 255, 0)
    fuchsia = (255, 0, 255)
    aqua = (255, 255, 0)
    blue = (255, 0, 0)
    teal = (128, 128, 0)
    silver = (192, 192, 192)

    @staticmethod
    def random(obj_id: int):
        color_list = [
            c
            for c in Color.__dict__.keys()
            if c[:2] != "__"
            and c not in ("random", "red", "white", "grey", "black", "silver")
        ]
        return getattr(Color, color_list[obj_id % len(color_list)])

if __name__ == "__main__":
    main()
