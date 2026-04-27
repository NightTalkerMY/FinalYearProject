import cv2
import numpy as np
import os

class OpenCVFaceDetector:
    """
    Drop-in replacement for S3FD that uses OpenCV's DNN face detector (runs on CPU).
    Returns detections in the same [x1, y1, x2, y2, confidence] format as S3FD.
    """

    def __init__(self):
        model_dir = os.path.dirname(__file__)
        prototxt = os.path.join(model_dir, "deploy.prototxt")
        caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.isfile(prototxt) or not os.path.isfile(caffemodel):
            raise FileNotFoundError(
                f"OpenCV face detector weights not found in {model_dir}.\n"
                "Download them:\n"
                "  deploy.prototxt: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt\n"
                "  res10_300x300_ssd_iter_140000.caffemodel: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            )

        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        # Force CPU backend — the whole point of this swap
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[FaceDetector] OpenCV DNN loaded (CPU-only)")

    def detect_faces(self, image, conf_th=0.8, scales=None):
        """
        Same interface as S3FD.detect_faces().
        `image` is RGB (H, W, 3). `scales` is ignored (not needed for the SSD model).
        Returns np.ndarray of shape (N, 5) with [x1, y1, x2, y2, confidence].
        """
        h, w = image.shape[:2]

        # OpenCV DNN expects BGR, but blobFromImage handles mean subtraction internally
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()  # shape: [1, 1, N, 7]

        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < conf_th:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bboxes.append([x1, y1, x2, y2, float(confidence)])

        if len(bboxes) == 0:
            return np.empty(shape=(0, 5))

        return np.array(bboxes)
