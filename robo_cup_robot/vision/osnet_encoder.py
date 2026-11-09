"""CPU ONNX appearance encoder for the bundled OSNet ReID model."""

import numpy as np


class OsnetEncoder:
    def __init__(self, model_path, threads=2):
        import cv2
        import onnxruntime as ort

        self.cv2 = cv2
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError("OSNet model must have exactly one input")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] != 3:
            raise RuntimeError(f"unexpected OSNet input shape: {shape}")
        self.input = inputs[0]
        self.batch = int(shape[0]) if isinstance(shape[0], int) else None
        self.height = int(shape[2])
        self.width = int(shape[3])
        self.mean = np.asarray([.485, .456, .406], np.float32).reshape(1, 1, 3)
        self.std = np.asarray([.229, .224, .225], np.float32).reshape(1, 1, 3)

    def _prepare(self, crop):
        if crop is None or crop.ndim != 3 or crop.shape[2] != 3 or not crop.size:
            raise ValueError("invalid person crop")
        if crop.shape[1] < 35 or crop.shape[0] < 65:
            raise ValueError("person crop is too small for reliable appearance encoding")
        rgb = self.cv2.cvtColor(
            self.cv2.resize(crop, (self.width, self.height)), self.cv2.COLOR_BGR2RGB
        )
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return normalized.transpose(2, 0, 1)

    def encode_batch(self, crops):
        if not crops:
            return []
        prepared = [self._prepare(crop) for crop in crops]
        batch_size = self.batch or len(prepared)
        embeddings = []
        for start in range(0, len(prepared), batch_size):
            part = prepared[start:start + batch_size]
            batch = np.stack(part).astype(np.float32)
            if self.batch is not None and len(part) < self.batch:
                batch = np.concatenate(
                    [batch, np.repeat(batch[-1:], self.batch - len(part), axis=0)]
                )
            output = np.asarray(
                self.session.run(None, {self.input.name: batch})[0], dtype=np.float32
            )
            if output.ndim != 2 or output.shape[0] != len(batch):
                raise RuntimeError(f"unexpected OSNet output shape: {output.shape}")
            for vector in output[:len(part)]:
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or norm < 1e-8:
                    raise RuntimeError("OSNet returned an invalid appearance vector")
                embeddings.append(vector / norm)
        return embeddings
