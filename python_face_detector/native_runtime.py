"""Bound native thread pools before OpenCV/NumPy allocate their runtime state."""
import os


NATIVE_THREAD_LIMITS = {
    'OPENBLAS_NUM_THREADS': '1',
    'GOTO_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'OPENCV_FOR_THREADS_NUM': '2',
}


def load_native_libraries():
    # These are process-local limits. They also apply when the Python script is
    # launched directly, without the batch runner or an activated environment.
    os.environ.update(NATIVE_THREAD_LIMITS)
    import cv2
    import numpy as np

    cv2.setNumThreads(int(NATIVE_THREAD_LIMITS['OPENCV_FOR_THREADS_NUM']))
    return cv2, np


def startup_error_message(error):
    message = f'[ERROR] Could not load OpenCV/NumPy: {error}'
    if isinstance(error, MemoryError) or any(word in str(error).lower()
            for word in ('paging file', 'pagefile', 'memory allocation', 'not enough memory')):
        message += ('\n[ERROR] Windows has insufficient committed-memory capacity '
                    '(RAM plus paging file). Close unnecessary apps or enable a '
                    'system-managed paging file, then retry. The app does not '
                    'change Windows memory settings.')
    return message
