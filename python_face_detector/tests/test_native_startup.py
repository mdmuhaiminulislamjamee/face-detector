"""Fresh-process checks for native import order and readable startup failures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeStartupTests(unittest.TestCase):
    def test_limits_apply_before_native_imports_even_with_conflicting_environment(self):
        code = '''
import importlib.abc
import json
import os
import sys

class CheckImportEnvironment(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ('cv2', 'numpy'):
            assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
            assert os.environ['OMP_NUM_THREADS'] == '1'
            assert os.environ['OPENCV_FOR_THREADS_NUM'] == '2'

sys.meta_path.insert(0, CheckImportEnvironment())
from native_runtime import load_native_libraries
cv2, np = load_native_libraries()
assert np.dot(np.ones((32, 32)), np.ones((32, 32)))[0, 0] == 32
print(json.dumps({'opencv_threads': cv2.getNumThreads()}))
'''
        env = dict(os.environ, OPENBLAS_NUM_THREADS='64', OMP_NUM_THREADS='64',
                   OPENCV_FOR_THREADS_NUM='64')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['opencv_threads'], 2)

    def test_paging_file_import_error_exits_cleanly_without_a_traceback(self):
        code = '''
import importlib.abc
import importlib.machinery
import runpy
import sys

class PagingFileFailure(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'cv2':
            return importlib.machinery.ModuleSpec(fullname, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise ImportError('DLL load failed: The paging file is too small')

sys.meta_path.insert(0, PagingFileFailure())
runpy.run_path('live_face_detector.py', run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn('paging file', result.stderr)
        self.assertIn('committed-memory capacity', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
