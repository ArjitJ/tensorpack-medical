from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("common",  ["common.py"]),
    Extension("dataReader",  ["dataReader.py"]),
    Extension("DQN",  ["DQN.py"]),
    Extension("DQNModel",  ["DQNModel.py"]),
    Extension("expreplay",  ["expreplay.py"]),
    Extension("medical",  ["medical.py"]),
    Extension("viewer",  ["viewer.py"])
]
setup(
    name = 'landmark',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
