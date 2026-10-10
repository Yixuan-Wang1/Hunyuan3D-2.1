"""Repository-root compatibility entry point for :mod:`hy3dshape`.

The distributable Python package lives in ``hy3dshape/hy3dshape``. Scripts
that run from this subproject already see that directory as the package root,
while scripts launched from the repository root see this outer directory
first. Add the real package directory to this package's search path so both
launch modes support the canonical ``hy3dshape.*`` imports used by the source
code and model configuration files.
"""

from pathlib import Path


_SOURCE_PACKAGE = str(Path(__file__).resolve().parent / "hy3dshape")
if _SOURCE_PACKAGE not in __path__:
    __path__.append(_SOURCE_PACKAGE)


from .pipelines import Hunyuan3DDiTPipeline, Hunyuan3DDiTFlowMatchingPipeline
from .postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier
from .preprocessors import ImageProcessorV2, IMAGE_PROCESSORS, DEFAULT_IMAGEPROCESSOR


del Path, _SOURCE_PACKAGE
