# src/data_analysis_crew/utils/instructions.py


INSTALL_LIB_TEMPLATE = (
    "You're in a Docker/script environment (not Jupyter). If a package is missing, use this install pattern:\n\n"
    "```python\n"
    "try:\n"
    "    import PACKAGE_NAME\n"
    "except ImportError:\n"
    "    import subprocess\n"
    "    subprocess.check_call(['pip', 'install', 'PACKAGE_NAME'])\n"
    "```\n\n"
    "✅ Example:\n"
    "```python\n"
    "try:\n"
    "    import pandas as pd\n"
    "    import numpy as np\n"
    "    import matplotlib.pyplot as plt\n"
    "    import seaborn as sns\n"
    "    import scipy as sp\n"
    "    import sklearn\n"
    "    import tensorflow as tf\n"
    "    import pytorch as torch\n"
    "except ImportError:\n"
    "    import subprocess\n"
    "    subprocess.check_call(['pip', 'install', 'pandas'])\n"
    "    subprocess.check_call(['pip', 'install', 'numpy'])\n"
    "    subprocess.check_call(['pip', 'install', 'matplotlib'])\n"
    "    subprocess.check_call(['pip', 'install', 'seaborn'])\n"
    "    subprocess.check_call(['pip', 'install', 'scipy'])\n"
    "    subprocess.check_call(['pip', 'install', 'sklearn'])\n"
    "    subprocess.check_call(['pip', 'install', 'tensorflow'])\n"
    "    subprocess.check_call(['pip', 'install', 'torch'])\n"
    "    subprocess.check_call(['pip', 'install', 'torch', 'torchvision'])\n"
    "```\n\n"
    "**DO NOT** use `!pip install`. It only works in notebooks."
)

AVAILABLE_LIBRARIES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scipy",
    "scikit-learn",
    "tensorflow",
    "torch",
    "torchvision",
    "shap"
]
