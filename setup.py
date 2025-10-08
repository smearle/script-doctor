from setuptools import setup, find_packages

setup(
    name='puzzlejax',
    version='0.1.0',
    packages=['puzzlejax','script_doctor'],
    package_data={
        "puzzlejax" : ["syntax.lark"],
        "script_doctor": ["*.txt"],  # relative to the package dir
    },
    install_requires=[

        'beautifulsoup4==4.13.5',
        'chex==0.1.90',
        'einops==0.8.1',
        'flask==3.1.2',
        'flax==0.11.2',
        'gym==0.26.2',
        'gymnax==0.0.9',
        'hydra-core==1.3.2',
        'hydra-submitit-launcher==1.2.0',
        'imageio==2.37.0',
        'javascript==1!1.2.5',
        # jax[cuda]==0.7.1  # or `jax` if no CUDA is not an option
        'jax>=0.7.1',
        'lark==1.2.2',
        'Levenshtein==0.27.1',
        'matplotlib==3.10.6',
        'openai==1.107.1',
        'opencv-python==4.11.0.86',
        'optax==0.2.5',
        'orbax-checkpoint==0.11.25',
        'pandas==2.3.2',
        'pathvalidate==3.3.1',
        'Pillow==11.3.0',
        'py-cpuinfo==9.0.0',
        'python-dotenv==1.1.1',
        'scikit-image==0.25.2',
        'selenium==4.34.1',
        'submitit==1.5.3',
        'tiktoken==0.11.0',
        'wandb==0.21.3',
        # web-browser
        'webdriver-manager==4.0.2'

    ],
    description='PuzzleJAX package',
    author='anon',
    author_email='anon',
    python_requires='>=3.6',
)
