from setuptools import setup, find_packages

setup(
    name='lamorel',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="0.1",
    install_requires=[
        'transformers',
        'accelerate',
        'hydra-core',
        'torch>=1.8.1'
    ],
    description="",
    author=""
)