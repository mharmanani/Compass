from setuptools import setup, find_packages

setup(
    name='Compass',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch', 'torchvision', 'numpy', 'omegaconf', 'einops', 'tqdm',
        'h5py', 'Pillow', 'scikit-learn', 'scikit-image', 'pydantic',
        'traitlets', 'pandas', 'wandb', 'transformers',
    ],
)
