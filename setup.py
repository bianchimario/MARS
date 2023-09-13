from setuptools import setup, find_packages

setup(
    name='MARS',
    version='1.0.0',
    description='Multivariate Asynchronous Random Shapelets (MARS)',
    author='Mario Bianchi',
    author_email='bianchi.mario@outlook.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'awkward'
    ]
)
