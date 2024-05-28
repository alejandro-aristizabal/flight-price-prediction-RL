from setuptools import setup, find_packages

setup(
    name='flight-price-qlearning',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.1',
        'numpy==1.22.3',
        'scikit-learn==1.0.2',
        'flask==2.0.3',
    ],
    entry_points={
        'console_scripts': [
            'flight-price-qlearning=main:main',
        ],
    },
)
