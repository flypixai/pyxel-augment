from setuptools import setup
  
setup(
    name='pyaugment',
    version='0.1',
    description='A package for automated object based data augmentation',
    author='Syrine Khammari',
    author_email='skhammari@aisupeiror.com',
    packages=['pyaugment'],
    install_requires=[
        'numpy',
        'pandas',
    ],
)
