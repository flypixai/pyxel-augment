from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pyaugment",
    version="0.1",
    description="A package for automated object based data augmentation",
    author="Syrine Khammari",
    author_email="skhammari@aisupeiror.com",
    packages=["pyaugment"],
    install_requires=requirements,
)
