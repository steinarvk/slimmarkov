from setuptools import setup

setup(
  name='slimmarkov',
  version='0.1.2',
  packages=['slimmarkov', 'slimmarkov.proto'],
  zip_safe=False,
  install_requires=[
    "Unidecode>=0.04.18",
    "argparse>=1.2.1",
    "humanize>=0.5.1",
    "markovify>=0.1.0",
    "numpy>=1.9.2",
    "protobuf>=2.6.1",
    "py>=1.4.30",
    "pytest>=2.7.2",
    "wsgiref>=0.1.2",
  ],
)
