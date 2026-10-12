from setuptools import setup, find_packages

setup(
      name='proteinfoundation',
      version='1.0.0',
      description='proteina',
      packages=find_packages(),
      package_dir={
          'proteinfoundation': './proteinfoundation',
          'openfold': './openfold',
          'graphein_utils': './graphein_utils'
      }
)