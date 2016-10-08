from setuptools import setup

setup(name='pipemodels',
      version='0.1',
      description='''
            Tiny framework for machine learning
            algorithms evaluation and comparison
      ''',
      url='http://github.com/denmoroz/pipemodels',
      author='Dzianis Dus',
      author_email='dzianisdus@gmail.com',
      license='MIT',
      packages=['pipemodels'],
      install_requires=[
          'luigi',
          'yaml'
      ],
      zip_safe=False)
