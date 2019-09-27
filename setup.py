from setuptools import setup

setup(
    name='Chrononet',
    version='0.1dev',
    packages=['chrononet',],
    author='Serena Jeblee',
    license='MIT license',
    long_description=open('README.md').read(),
    install_requires=[
                        'networkx', 'numpy' #'learning2rank',
                    ],
    #dependency_links=['http://github.com/sjeblee/learning2rank/tarball/master#egg=package-1.0']
)
