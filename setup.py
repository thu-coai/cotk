from setuptools import setup, find_packages

setup(
    name='cotk',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Coversational Processing Toolkits',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url='https://github.com/',
    author='cotk-THU',
    author_email='cotk@example.com'
)
