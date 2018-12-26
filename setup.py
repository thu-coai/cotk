'''
setup.py for contk
'''
from setuptools import setup, find_packages

setup(
	name='contk',
	version='0.0.1',
	packages=find_packages(exclude=['tests*']),
	license='MIT',
	description='Coversational Toolkits',
	long_description=open('README.md', encoding='UTF-8').read(),
	install_requires=[
		'numpy>=1.13',
		'nltk>=3.2'
	],
	url='https://github.com/thu-coai/contk',
	author='thu-coai',
	author_email='thu-coai-developer@googlegroups.com'
)
