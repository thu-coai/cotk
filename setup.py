'''
setup.py for cotk
'''
import sys
import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

class LibTest(TestCommand):
	def run_tests(self):
		# import here, cause outside the eggs aren't loaded
		ret = os.system("pytest --cov=cotk tests/ --cov-report term-missing")
		sys.exit(ret >> 8)

setup(
	name='cotk',
	version='0.0.2',
	packages=find_packages(exclude=[]),
	license='Apache',
	description='Conversational Toolkits',
	long_description=open('README.md', encoding='UTF-8').read(),
	long_description_content_type="text/markdown",
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'License :: OSI Approved :: Apache Software License',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
	],
	install_requires=[
		'numpy>=1.13',
		'nltk>=3.4',
		'tqdm>=4.30',
		'checksumdir>=1.1',
		'requests'
	],
	extras_require={
		'develop':  [
			"torch>=1.0.0",
			"python-coveralls",
			"pytest-dependency",
			"pytest-mock",
			"requests-mock",
			"pytest>=3.6.0",
			"pytest-cov==2.4.0",
			"coverage<5.0",
			"checksumdir",
			"transformers"
		]
	},
	cmdclass={'test': LibTest},
	entry_points={
		'console_scripts': [
			"cotk=cotk.scripts.main:main",
		]
	},
	include_package_data=True,
	url='https://github.com/thu-coai/cotk',
	author='thu-coai',
	author_email='thu-coai-developer@googlegroups.com',
	python_requires='>=3.5',
	zip_safe=False
)
