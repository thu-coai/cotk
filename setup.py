'''
setup.py for contk
'''
import sys
from setuptools import setup, find_packages

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(
	name='contk',
	version='0.0.1',
	packages=find_packages(exclude=[]),
	license='Apache',
	description='Coversational Toolkits',
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
	setup_requires=[] + pytest_runner,
    tests_require=[
		"pytest",
		"sphinx",
		"pytest-cov==2.4.0",
		"python-coveralls",
		"pytest-dependency",
		"pytest-mock",
		"requests-mock"
	],
	include_package_data=True,
	url='https://github.com/thu-coai/contk',
	author='thu-coai',
	author_email='thu-coai-developer@googlegroups.com',
	python_requires='>=3.5',
	zip_safe=False
)
