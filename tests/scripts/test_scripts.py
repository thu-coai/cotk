import os
import pytest
from cotk.scripts import report
import re
import json
import subprocess
from subprocess import PIPE

class TestScripts():
	def test_download_git_repo(self):
		os.system("pip install -r ./tests/scripts/requirements.txt")
		with pytest.raises(ValueError) as excinfo:
			report.dispatch('download', ['--zip_url', 'http://wrong_url'])
		assert "Invalid zip url." in str(excinfo.value)

		with pytest.raises(RuntimeError) as excinfo:
			report.dispatch('download', ['--zip_url', 'http://github.com/user/repo/archive/wrong_commit.zip'])
		assert "It should be public." in str(excinfo.value)

		with pytest.raises(FileNotFoundError) as excinfo:
			report.dispatch('download',\
							['--zip_url', 'https://github.com/ZhihongShao/CVAE_test/archive/no_result_file.zip'])
		assert re.search(r"Config file .* is not found", str(excinfo.value))

		with pytest.raises(json.JSONDecodeError) as excinfo:
			report.dispatch('download', \
							['--zip_url', 'https://github.com/ZhihongShao/CVAE_test/archive/invalid_json.zip'])

		with pytest.raises(RuntimeError) as excinfo:
			report.dispatch('download', \
							['--zip_url', 'https://github.com/ZhihongShao/CVAE_test/archive/keys_undefined.zip'])
		assert 'Undefined keys in `config.json`: working_dir, entry, args' == str(excinfo.value)

		with pytest.raises(FileNotFoundError) as excinfo:
			report.dispatch('download', \
							['--zip_url', 'https://github.com/ZhihongShao/CVAE_test/archive/no_output.zip'])
		assert "New result file not found." == str(excinfo.value)
		report.dispatch('download', ['--zip_url', 'https://github.com/ZhihongShao/CVAE_test/archive/run_and_test.zip'])
