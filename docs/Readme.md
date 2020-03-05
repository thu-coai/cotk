# CoTK Docs 

## TL;DR

* For user, run `make public` in this dir. It generates docs for user.
* For developer, run `make internal` in this dir. It generate docs for developer, with a full internal api in CoTK.
* The index of docs will be placed in `./build/html/index.html`

## Introduction

CoTK docs are built in two steps:

* Preprocess by meta processor (`./meta/update_doc.py`). Convert meta doc(`./meta`) into source(`./source`).
* Build html(`./build/html/index`) by sphinx.

### Preprocess

* If you want to generate docs for user, run `cd meta && python update_doc.py -D public`.
* If you want to generate docs for developer, run `cd meta && python update_doc.py -D internal`.

The commands will update `./source` and `../Readme.md`.

### Build Html

* Install sphinx 2.4.4
* Run `make clean`. (optional, see below)
* Run `make html`.

The command will update `./build/html/`

#### Why `make clean`

The files only affects model zoo. If you do not care about them, you do not need to run it.

In the process, the scripts may download files from github. If the files are downloaded, they won't be updated until deleted. `make clean` will force the scripts download the files again.
