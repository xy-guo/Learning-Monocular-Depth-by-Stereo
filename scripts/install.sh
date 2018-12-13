#!/bin/bash
cd ./func/correlation1d_package
python setup.py install --user
cd ../resample1d_package
python setup.py install --user
cd ..