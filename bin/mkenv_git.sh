# the following assumes we have a C compiler that is used for compiling the 
# Cythonized extension modules for Python, on Windows one needs MSVC for CPython
# from python.org
# see also:
# https://github.com/cython/cython/wiki/CythonExtensionsOnWindows#using-windows-sdk-cc-compiler-works-for-all-python-versions

cd /tmp
python3 -m venv env3
source /tmp/env3/bin/activate

pip install Cython
pip install psycopg2
pip install six

# install packages in edit mode (so you can change the source code)
# they end up in the virtual environment
# under /tmp/env3/src/<packagename>
pip install -e git+https://github.com/bmmeijers/sink/#egg=sink
pip install -e git+https://github.com/bmmeijers/simplegeom/#egg=simplegeom
pip install -e git+https://github.com/bmmeijers/connection/#egg=connection
pip install -e git+https://github.com/bmmeijers/predicates/#egg=geompreds
pip install -e git+https://github.com/bmmeijers/tri/#egg=tri
pip install -e git+https://github.com/bmmeijers/splitarea/#egg=splitarea
pip install -e git+https://github.com/bmmeijers/topomap/#egg=topomap
pip install -e git+https://github.com/bmmeijers/quadtree/#egg=quadtree
pip install -e git+https://github.com/bmmeijers/oseq/#egg=oseq
pip install -e git+https://github.com/bmmeijers/grassfire/#egg=grassfire
pip install -e git+https://github.com/bmmeijers/tgap-ng/#egg=tgap_ng

# replace DBUSER, DBPASS, DB, DBHOST with 
# right connection parameters for postgres
echo "[database]
username=DBUSER
password=DBPASS
database=DB
host=DBHOST
port=5432
sslmode=prefer" > /tmp/env3/src/connection/src/connection/config/default.ini

# which ini file contains the connection authentication?
# if the environment variable is absent, the default.ini file will be read
# export DBCONFIG=pakhuis.ini

# work on the environment
# on windows this might be: .\tmp\env3\Scripts\activate.bat
# see: https://docs.python.org/3/library/venv.html 
source /tmp/env3/bin/activate

# run the tgap build, edit the parameters in there to obtain a tGAP for a different dataset
tgap-ng.py
