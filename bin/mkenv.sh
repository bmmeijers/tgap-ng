#import os

cd /tmp
python3 -m venv env3
source /tmp/env3/bin/activate

pip install Cython
pip install psycopg2
pip install six

cd /home/martijn/workspace/sink
pip install -e .

cd /home/martijn/workspace/simplegeom
pip install -e .

cd /home/martijn/workspace/connection
pip install -e .

cd /home/martijn/workspace/predicates
pip install -e .

cd /home/martijn/workspace/tri
pip install -e .

cd /home/martijn/workspace/splitarea
pip install -e .

cd /home/martijn/workspace/topomap
pip install -e .

cd /home/martijn/workspace/quadtree
pip install -e .

cd /home/martijn/workspace/oseq
pip install -e .

cd /home/martijn/workspace/grassfire
pip install -e .

cd /home/martijn/Documents/work/2020-07_tgap-ng/tgap-ng
pip install -e .
