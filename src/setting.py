import numpy as np

#LENGTH_LIMIT = 2  # [cm]
POWER_LIMIT = int(np.ceil(np.log2(LENGTH_LIMIT)))
LENGTH_PRECISION = 5


PCD_MODELS = {'Kzaa': {2019: 'Kza5m2019Cld.pcd', 2020: 'Kza5m2020Cld.pcd'},
              'C2'  : {2019: 'C2_19.pcd',        2020: 'C2_20.pcd'},
              'Cube': {1000: 'Cube.pcd'}}

MESH_MODELS = {'Kzaa': {2019: 'Kza5m2019.ply',2020: 'Kza5m2020.ply', 2022: 'Kza22_registered.ply'},
               'C1':   {2019: 'C12019.ply',    2020: 'C12020.ply', 2022: 'C1_2022_registered.ply'},
               'C2':   {2019: 'C22019.ply',    2020: 'C22020.ply', 2022: 'C2_2022_registered.ply'},
               #'C3':   {2019: 'C32019.ply', 2020: 'C32020.ply', 2022: 'C3_2022_registered.ply'},#,    2020: 'C32020.ply', 2022: 'C3_2022.ply'
               'C3':   { '2019': 'C32019.ply'},
               'C4':   {2019: 'C42019.ply', 2020: 'C42020.ply', 2022: 'C4_2022_registered.ply'},
               'C5':   {2019: 'C52019.ply',    2020: 'C52020.ply',2022: 'C5_2022.ply'},
               'NR1':  {2019: 'NR12019Reg.ply',   2020: 'NR12020Reg.ply', 2022: 'NR12022_registered.ply'},
                       }
