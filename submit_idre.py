import sys
import os
import pickle
import time

# file_name = str(sys.argv[1])
# job_type = str(sys.argv[2])

compute_chunk = int(sys.argv[1])

software = sys.argv[2] #'galkin', 'jampy'][1]
ani_model = sys.argv[3] #['om', 'constant', 'step'][0]
slit = sys.argv[4] #['single_slit', 'ifu'][1]
spherical = str(sys.argv[5])

if spherical == 'True':
    spherical = True
else:
    spherical = False

job_name_list = input
num_compute = 21000 #125000
for i in range(int(num_compute/compute_chunk)):
    start_index = i*compute_chunk
    os.system('./idre_submit_job.sh '+str(start_index)+' '+str(compute_chunk)+' '+str(software)+' '+str(ani_model)+' '+str(slit)+' '+str(spherical))
    print('./idre_submit_job.sh '+str(start_index)+' '+str(compute_chunk)+' '+str(software)+' '+str(ani_model)+' '+str(slit)+' '+str(spherical))
    time.sleep(1)

print(i+1, 'jobs submitted!')
