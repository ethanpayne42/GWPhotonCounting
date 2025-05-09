import os
import sys
import numpy as np

# set up list for strings to be written to file and adding the first check submit file
# This job is simply so that we can ensure the first job succeeds
dag_lines = [f"JOB first /home/ethan.payne/projects/calibration_marginalization/event_set_submission/first_check_submit.sub"]

launch_string = 'PARENT first CHILD '


for index in range(0,100):
    for snr in np.logspace(np.log10(0.02), np.log10(6), 30):
                
        dag_lines += ['\n',
            f"JOB {index}_{snr} /home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/snr_CI_distribution/injection_submit.sub",
            f"VARS {index}_{snr} "\
            f"snr=\"{snr}\" "\
            f"index=\"{index}\" "]

        launch_string += f"{index}_{snr} "

dag_lines.append(launch_string)

# Write out to file
with open(f'injection_submission.dag', 'w+') as filehandle:
    for listitem in dag_lines:
        filehandle.write('%s\n' % listitem)
