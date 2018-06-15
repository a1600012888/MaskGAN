import os

base_command = 'rlaunch --cpu=8 --gpu=1 --memory=8192 -- python3 train.py --adv=0 --exp='

def run_tv_loss():

    tv_Weights = [5e-2, 5e-3, 1e-3, 1e-4]

    for tv in tv_Weights:

        command = base_command + 'noadv_tv_{}'.format(tv) + ' --tv={} &'.format(tv)

        os.system(command)


def run_content_loss():
    content_weights = [5.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2]

    for c in content_weights:

        command = base_command + 'noadv_content_{}'.format(c) + ' --content={} &'.format(c)

        os.system(command)

if __name__ == '__main__':

    run_content_loss()
    run_tv_loss()