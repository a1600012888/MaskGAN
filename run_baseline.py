import os

base_command = 'rlaunch --cpu=8 --gpu=1 --memory=8192 -- python3 train2.py --exp='

def run_tv_loss():

    tv_Weights = [5e-3,  1e-4]

    for tv in tv_Weights:

        command = base_command + 'tv_{}'.format(tv) + ' --tv={} &'.format(tv)

        os.system(command)


def run_content_loss():
    content_weights = [30, 5.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2]

    for c in content_weights:

        command = base_command + 'content_{}'.format(c) + ' --content={} &'.format(c)

        os.system(command)

def run_ad_loss():

    generator_Weights = [70, 30, 10, 5, 1, 0.3]

    for g in generator_Weights:

        command = base_command + 'AdvGene_{}'.format(g) + ' --adv={} &'.format(g)

        os.system(command)

def run_fake_loss():

    fake_Weights = [30, 10, 5, 3, 0.3, 0.05]

    for f in fake_Weights:

        command = base_command + 'FakeGene_{}'.format(f) + ' --fake={} &'.format(f)

        os.system(command)


if __name__ == '__main__':

    run_ad_loss()
    run_fake_loss()
    run_content_loss()

