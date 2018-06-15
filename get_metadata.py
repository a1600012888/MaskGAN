import os
import cv2 as cv
import json
import numpy as np
ORI_DATA_ROOT = '../leftImg8bit_trainvaltest/leftImg8bit'
NEW_DATA_ROOT = '../cvdl/cityscapes/data'
BENCHMARK_ROOT = os.path.join(NEW_DATA_ROOT, 'benchmark.json')
def process_data(phase):
    results = []
    data_root = os.path.join(NEW_DATA_ROOT, phase)
    if os.path.exists(data_root) == False:
        os.mkdir(data_root)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ORI_DATA_ROOT, phase)):
        for filepath in filenames:
            if filepath[0] == '.':
                continue
            ori_img = cv.imread(os.path.join(dirpath, filepath))
            ori_img = cv.resize(ori_img, (512, 256))
            img_path = os.path.join(data_root, filepath)
            cv.imwrite(img_path, ori_img)

            sy = np.random.randint(0, 255-64)
            sx = np.random.randint(0, 511-128)
            ey = sy + 64
            ex = sx + 128
            results.append({'img_path': img_path,
                            'sy': sy,
                            'sx': sx,
                            'ey': ey,
                            'ex': ex
                            })
    with open(os.path.join(NEW_DATA_ROOT, phase+'.json'), 'w') as fp:
        json.dump(results, fp)
    print('processing data is done.')

def get_benchmark(nr_samples=100, seed=1234, att = False):
    # generate benchmark and store it into a json file
    global BENCHMARK_ROOT
    global NEW_DATA_ROOT
    if att:
        BENCHMARK_ROOT = BENCHMARK_ROOT = os.path.join(NEW_DATA_ROOT, 'benchmark_att.json')
    results = []
    np.random.seed(seed)
    count = 1
    for dirpath, dirnames, filenames in os.walk(os.path.join(ORI_DATA_ROOT, 'test')):
        for filepath in filenames:
            if filepath[0] == '.':
                continue
            rn = np.random.rand()
            if rn > 0.2:
                continue
            ori_path = os.path.join(dirpath, filepath)
            ori_img = cv.imread(ori_path)
            ori_img = cv.resize(ori_img, (512, 256))
            filepath = '%04d.png' % count
            #img_path = os.path.join(BENCHMARK_ROOT, filepath)
            count += 1
            #cv.imwrite(img_path, ori_img)


            sy = np.random.randint(0, 255-64)
            sx = np.random.randint(0, 511-128)
            ey = sy + 64
            ex = sx + 128
            if att:
                sy = np.random.randint(0, 127 - 32)
                sx = np.random.randint(0, 127 - 32)
                ey = sy + 32
                ex = sx + 32
            results.append({'filename': filepath,
                            'ori_path': ori_path,
                            'sy': sy,
                            'sx': sx,
                            'ey': ey,
                            'ex': ex
                            })
            if len(results) == nr_samples:
                break
        if len(results) == nr_samples:
            break

    with open(BENCHMARK_ROOT, 'w') as fp:
        json.dump(results, fp)
    print('total file:', len(results))
    print('benchmark generation is done.')


if __name__ == '__main__':
    #process_data('train')
    #process_data('val')
    #process_data('test')

    get_benchmark()
    get_benchmark(att = True)