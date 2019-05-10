import cv2
import numpy as np
import ipdb
import json
pdb = ipdb.set_trace


def get_data(input_path, cache=False):

    if cache:
        print('直接加载cache from cache_train.json ...')
        data = json.load(open('cache_train.json'))
        all_data = data['all_data']
        classes_count = data['classes_count']
        class_mapping = data['class_mapping']
        return all_data, classes_count, class_mapping

    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True
    annsName = input_path + '/anns.txt'
    with open(annsName, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(
                float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)), 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys(
                ) if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        import ipdb; ipdb.set_trace()
        if cache == False:
            cache = {'all_data': all_data, 'classes_count': classes_count,
                    'class_mapping': class_mapping}
            cache_data = json.dumps(cache)
            with open('./cache_train.json', 'w') as f:
                f.write(cache_data)
        return all_data, classes_count, class_mapping
