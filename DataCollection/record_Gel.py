import rospy
from datetime import datetime
import os
import cv2
from stable_stacking.utils import utils

# sys.path.append("./sensors")
from gelsight import GelSight


def record_Gel():
    rospy.init_node('record')

    now = datetime.now()
    # Directory to save data
    parent_dir = '../data/gel/gs_mini_28N0_295H'
    utils.make_dir(parent_dir)
    # Create Date Directiory
    date = now.strftime('%m_%d_%Y')
    date_dir = os.path.join(parent_dir, date)
    if os.path.isdir(date_dir):
        print('Date Directory Exists\n')
    else:
        print('Creating Date Directory: %s\n' % date)
        os.mkdir(date_dir)

    # Create Object Directory
    default_object = 'unnamed_%s' % now.strftime('d%d%m%Y_t%H%M%S')
    object_name = input('Please enter object name (default name: %s): ' % default_object)
    if object_name == '':
        object_name = default_object
    else:
        object_name = object_name
    # Input how many data frames to collect
    num_frames = int(input('Please enter number of frames to collect: '))
    object_dir = os.path.join(date_dir, object_name)
    os.mkdir(object_dir)
    print('Current Object Directory:')
    print(object_dir)

    # Data Collection Modules Instantiation
    gelsight = GelSight(object_dir)

    rate = rospy.Rate(100)
    print("Data Collection Started!")
    while not rospy.is_shutdown():
        rate.sleep()
        key = input("input s to save a new frame: ")
        if key == 's':
            filename = 'frame_' + str(gelsight.gelsight_count) + '.jpg'
            cv2.imwrite(gelsight.gel_path + '/' + filename, gelsight.img)
            gelsight.gelsight_count += 1
            print("save frame: " + str(gelsight.gelsight_count))
        if gelsight.gelsight_count == num_frames:
            gelsight.stopRecord()
            break
    print("Data Collection Completed!")


if __name__ == '__main__':
    record_Gel()
