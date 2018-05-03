#!/usr/bin/env python

# Standard Python Imports
import os
import copy
import time
import math
import numpy as np
import scipy
# OpenRAVE
import openravepy
from numpy import linalg as LA
from pyquaternion import Quaternion

# openravepy.RaveInitialize(True, openravepy.DebugLevel.Debug)


np.random.seed(0)
PACKAGE_NAME = 'double'

curr_path = os.getcwd() + '/src/double_grasping/scripts'
relative_ordata = '/models'
ordata_path_thispack = curr_path + relative_ordata


# this sets up the OPENRAVE_DATA environment variable to include the files
# we're using
openrave_data_path = os.getenv('OPENRAVE_DATA', '')
openrave_data_paths = openrave_data_path.split(':')
if ordata_path_thispack not in openrave_data_paths:
    if openrave_data_path == '':
        os.environ['OPENRAVE_DATA'] = ordata_path_thispack
    else:
        datastr = str('%s:%s' % (ordata_path_thispack, openrave_data_path))
        # set database file to be in this folder only
        os.environ['OPENRAVE_DATA'] = datastr
relative_ordatabase = '/database'
ordatabase_path_thispack = curr_path + relative_ordatabase
os.environ['OPENRAVE_DATABASE'] = ordatabase_path_thispack

# get rid of warnings
openravepy.RaveInitialize(True, openravepy.DebugLevel.Fatal)
openravepy.misc.InitOpenRAVELogging()

class RoboHandler:
    def __init__(self):
        self.is_test = True
        self.openrave_init()
        
        # order grasps based on your own scoring metric
        self.is_test = True

        # order grasps with noise
        # self.order_grasps_noisy()

    def set_object(self, object):
        self.problem_init(object)
        self.object = object

    # the usual initialization for openrave
    def openrave_init(self):
        self.env = openravepy.Environment()
        self.env.SetViewer('qtcoin')
        self.env.GetViewer().SetName('HW1 Viewer')
        self.env.Load(curr_path + '/models/%s.env.xml' % PACKAGE_NAME)

        # time.sleep(3) # wait for viewer to initialize. May be helpful to
        # uncomment
        self.robot_left = self.env.GetRobots()[0]
        self.robot_right = self.env.GetRobots()[1]


        right_relaxed = [5.65, -1.76, -0.26, 1.96, -1.15, 0.87, -1.43]
        left_relaxed = [0.64, -1.76, 0.26, 1.96, 1.16, 0.87, 1.43]

        self.right_manip = self.robot_right.GetManipulator('rightarm')
        # self.right_manip = self.robot_right.GetManipulator('right_wam')
        # self.robot.SetActiveDOFs(right_manip.GetArmIndices())
        # self.robot.SetActiveDOFValues(right_relaxed)

        self.left_manip = self.robot_left.GetManipulator('leftarm')
        # self.robot.SetActiveDOFs(left_manip.GetArmIndices())
        # self.robot.SetActiveDOFValues(left_relaxed)


        self.robot_right.SetActiveManipulator('rightarm')
        self.robot_left.SetActiveManipulator('leftarm')

        # self.manip = self.robot.GetActiveManipulator()
        # self.end_effector = self.manip.GetEndEffector()

    # problem specific initialization - load target and grasp module
    def problem_init(self, init_object='wood_block'):
        AVAILABLE_OBJECTS = ['wood_block_1inx1in', 'cheezit_1inx1in', 'domino_1inx1in']
        for available in AVAILABLE_OBJECTS:
            if self.env.GetKinBody(available):
                self.env.RemoveKinBody(self.env.GetKinBody(available))
        self.target_kinbody = self.env.ReadKinBodyXMLFile(curr_path + '/models/objects/' + init_object + '.kinbody.xml')
        # self.target_kinbody = self.env.ReadKinBodyXMLFile('models/objects/wood_block.kinbody.xml')
        # self.target_kinbody = self.env.ReadKinBodyXMLFile('models/objects/cheezit.kinbody.xml')
        # self.target_kinbody = self.env.ReadKinBodyURI('models/objects/champagne.iv')
        # self.target_kinbody = self.env.ReadKinBodyURI('models/objects/winegoblet.iv')
        # self.target_kinbody = self.env.ReadKinBodyURI('models/objects/black_plastic_mug.iv')

        # change the location so it's not under the robot
        T = self.target_kinbody.GetTransform()
        T[0:3, 3] += np.array([0.5, 0.5, 0.5])
        self.target_kinbody.SetTransform(T)
        self.env.AddKinBody(self.target_kinbody)

        # create a grasping module
        self.gmodel_left = openravepy.databases.grasping.GraspingModel(
            self.robot_left, self.target_kinbody)

        self.gmodel_right = openravepy.databases.grasping.GraspingModel(
            self.robot_right, self.target_kinbody)
        # if you want to set options, e.g.  friction
        options = openravepy.options
        options.friction = 0.1
        if not self.gmodel_left.load():
            self.gmodel_left.autogenerate(options)
        if not self.gmodel_right.load():
            self.gmodel_right.autogenerate(options)

        self.graspindices_left = self.gmodel_left.graspindices
        self.grasps_left = self.gmodel_left.grasps

        self.graspindices_right = self.gmodel_right.graspindices
        self.grasps_right = self.gmodel_right.grasps


        # self.graspindices_right = self.gmodel_left.graspindices
        # self.grasps_right = self.gmodel_left.grasps

    def filter_grasp(self, grasp, indices, gmodel):

        # 1. Filter out bottom approaching direction
        dir_orig = grasp[indices['igraspdir']]
        grasp[indices['igraspdir']] = dir_orig
        # self.show_grasp(grasp)

        if dir_orig[1] >= 0: # upward
            return False

        gmodel.showgrasp(grasp, showfinal=True, delay=0.5)

        # # 2. Filter out contacts points at the bottom of the object
        # contacts, finalconfig, mindist, volume = gmodel.testGrasp(
        #     grasp=grasp, translate=True, forceclosure=False)
        # obj_position = gmodel.target.GetTransform()[0:3, 3]
        # if len(contacts) == 0:
        #     return -1
        # for c in contacts:
        #     pos = c[0:3] - obj_position
        #     if abs(pos[1]) < 0.001:
        #         return False

        return True

    def check_collision(self, grasp_left, grasp_right):
        with openravepy.RobotStateSaver(self.gmodel_left.robot):
            with openravepy.RobotStateSaver(self.gmodel_right.robot):
                with self.env:
                    contacts_left, finalconfig_left, mindist_left, volume_left = self.gmodel_left.testGrasp(
                        grasp=grasp_left, translate=True, forceclosure=True)

                    contacts_right, finalconfig_right, mindist_right, volume_right = self.gmodel_right.testGrasp(
                        grasp=grasp_right, translate=True, forceclosure=True)

                    self.gmodel_left.robot.GetController().Reset(0)
                    self.gmodel_left.robot.SetDOFValues(finalconfig_left[0])
                    self.gmodel_left.robot.SetTransform(finalconfig_left[1])
                    self.gmodel_right.robot.GetController().Reset(0)
                    self.gmodel_right.robot.SetDOFValues(finalconfig_right[0])
                    self.gmodel_right.robot.SetTransform(finalconfig_right[1])


                    self.env.UpdatePublishedBodies()

                    return self.env.CheckCollision(self.robot_left, self.robot_right)

        # self.robot.SetTransform(np.eye(4))  # have to reset transform in order to remove randomness
        # self.robot.SetDOFValues(grasp1[self.graspindices.get('igrasppreshape')], self.manip.GetGripperIndices())
        # # self.robot.SetActiveDOFs(self.manip.GetGripperIndices(),
        # #                          self.robot.DOFAffine.X + self.robot.DOFAffine.Y + self.robot.DOFAffine.Z)

    def quaternion_from_matrix(self, matrix, isprecise=False):
        """Return quaternion from rotation matrix.

        If isprecise is True, the input matrix is assumed to be a precise rotation
        matrix and a faster algorithm is used.

        q = quaternion_from_matrix(np.identity(4), True)
        np.allclose(q, [1, 0, 0, 0])
        True
        q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
        np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
        True
        R = rotation_matrix(0.123, (1, 2, 3))
        q = quaternion_from_matrix(R, True)
        np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
        True
        R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
        ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
        q = quaternion_from_matrix(R)
        np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
        True
        R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
        ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
        q = quaternion_from_matrix(R)
        np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
        True
        R = random_rotation_matrix()
        q = quaternion_from_matrix(R)
        is_same_transform(R, quaternion_matrix(q))
        True
        is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
        ...                    quaternion_from_matrix(R, isprecise=True))
        True
        R = euler_matrix(0.0, 0.0, np.pi/2.0)
        is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
        ...                    quaternion_from_matrix(R, isprecise=True))
        True

        """
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4,))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[1, 1] > M[0, 0]:
                    i, j, k = 1, 2, 0
                if M[2, 2] > M[i, i]:
                    i, j, k = 2, 0, 1
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                          [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                          [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                          [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q

    def get_grasp_relative_pose(self, grasp, gmodel, graspindices):
        direction=grasp[graspindices.get('igraspdir')]
        roll=grasp[graspindices.get('igrasproll')]
        position=grasp[graspindices.get('igrasppos')]
        standoff=grasp[graspindices.get('igraspstandoff')]
        manipulatordirection=grasp[graspindices.get('imanipulatordirection')]

        # pose = [position, direction]
        # gmodel.moveToPreshape(grasp)


        Tgrasp = gmodel.GetLocalGraspTransform(grasp, collisionfree=True)
        quaternion = self.quaternion_from_matrix(Tgrasp)
        # basemanip = openravepy.interfaces.BaseManipulation(self.robot_left)
        # basemanip.MoveToHandPosition(matrices=[Tgrasp+

        # print("test")
        position = Tgrasp[:3, 3]
        return (position, quaternion)

        # order the grasps - call eval grasp on each, set the 'performance' index,
    # and sort
    def order_grasps(self):

        # 1. filter out single grasp
        if self.is_test:
            import IPython
            IPython.embed()
            index_left = 0
            index_right = 0
            if self.object == "wood_block":
                index_left = 47
                index_right = 66
            elif self.object == "cheezit":
                index_left = 0
                index_right = 3
            elif self.object == "domino":
                index_left = 17
                index_right = 28 # 27 other direction

            grasp_left = self.grasps_left[index_left]
            grasp_right = self.grasps_right[index_right]

            # grasp_left[self.graspindices_left.get('igrasppos')] = np.array([0,0,0])

            # self.gmodel_left.showgrasp(grasp_left, showfinal=True)

            grasp_left_relative_pose = self.get_grasp_relative_pose(grasp_left, self.gmodel_left, self.graspindices_left)
            grasp_right_relative_pose = self.get_grasp_relative_pose(grasp_right, self.gmodel_right, self.graspindices_right)

            return {'grasp_left_relative_pose':grasp_left_relative_pose, 'grasp_right_relative_pose':grasp_right_relative_pose}

            # Tgrasp_left = self.gmodel_left.getGlobalGraspTransform(grasp_left, collisionfree=True)
            # contacts, finalconfig, mindist, volume = self.gmodel_left.testGrasp(grasp=grasp_left, translate=True,
                                                                                # forceclosure=False)
            # self.robot.SetDOFValues(grasp_left[self.graspindices['igrasppreshape']], self.manip.GetGripperIndices())
        else:
            self.grasps_filtered_left = []
            self.grasps_filtered_right = []
            for grasp in self.grasps_left:
                if self.filter_grasp(grasp, indices=self.graspindices_left, gmodel=self.gmodel_left):
                    self.grasps_filtered_left.append(grasp)


            for grasp in self.grasps_right:
                if self.filter_grasp(grasp, indices=self.graspindices_right, gmodel=self.gmodel_right):
                    self.grasps_filtered_right.append(grasp)

            self.grasps_filtered_left = self.grasps_left
            self.grasps_filtered_right = self.grasps_right

            # 2. Construct double grasps pairs
            self.double_grasps = []

            for grasp_left in self.grasps_filtered_left:
                for grasp_right in self.grasps_filtered_right:
                    grasp_pair = [grasp_left, grasp_right]
                    self.double_grasps.append(grasp_pair)

            from random import shuffle
            shuffle(self.double_grasps)

            # 3. check double grasp collision
            self.double_filtered_grasps = []
            for grasp_pair in self.double_grasps:
                if not self.check_collision(grasp_left=grasp_pair[0], grasp_right=grasp_pair[1]):
                    self.double_filtered_grasps.append(grasp_pair)

        # for grasp in self.grasps_ordered:
        #     grasp[self.graspindices.get('performance')] = self.eval_grasp(grasp)
        #
        # # sort!
        # order = np.argsort(self.grasps_ordered[:, self.graspindices.get('performance')[0]])
        # order = order[::-1]
        # self.order_mapping = order
        # self.grasps_ordered = self.grasps_ordered[order]
        #
        # print "top grasps: "
        # print order[0:4]
        #
        # for grasp in self.grasps_ordered[0:4]:
        #     print grasp[self.graspindices.get('performance')]
        #     self.show_grasp(grasp, delay=2)

    # order the grasps - but instead of evaluating the grasp, evaluate random
    # perturbations of the grasp
    def order_grasps_noisy(self):
        # you should change the order of self.grasps_ordered_noisy
        # self.grasps_ordered_noisy = self.grasps.copy()  # for testing only
        self.grasps_ordered_noisy = self.grasps_ordered.copy()
        random_no = 5
        for grasp in self.grasps_ordered_noisy[0:50]: #Only compare the top 50
            # QA: take average
            total_performance = 0
            for j in range(random_no):
                grasp_noise = self.sample_random_grasp(grasp)
                performance = self.eval_grasp(grasp_noise)
                total_performance += performance

            grasp[self.graspindices.get('performance')] = total_performance / random_no

        # sort!
        order = np.argsort(self.grasps_ordered_noisy[:, self.graspindices.get('performance')[0]])

        # order = np.argsort(self.grasps[:, self.graspindices.get('performance')[0]])  # for testing only
        order = order[::-1]

        print "top grasps (noisy): "
        self.grasps_ordered_noisy = self.grasps_ordered_noisy[order]
        old_older = []
        for new_idx in order[0:4]:
            old_older.append(self.order_mapping[new_idx])
        print old_older

        for grasp in self.grasps_ordered_noisy[0:4]:
            print grasp[self.graspindices.get('performance')]
            self.show_grasp(grasp, delay=3)

    # function to evaluate grasps
    # returns a score, which is some metric of the grasp
    # higher score should be a better grasp
    def eval_grasp(self, grasp):
        with self.robot:
            try:
                contacts, finalconfig, mindist, volume = self.gmodel.testGrasp(
                    grasp=grasp, translate=True, forceclosure=False)
                obj_position = self.gmodel.target.GetTransform()[0:3, 3]
                if len(contacts) == 0:
                    return -1
                G_transposed = np.empty((0, 6))  # the transposed wrench matrix
                for c in contacts:

                    pos = c[0:3] - obj_position
                    dir = -c[3:]  # this is already a unit vector
                    torque_temp = np.cross(pos, dir)
                    wrench_i = np.concatenate([dir, torque_temp])
                    G_transposed = np.append(G_transposed, [wrench_i], axis=0)

                G = np.transpose(G_transposed)

                G_squared = np.matmul(G, G_transposed)
                w, v = LA.eig(G_squared)
                eig_srt = np.sqrt(np.absolute(w))
                eig_srt_min = abs(eig_srt[-1])
                eig_srt_max = abs(eig_srt[0])

                grasp_isotropy_index = 0
                if eig_srt_max > 0:
                    grasp_isotropy_index = eig_srt_min / eig_srt_max

                return grasp_isotropy_index

            except openravepy.planning_error as e:
                # you get here if there is a failure in planning
                # example: if the hand is already intersecting the object at
                # the initial position/orientation
                return -1  # TODO you may want to change this

            # heres an interface in case you want to manipulate things more specifically
            # NOTE for this assignment, your solutions cannot make use of graspingnoise

            # self.robot.SetTransform(np.eye(4)) # have to reset transform in order to remove randomness
            # self.robot.SetDOFValues(grasp[self.graspindices.get('igrasppreshape')], self.manip.GetGripperIndices())
            # self.robot.SetActiveDOFs(self.manip.GetGripperIndices(), self.robot.DOFAffine.X + self.robot.DOFAffine.Y + self.robot.DOFAffine.Z)
            # self.gmodel.grasper = openravepy.interfaces.Grasper(self.robot, friction=self.gmodel.grasper.friction, avoidlinks=[], plannername=None)
            # contacts, finalconfig, mindist, volume = self.gmodel.grasper.Grasp( \
            #    direction             = grasp[self.graspindices.get('igraspdir')], \
            #    roll                  = grasp[self.graspindices.get('igrasproll')], \
            #    position              = grasp[self.graspindices.get('igrasppos')], \
            #    standoff              = grasp[self.graspindices.get('igraspstandoff')], \
            #    manipulatordirection  = grasp[self.graspindices.get('imanipulatordirection')], \
            #    target                = self.target_kinbody, \
            #    graspingnoise         = 0.0, \
            #    forceclosure          = True, \
            #    execute               = False, \
            #    outputfinal           = True, \
            #    translationstepmult   = None, \
            #    finestep              = None )

    # given grasp_in, create a new grasp which is altered randomly
    # you can see the current position and direction of the grasp by:
    # grasp[self.graspindices.get('igrasppos')]
    # grasp[self.graspindices.get('igraspdir')]
    def sample_random_grasp(self, grasp_in):
        grasp = grasp_in.copy()

        # sample random position
        RAND_DIST_SIGMA = 0.01
        pos_orig = grasp[self.graspindices['igrasppos']]
        pos_new = np.random.normal(pos_orig, RAND_DIST_SIGMA)

        # sample random orientation
        RAND_ANGLE_SIGMA = np.pi / 24
        dir_orig = grasp[self.graspindices['igraspdir']]
        roll_orig = grasp[self.graspindices['igrasproll']]
        dir_new = np.random.normal(dir_orig, RAND_ANGLE_SIGMA)
        roll_new = np.random.normal(roll_orig, RAND_ANGLE_SIGMA)

        grasp[self.graspindices['igrasppos']] = pos_new
        grasp[self.graspindices['igraspdir']] = dir_new
        grasp[self.graspindices['igrasproll']] = roll_new

        return grasp

    def replay_grasps(self, grasp_inds):
        performance = list()
        for ind in grasp_inds:
            grasp = self.grasps[ind]
            performance.append(self.eval_grasp(grasp))
            self.show_grasp(grasp, delay=3)
            print(str(ind) + ": " + str(performance[-1]))
        print(grasp_inds)
        print(performance)

    # displays the grasp
    def show_grasp(self, grasp, delay=0.5):
        with openravepy.RobotStateSaver(self.gmodel.robot):
            with self.gmodel.GripperVisibility(self.gmodel.manip):
                # time.sleep(0.1)  # let viewer update?
                try:
                    with self.env:
                        contacts, finalconfig, mindist, volume = self.gmodel.testGrasp(
                            grasp=grasp, translate=True, forceclosure=True)
                        # if mindist == 0:
                        #  print 'grasp is not in force closure!'
                        contactgraph = self.gmodel.drawContacts(
                            contacts) if len(contacts) > 0 else None
                        self.gmodel.robot.GetController().Reset(0)
                        self.gmodel.robot.SetDOFValues(finalconfig[0])
                        self.gmodel.robot.SetTransform(finalconfig[1])
                        self.env.UpdatePublishedBodies()
                        time.sleep(delay)
                except openravepy.planning_error as e:
                    print 'bad grasp!', e


if __name__ == '__main__':
    robo = RoboHandler()
    # time.sleep(10000) #to keep the openrave window open
