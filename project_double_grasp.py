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

# openravepy.RaveInitialize(True, openravepy.DebugLevel.Debug)


np.random.seed(0)
PACKAGE_NAME = 'hw1'

curr_path = os.getcwd()
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
        self.openrave_init()
        self.problem_init()
        # order grasps based on your own scoring metric
        self.order_grasps()

        # order grasps with noise
        # self.order_grasps_noisy()

    # the usual initialization for openrave
    def openrave_init(self):
        self.env = openravepy.Environment()
        self.env.SetViewer('qtcoin')
        self.env.GetViewer().SetName('HW1 Viewer')
        self.env.Load('models/%s.env.xml' % PACKAGE_NAME)
        # time.sleep(3) # wait for viewer to initialize. May be helpful to
        # uncomment
        self.robot = self.env.GetRobots()[0]


        right_relaxed = [5.65, -1.76, -0.26, 1.96, -1.15, 0.87, -1.43]
        left_relaxed = [0.64, -1.76, 0.26, 1.96, 1.16, 0.87, 1.43]

        self.right_manip = self.robot.GetManipulator('right_wam')
        # self.robot.SetActiveDOFs(right_manip.GetArmIndices())
        # self.robot.SetActiveDOFValues(right_relaxed)

        self.left_manip = self.robot.GetManipulator('left_wam')
        # self.robot.SetActiveDOFs(left_manip.GetArmIndices())
        # self.robot.SetActiveDOFValues(left_relaxed)

        self.manip = self.robot.GetActiveManipulator()
        self.end_effector = self.manip.GetEndEffector()

    # problem specific initialization - load target and grasp module
    def problem_init(self):
        self.target_kinbody = self.env.ReadKinBodyURI('models/objects/champagne.iv')
        # self.target_kinbody = self.env.ReadKinBodyURI('models/objects/winegoblet.iv')
        # self.target_kinbody = self.env.ReadKinBodyURI('models/objects/black_plastic_mug.iv')

        # change the location so it's not under the robot
        T = self.target_kinbody.GetTransform()
        T[1,2] = -1
        T[2,2] = 0
        T[2,1] = 1
        T[1,1] = 0
        T[0:3, 3] += np.array([0.35, 0, 0.5])
        self.target_kinbody.SetTransform(T)
        self.env.AddKinBody(self.target_kinbody)

        # create a grasping module
        self.gmodel = openravepy.databases.grasping.GraspingModel(
            self.robot, self.target_kinbody)
        # if you want to set options, e.g.  friction
        options = openravepy.options
        options.friction = 0.1
        if not self.gmodel.load():
            self.gmodel.autogenerate(options)

        self.graspindices = self.gmodel.graspindices
        self.grasps = self.gmodel.grasps

    def filter_grasp(self, grasp):

        # 1. Filter out bottom approaching direction
        dir_orig = grasp[self.graspindices['igraspdir']]
        grasp[self.graspindices['igraspdir']] = dir_orig
        # self.show_grasp(grasp)
        self.gmodel.showgrasp(grasp, showfinal=True)


        if dir_orig[1] >= 0: # upward
            return False

        # 2. Filter out contacts points at the bottom of the object
        contacts, finalconfig, mindist, volume = self.gmodel.testGrasp(
            grasp=grasp, translate=True, forceclosure=False)
        obj_position = self.gmodel.target.GetTransform()[0:3, 3]
        if len(contacts) == 0:
            return -1
        for c in contacts:
            pos = c[0:3] - obj_position
            if abs(pos[1]) < 0.001:
                return False

        return True

    def check_collision(self, grasp1, grasp2):
        self.robot.SetActiveManipulator(self.manip)
        self.robot.SetTransform(np.eye(4))  # have to reset transform in order to remove randomness
        self.robot.SetDOFValues(grasp1[self.graspindices.get('igrasppreshape')], self.manip.GetGripperIndices())
        self.robot.SetActiveDOFs(self.manip.GetGripperIndices(),
                                 self.robot.DOFAffine.X + self.robot.DOFAffine.Y + self.robot.DOFAffine.Z)
        self.env.UpdatePublishedBodies()
        time.sleep(1)
        #
        # self.robot.SetTransform(np.eye(4))  # have to reset transform in order to remove randomness
        # self.robot.SetDOFValues(grasp1[self.graspindices.get('igrasppreshape')], self.manip.GetGripperIndices())
        # # self.robot.SetActiveDOFs(self.manip.GetGripperIndices(),
        # #                          self.robot.DOFAffine.X + self.robot.DOFAffine.Y + self.robot.DOFAffine.Z)

    # order the grasps - call eval grasp on each, set the 'performance' index,
    # and sort
    def order_grasps(self):
        self.grasps_ordered = []

        for grasp in self.grasps:
            self.check_collision(grasp, grasp)
            if self.filter_grasp(grasp):
                self.grasps_ordered.append(grasp)

        for grasp in self.grasps_ordered:
            grasp[self.graspindices.get('performance')] = self.eval_grasp(grasp)

        # sort!
        order = np.argsort(self.grasps_ordered[:, self.graspindices.get('performance')[0]])
        order = order[::-1]
        self.order_mapping = order
        self.grasps_ordered = self.grasps_ordered[order]

        print "top grasps: "
        print order[0:4]

        for grasp in self.grasps_ordered[0:4]:
            print grasp[self.graspindices.get('performance')]
            self.show_grasp(grasp, delay=2)

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