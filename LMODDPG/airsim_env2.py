

import time
import numpy as np
import airsim
import math

clockspeed = 0.5
timeslice = clockspeed /2
ahpal=0.3
beta = 0.3
gama=0.2
theata=0.2

wholeplace=0


goalY = 4100
outY = -1
floorZ = 1500


goal1 = [-45.96, 26.8, -2]
goal2 = [-12.11, -14.77, 2]
goal3 = [-87.39, -10.68, 7]
goal4 = [-10.54, -72.48, 12.8]
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']


class Env:
    start_time = time.time()
    whole_collsion = 0
    initial_yaw=0
    location=0
    testflag = 1
    flag=0
    speed_low_range=8
    comsum=0
    lidarrange=7
    disini = 45.96 * 45.96 + 26.8 * 26.8 + 0.5 * 0.5
    d1 = d2 = 0
    final_yaw = 0
    def yaw(goal):
        yaw=0
        if goal[0]>=0 and goal[1]>=0:
            yaw=math.atan(goal[1]/goal[0])
            Env.location=1
            return math.degrees(yaw)
        elif goal[0]>=0 and goal[1]<=0:
            yaw = math.atan(goal[1]/goal[0])
            Env.location = 2
            return math.degrees(yaw)
        elif goal[0]<=0 and goal[1]<=0:
            yaw = math.atan(goal[1]/goal[0])
            Env.location = 3
            return math.degrees(yaw)-180
        else:
            yaw = math.atan(goal[1] / goal[0])
            Env.location = 4
            return math.degrees(yaw) + 180

    def __init__(self):

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_size = 3
        self.level = 0

    def reset(self):
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        Env.whole_collsion=0

        self.client.simPause(False)
        drivetrain = airsim.DrivetrainType.ForwardOnly
        goal = np.array([goal1[0], goal1[1]])
        yaw=Env.yaw(goal)
        yaw_mode = airsim.YawMode(False, yaw)
        Env.initial_yaw=yaw

        self.client.moveByVelocityAsync(0, 0, -1, 2,drivetrain,yaw_mode).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position

        quad_pos = np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val])

        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        col=0
        observation = [responses, quad_vel, quad_pos,Env.location,yaw,col]
        return observation

    def step(self, quad_offset):

        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_pos = np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val])

        drivetrain = airsim.DrivetrainType.ForwardOnly
        yaw_mode = airsim.YawMode(False, 0)

        yaw = Env.initial_yaw

        increase_high = 0

        if quad_pos[2]>0:
            increase_high= -2
        elif quad_pos[2] <-5:
            increase_high = 2

        quad_offset = [float(i) for i in quad_offset]
        self.client.simPause(False)

        has_collided = False
        landed = False
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2]+increase_high, timeslice, drivetrain,yaw_mode)
        Env.final_yaw = yaw

        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < timeslice:

            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity



            collided = self.client.simGetCollisionInfo().has_collided

            landed = (quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = landed or quad_pos.z_val > floorZ
            collision = collided or landed
            if collision:
                collision_count += 1
                Env.whole_collsion += 1
            if collision_count > 20:
                has_collided = True
                break
                t=time.time()-start_time


        self.client.simPause(True)


        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])


        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        dead = has_collided
        done = dead or (quad_pos.x_val <= goal1[0] and quad_pos.y_val >= goal1[1])


        reward = self.compute_reward(quad_pos, quad_vel, collision_count)


        info = {}
        info['position'] = np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val])
        info['level'] = self.level
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.y_val <= outY:
            info['status'] = 'out'
        elif quad_pos.x_val <= goal1[0] and quad_pos.y_val >= goal1[1]:
            info['status'] = 'goal1'
        elif quad_pos.x_val == goal2[0] and quad_pos.y_val == goal2[1] and quad_pos.z_val == goal2[2]:
            info['status'] = 'goal2'
        elif quad_pos.x_val == goal3[0] and quad_pos.y_val == goal3[1] and quad_pos.z_val == goal3[2]:
            info['status'] = 'goal3'
        elif quad_pos.x_val == goal4[0] and quad_pos.y_val == goal4[1] and quad_pos.z_val == goal4[2]:
            info['status'] = 'goal4' 
        else:
            info['status'] = 'going'

        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])

        quad_pos = np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val])


        observation = [responses, quad_vel, quad_pos]

        return observation, reward, done, info

    def parse_lidarData(self, data):


        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))

        return points

    def lidar(self):

        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_pos = np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val])

        lidarData1 = self.client.getLidarData(lidar_name="LidarSensor1")
        points1 = self.parse_lidarData(lidarData1)
        a1 = 100
        point1 = -2
        mid1=-1
        for i in range(0, int(points1.size / 3)):

            x = points1[i][0]
            y = points1[i][1]
            b = x * x + y * y
            b = math.sqrt(b)
            if b < Env.speed_low_range:
                point1 = -1
                if b < Env.lidarrange:
                    if b < a1:
                        a1 = b
                        mid1 = i

        if mid1!=-1:
            point1=mid1



        lidarData2 = self.client.getLidarData(lidar_name="LidarSensor2")
        points2 = self.parse_lidarData(lidarData2)
        a2 = 100
        point2 = -2
        mid2=-1
        for i in range(0, int(points2.size / 3)):

            x = points2[i][0]
            y = points2[i][1]
            b = x * x + y * y
            b = math.sqrt(b)
            if b < Env.speed_low_range:
                point2 = -1
                if b < Env.lidarrange:
                    if b < a2:
                        a2 = b
                        mid2 = i

        if mid2!=-1:
            point2=mid2


        if (point1 == -2 or point1 == -1) and (point2 == -1 or point2 == -2):
            obstacle = 0
            if point1==-1:
                wholeplace=2
            if point2==-1:
                wholeplace=1
        elif point1 ==-2 and point2==-2 :
            obstacle=-1
        else:
            obstacle= 1
        return obstacle

    def directflying(self):
        drivetrain = airsim.DrivetrainType.ForwardOnly
        yaw_mode = airsim.YawMode(False, 0)
        self.client.simPause(False)
        self.client.moveByVelocityAsync(-1.27, 1, 0, 1, drivetrain, yaw_mode)
        self.client.simPause(True)

    def ustatus(self):
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_pos = np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val])

        return quad_pos

    def compute_reward(self, quad_pos, quad_vel, collision_count):
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float)
        dis1 = (quad_pos.x_val - goal1[0] ) * (quad_pos.x_val - goal1[0])
        dis2 = (quad_pos.y_val - goal1[1] ) * (quad_pos.y_val - goal1[1] )
        dis3 = (quad_pos.z_val - goal1[2] ) * (quad_pos.z_val - goal1[2] )
        dis = dis1 + dis2 + dis3
        disnew=abs(dis-Env.disini)


        dis_value = ahpal * math.exp(-disnew / 1000)

        Env.disini=dis

        reward = dis_value

        return reward

    def change(self,number):
        if number>0:
            return number
        else:
            return abs(number)

    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')