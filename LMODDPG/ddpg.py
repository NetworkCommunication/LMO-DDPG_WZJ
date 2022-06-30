
import os
import math
import csv
import time
import random
import argparse
from collections import deque
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import TimeDistributed, BatchNormalization, Flatten, Add, Lambda
from keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU, Activation
from keras.optimizers import Adam
from keras.models import Model
from PIL import Image
from airsim_env2 import Env
import datetime


np.set_printoptions(suppress=True, precision=4)
agent_name = 'ddpg'

whole_location=0
ini_yaw=0

flag=0
class DDPGAgent(object):

    def __init__(self, state_size, action_size, actor_lr, critic_lr, tau,
                 gamma, lambd, batch_size, memory_size,
                 epsilon, epsilon_end, decay_step, load_model):
        self.state_size = state_size
        self.vel_size = 3
        self.action_size = action_size
        self.action_high = 1.5
        self.action_low = -self.action_high
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.lambd = lambd
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_step = decay_step
        self.epsilon_decay = (epsilon - epsilon_end) / decay_step

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.actor_update = self.build_actor_optimizer()
        self.critic_update = self.build_critic_optimizer()
        self.sess.run(tf.global_variables_initializer())
        if load_model:
            self.load_model('./save_model/' + agent_name)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.memory = deque(maxlen=self.memory_size)

    def build_model(self):

        image = Input(shape=self.state_size)
        image_process = BatchNormalization()(image)
        image_process = TimeDistributed(
            Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Conv2D(8, (1, 1), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Flatten())(image_process)
        image_process = GRU(48, kernel_initializer='he_normal', use_bias=False)(image_process)
        image_process = BatchNormalization()(image_process)
        image_process = Activation('tanh')(image_process)
        vel = Input(shape=[self.vel_size])
        vel_process = Dense(48, kernel_initializer='he_normal', use_bias=False)(vel)
        vel_process = BatchNormalization()(vel_process)
        vel_process = Activation('tanh')(vel_process)
        state_process = Add()([image_process, vel_process])

        policy = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_process)
        policy = BatchNormalization()(policy)
        policy = ELU()(policy)
        policy = Dense(32, kernel_initializer='he_normal', use_bias=False)(policy)
        policy = BatchNormalization()(policy)
        policy = ELU()(policy)
        policy = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(
            policy)
        policy = Lambda(lambda x: K.clip(x, self.action_low, self.action_high))(policy)
        actor = Model(inputs=[image, vel], outputs=policy)

        action = Input(shape=[self.action_size])
        action_process = Dense(48, kernel_initializer='he_normal', use_bias=False)(action)
        action_process = BatchNormalization()(action_process)
        action_process = Activation('tanh')(action_process)
        state_action = Add()([state_process, action_process])

        Qvalue = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_action)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(32, kernel_initializer='he_normal', use_bias=False)(Qvalue)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue)
        critic = Model(inputs=[image, vel, action], outputs=Qvalue)

        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic


    def build_actor_optimizer(self):
        pred_Q = self.critic.output
        action_grad = tf.gradients(pred_Q, self.critic.input[2])
        target = -action_grad[0] / self.batch_size
        params_grad = tf.gradients(
            self.actor.output, self.actor.trainable_weights, target)
        params_grad, global_norm = tf.clip_by_global_norm(params_grad, 5.0)
        grads = zip(params_grad, self.actor.trainable_weights)
        optimizer = tf.train.AdamOptimizer(self.actor_lr)
        updates = optimizer.apply_gradients(grads)
        train = K.function(
            [self.actor.input[0], self.actor.input[1], self.critic.input[2]],
            [global_norm],
            updates=[updates]
        )
        return train


    def build_critic_optimizer(self):
        y = K.placeholder(shape=(None, 1), dtype='float32')
        pred = self.critic.output

        loss = K.mean(K.square(pred - y))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function(
            [self.critic.input[0], self.critic.input[1], self.critic.input[2], y],
            [loss],
            updates=updates
        )
        return train


    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        noise = np.random.normal(0, self.epsilon, self.action_size)
        action = np.clip(policy + noise, self.action_low,
                         self.action_high)
        return action, policy


    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)
        batch=np.array(batch)

        images = np.zeros([self.batch_size] + self.state_size)
        vels = np.zeros([self.batch_size, self.vel_size])
        actions = np.zeros((self.batch_size, self.action_size))
        rewards = np.zeros((self.batch_size, 1))
        next_images = np.zeros([self.batch_size] + self.state_size)
        next_vels = np.zeros([self.batch_size, self.vel_size])
        dones = np.zeros((self.batch_size, 1))


        for i, sample in enumerate(batch):

            images[i] = sample[0][0]
            vels[i] = sample[0][1]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_images[i] = sample[3][0]
            next_vels[i] = sample[3][1]
            dones[i] = sample[4]


        states = [images, vels]
        next_states = [next_images, next_vels]
        policy = self.actor.predict(states)
        target_actions = self.target_actor.predict(next_states)
        target_next_Qs = self.target_critic.predict(next_states + [target_actions])
        targets = rewards + self.gamma * (1 - dones) * target_next_Qs

        actor_loss = self.actor_update(states + [policy])
        critic_loss = self.critic_update(states + [actions, targets])
        return actor_loss[0], critic_loss[0]


    def check_batch(self,batch):

        notgood = [-1]
        for i, sample in enumerate(batch):
            vels = sample[0][1][0]
            yaw = speedyaw(vels)
            if ini_yaw <= 160 and ini_yaw >= 0:
                if yaw < ini_yaw - 20 or yaw > ini_yaw + 20:
                    np.append(notgood, i)
            if ini_yaw <= 0 and ini_yaw >= -160:
                if yaw < ini_yaw - 20 or yaw > ini_yaw + 20:
                    np.append(notgood, i)
            if ini_yaw >= 160 and ini_yaw <= 180:
                if yaw < ini_yaw - 20 or yaw < ini_yaw + 20 - 180 - 180:
                    np.append(notgood, i)
            if ini_yaw <= -160 and ini_yaw >= -180:
                if yaw > ini_yaw + 20 or yaw < (ini_yaw - 20 + 180) + 180:
                    np.append(notgood, i)
        return notgood


    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded')


    def save_model(self, name):
        self.actor.save_weights(name + '_actor.h5')
        self.critic.save_weights(name + '_critic.h5')


    def update_target_model(self):
        self.target_actor.set_weights(
            self.tau * np.array(self.actor.get_weights()) \
            + (1 - self.tau) * np.array(self.target_actor.get_weights())
        )
        self.target_critic.set_weights(
            self.tau * np.array(self.critic.get_weights()) \
            + (1 - self.tau) * np.array(self.target_critic.get_weights())
        )



def transform_input(responses, img_height, img_width):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = np.array(np.clip(255 * 3 * img1d, 0, 255), dtype=np.uint8)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    image = Image.fromarray(img2d)
    image = np.array(image.resize((img_width, img_height)).convert('L'))
    image = np.float32(image.reshape(1, img_height, img_width, 1))
    image /= 255.
    return image



def speedyaw(goal):
    yaw = 0
    if goal[0] >= 0 and goal[1] >= 0:
        yaw = math.atan(goal[1] / goal[0])
        return math.degrees(yaw)
    elif goal[0] >= 0 and goal[1] <= 0:
        yaw = math.atan(goal[1] / goal[0])
        return math.degrees(yaw)
    elif goal[0] <= 0 and goal[1] <= 0:
        yaw = math.atan(goal[1] / goal[0])
        return math.degrees(yaw) - 180
    else:
        yaw = math.atan(goal[1] / goal[0])
        return math.degrees(yaw) + 180


def transform_action(action):
    a = action[0]
    b = action[1]
    vel_yaw=np.array([a,b])
    vel_yaw=speedyaw(vel_yaw)

    if whole_location ==1:
        yaw=np.array([ini_yaw-90.0,ini_yaw+90.0])

        if a >= 0 and b <= 0:
            if vel_yaw <= yaw[0]:
                action[1] = -b
        if a <= 0 and b <= 0:
            action[0]=-a
            action[1]=-b
        if a <= 0 and b >= 0:
            if vel_yaw >= yaw[1]:
                action[0] = -a

    elif whole_location==2:
        yaw=np.array([ini_yaw-90.0,ini_yaw+90.0])


        if a >= 0 and b >= 0:
            if vel_yaw >= yaw[1]:
                action[1] = -b
        if a <= 0 and b <= 0:
            if vel_yaw <= yaw[0]:
                action[0] = -a
        if a <= 0 and b >= 0:
            action[0] = -a
            action[1] = -b

    elif whole_location==3:
        yaw=np.array([-180.0,-(ini_yaw+180.0),ini_yaw+270.0,180.0])

        if a >= 0 and b >= 0:
            action[0] = -a
            action[1] = -b
        if a >= 0 and b <= 0:
            if vel_yaw >= yaw[1]:
                action[0] = -a
        if a <= 0 and b >= 0:
            if vel_yaw <= yaw[2]:
                action[1] = -b

    elif whole_location==4:
        yaw=np.array([-180.0,ini_yaw-270.0,ini_yaw-90.0,180.0])

        if a >= 0 and b >= 0:
            if vel_yaw <= yaw[2]:
                action[0] = -a
        if a >= 0 and b <= 0:
            action[0] = -a
            action[1] = -b
        if a <= 0 and b <= 0:
            if vel_yaw >= yaw[1]:
                action[1] = -b


    real_action = np.array(action)
    return real_action





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--img_height', type=int, default=72)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=5e-4)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambd', type=float, default=0.90)
    parser.add_argument('--seqsize', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--train_start', type=int, default=500)

    parser.add_argument('--train_rate', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--decay_step', type=int, default=2000)

    args = parser.parse_args()

    if not os.path.exists('save_graph/' + agent_name):
        os.makedirs('save_graph/' + agent_name)
    if not os.path.exists('save_model' ):
        os.makedirs('save_model')
    if not os.path.exists('save_memory/' + agent_name):
        os.makedirs('save_memory/' + agent_name)

    state_size = [args.seqsize, args.img_height, args.img_width, 1]
    action_size = 3
    agent = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        lambd=args.lambd,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model
    )

    episode = 0
    ok=0
    env = Env()
    t0 = datetime.datetime.now()
    if args.play:
        while True:
            try:
                done = False
                bug = False


                bestY, timestep, score, avgvel, avgQ = 0., 0, 0., 0., 0.
                besty1, besty2, besty3, besty4 = -25.14, -14.77, -10.68, -72.48
                observe = env.reset()
                image = observe[0]
                vel = observe[1]
                try:
                    image = transform_input(image, args.img_height, args.img_width)
                except:
                    continue
                history = np.stack([image] * args.seqsize, axis=1)
                vel = vel.reshape(1, -1)
                state = [history, vel]
                while not done:
                    timestep += 1

                    action = agent.actor.predict(state)[0]
                    noise = [np.random.normal(scale=args.epsilon) for _ in range(action_size)]
                    noise = np.array(noise, dtype=np.float32)
                    action = np.clip(action + noise, -1, 1)
                    real_action = transform_action(action)
                    observe, reward, done, info = env.step(transform_action(real_action))
                    image = observe[0]
                    vel = observe[1]
                    try:
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        bug = True
                        break
                    history = np.append(history[:, 1:], [image], axis=1)
                    vel = vel.reshape(1, -1)
                    next_state = [history, vel]

                    avgQ += float(agent.critic.predict([state[0], state[1], action.reshape(1, -1)])[0][0])
                    avgvel += float(np.linalg.norm(real_action))
                    score += reward
                    if info['status'] == 'goal1':
                        if info['position'][1] >= besty1:
                            bestY = info['position'][1]
                        print('%s' % (real_action), end='\r', flush=True)

                    if info['status'] == 'goal2':
                        if info['position'][1] <= besty2:
                            bestY = info['position'][1]
                        print('%s' % (real_action), end='\r', flush=True)

                    if info['status'] == 'goal3':
                        if info['position'][1] <= besty3:
                            bestY = info['position'][1]
                        print('%s' % (real_action), end='\r', flush=True)

                    if info['status'] == 'goal4':
                        if info['position'][1] >= besty4:
                            bestY = info['position'][1]
                        print('%s' % (real_action), end='\r', flush=True)

                    if args.verbose:
                        print(
                            'Step %d Action %s Reward %.2f Info %s:' % (timestep, real_action, reward, info['status']))

                    state = next_state

                if bug:
                    continue

                avgQ /= timestep
                avgvel /= timestep

                print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f AvgVel %.2f'
                      % (episode, bestY, timestep, score, avgQ, avgvel))

                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
    else:

        if flag !=1:
            time_limit = 2000
            highscoreY = 0.


            global_step = 0
            while True:
                try:
                    print('episode:', episode)
                    done = False
                    bug = False

                    bestY, timestep, score, avgvel, avgQ, avgAct = 0., 1, 0., 0., 0., 0.
                    train_num, actor_loss, critic_loss = 0, 0., 0.
                    besty1, besty2, besty3, besty4 =  -25.14, -14.77, -10.68, -72.48
                    col=0
                    arrv = [[0, 0, 0]]
                    arrp = [[0, 0, 0]]
                    p = c = 0
                    observe = env.reset()
                    location = observe[3]
                    yaw = observe[4]


                    image = observe[0]
                    vel=observe[1]
                    pos=observe[2]

                    ini_yaw = yaw
                    whole_location = location
                    pos_x = pos[0]
                    pos_y = pos[1]

                    try:
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        continue
                    history = np.stack([image] * args.seqsize, axis=1)

                    vel = vel.reshape(1, -1)
                    state = [history, vel]
                    while not done and timestep < time_limit:
                        timestep += 1
                        global_step += 1

                        if len(agent.memory) >= args.train_start and global_step >= args.train_rate:
                            for _ in range(args.epoch):
                                a_loss, c_loss = agent.train_model()
                                actor_loss += float(a_loss)
                                critic_loss += float(c_loss)
                                train_num += 1

                            agent.update_target_model()
                            global_step = 0
                        action, policy = agent.get_action(state)
                        real_action, real_policy = transform_action(action), transform_action(policy)
                        observe, reward, done, info = env.step(real_action)
                        arrv.append(real_action)
                        arrp.append(observe[2])
                        p = p + int(Env.comsum)
                        c = c + int(Env.whole_collsion)
                        image = observe[0]
                        vel = observe[1]
                        try:
                            image = transform_input(image, args.img_height, args.img_width)
                        except:
                            bug = True
                            break
                        history = np.append(history[:, 1:], [image], axis=1)
                        vel = vel.reshape(1, -1)
                        next_state = [history, vel]
                        agent.append_memory(state, action, reward, next_state, done)
                        avgQ += float(agent.critic.predict([state[0], state[1], action.reshape(1, -1)])[0][0])
                        avgvel += float(np.linalg.norm(real_policy))
                        avgAct += float(np.linalg.norm(real_action))
                        score += reward
                        state = next_state

                        if agent.epsilon > agent.epsilon_end:
                            agent.epsilon -= agent.epsilon_decay

                    if bug:
                        continue
                    if train_num:
                        actor_loss /= train_num
                        critic_loss /= train_num
                    avgQ /= timestep
                    avgvel /= timestep
                    avgAct /= timestep

                    if args.verbose or episode % 10 == 0:
                        print(
                                'Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f AvgVel %.2f AvgAct %.2f Level %.2f'
                                % (episode, bestY, timestep, score, avgQ, avgvel, avgAct, info['level']))
                    stats = [
                            episode, timestep, score, bestY, avgvel, \
                            actor_loss, critic_loss, info['level'], avgQ, avgAct, info['status'],
                            time.asctime(time.localtime(time.time()))
                        ]

                    agent.save_model('./save_model/' + agent_name)

                    episode += 1

                    x =episode

                    print('status:', stats[10])
                    if stats[10] == 'goal1':
                        ok =ok + 1
                        print('ok',ok)


                    y=ok/(x*1.0)
                    t1=datetime.datetime.now()
                    t=(t1-t0).seconds
                    t0=t1
                    x=str(x)
                    y = str(y)
                    t=str(t)
                    f1=open('save_memory/'+agent_name + '/ddpg_x.csv',mode='a')
                    f1.write(x + '\n')
                    f1.close()

                    f2 = open('save_memory/' +agent_name+ '/ddpg_y.csv', mode='a')
                    f2.write(y + '\n')
                    f2.close()


                    f3 = open('save_memory/'+agent_name + '/ddpg_t.csv', mode='a')
                    f3.write(t + '\n')
                    f3.close()

                    p = str(p)
                    c = str(c)

                    f2 = open('save_memory/'+agent_name + '/ddpg_p.csv', mode='a')
                    f2.write(p + '\n')
                    f2.close()

                    f3 = open('save_memory/'+agent_name + '/ddpg_c.csv', mode='a')
                    f3.write(c + '\n')
                    f3.close()


                    v = str(arrv)
                    f6 = open('save_memory/' +agent_name+ '/ddpg_v.csv', mode='a')
                    f6.write(v + '\n')
                    f6.close()
                    arrv.clear()

                    position = str(arrp)
                    f7 = open('save_memory/'+agent_name + '/ddpg_position.csv', mode='a')
                    f7.write(position + '\n')
                    f7.close()
                    arrp.clear()


                except KeyboardInterrupt:
                    env.disconnect()
                    break