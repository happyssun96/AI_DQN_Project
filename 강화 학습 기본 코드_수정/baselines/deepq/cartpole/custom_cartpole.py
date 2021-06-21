import gym
import sys
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import common.tf_util as U
import logger
import deepq
from deepq.replay_buffer import ReplayBuffer
from deepq.utils import ObservationInput
from common.schedules import LinearSchedule
### 라이브러리 환경 설정한 모음입니다.
import time
start = time.time()  # 학습 시작 시간을 저장합니다.

def model(inpt, num_actions, scope, reuse=False):
    """이 모델은 observation을 입력으로 하고 모든 행동의 값들을 반환합니다."""

    with tf.variable_scope(scope, reuse=reuse):
        out = inpt

        ### num_outputs 값 수정(32, 64, 128)_DQN Hidden Layer Neuron수를 조정합니다.
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        ################

        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session(num_cpu=1):
        # CartPole-v0이라는 gym 환경을 만들어서 env에 저장합니다.
        env = gym.make("CartPole-v0")
        # 모델 학습에 필요한 모든 함수들(act, train, update_target, debug)을 만듭니다.
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,  # 타겟 Q함수는 학습의 대상이 되는 Q함수가 학습이 되면서 계속 바뀌는 문제점을 해결하기 위한 해법입니다. 이렇게 함으로써 일정 스텝이 될 때마다 Q함수 가중치들을 타겟 Q함수에 업데이트합니다.
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # replay buffer를 생성합니다.
        replay_buffer = ReplayBuffer(50000)

        # 탐험할 스케줄을 1부터 0.02까지 감소하도록 만듭니다. 10000 step만 실행합니다.
        # 이때, 각 행동은 랜덤으로 하고 모델에 의해 예측된 값들에 의해 98% 의 행동들이 선택됩니다.
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # 파라미터들을 초기화하고 target network로 복사합니다.
        U.initialize()
        update_target()
        reward_list = []  # reward의 총합을 파일에 저장하는 리스트를 만듭니다.
        episode_rewards = [0.0]
        obs = env.reset()  # 환경을 초기화합니다.
        for t in itertools.count():  # 학습이 끝날 때까지 반복합니다.
            # 행동을 하고 가장 새로운 값으로 탐험을 업데이트합니다.
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)  # 에이전트에게 명령을 내리고 환경에서 observation 변수 등 여러 변수들을 반환합니다.
            # replay buffer에 바뀐 값을 저장합니다.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            reward_list.append(rew)  # 리스트에 reward를 추가합니다.
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()  # 환경을 초기화합니다.
                episode_rewards.append(0)

                ###Reward 파일에 저장합니다.(각각 맞게 파일명을 변경)
                with open("../../target1000_3.txt", "a") as f:
                    f.write(str(sum(reward_list)) + "\n")
                reward_list = []  # reward 리스트를 초기화합니다.

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            is_finished = len(episode_rewards) > 2000 and is_solved

            if is_solved:
                if len(episode_rewards) > 1998:
                    # 결과 상태를 화면에 출력합니다.
                    env.render()
                if is_finished:
                    sys.exit(0)

            else:
                # replay buffer로부터 샘플된 배치에 벨만 방정식의 에러를 최소화합니다.
                if t > 1000:
                    ###Replay Buffer Sample 수 수정(16, 32, 64)
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    ################
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                # target network를 주기적으로 업데이트합니다.
                ### Target Update 주기를 수정합니다(250, 500, 1000)
                if t % 1000 == 0:
                    #################
                    update_target()

            if done and len(episode_rewards) % 10 == 0:  #에피소드가 끝날 때마다 기록을 남깁니다.
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                print("time :", time.time() - start)  # 끝에 학습 소요 시간을 같이 출력합니다.
