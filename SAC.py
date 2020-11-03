import datetime
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

warm_start_iterations = 1000
gamma = 0.99
alpha = 0.2
lr = 3e-4

state_dim = 3
action_dim = 1

replay_buffer_pos = 0
replay_buffer_cap = 1000000
replay_buffer_populated = False
replay_buffer = {
    "states": np.zeros((replay_buffer_cap, state_dim)),
    "actions": np.zeros((replay_buffer_cap, action_dim)),
    "rewards": np.zeros((replay_buffer_cap, )),
    "next_states": np.zeros((replay_buffer_cap, state_dim)),
    "done": np.zeros((replay_buffer_cap, ))
}

def add_to_replay_buffer(s, a, r, s2, d):
    global replay_buffer_pos, replay_buffer_populated
    if replay_buffer_pos >= replay_buffer_cap:
        replay_buffer_pos = 0
        replay_buffer_populated = True
    replay_buffer["states"][replay_buffer_pos, :] = s
    replay_buffer["actions"][replay_buffer_pos, :] = a
    replay_buffer["rewards"][replay_buffer_pos] = r
    replay_buffer["next_states"][replay_buffer_pos, :] = s2
    replay_buffer["done"][replay_buffer_pos] = d
    replay_buffer_pos += 1

def build_critic(input_dim):
    return Sequential([
        Input(input_dim),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1)
    ])

c1 = build_critic(state_dim + action_dim)
c2 = build_critic(state_dim + action_dim)
c1t = tf.keras.models.clone_model(c1)
c2t = tf.keras.models.clone_model(c2)

copt = Adam(learning_rate=lr)

def polyak(target, source, factor=0.995):
    for lt, ls in zip(target.layers, source.layers):
        for wt, ws in zip(lt.trainable_weights, ls.trainable_weights):
            wt.assign(wt * factor + (1-factor)*ws)

def build_actor():
    input = Input(state_dim)
    stub = Sequential([
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
    ])(input)
    mean = Dense(action_dim)(stub)
    log_std = Dense(action_dim)(stub)
    return Model(inputs=input, outputs=(mean, log_std))
actor = build_actor()
aopt = Adam(learning_rate=lr)

def sample_action(state):
    mean, log_std = actor(state)
    d = tfp.distributions.Normal(mean, tf.exp(log_std))
    action = d.sample()
    lp = tf.reduce_sum(d.log_prob(action), axis=-1)
    lp -= tf.reduce_sum(2 * (tf.math.log(2.0) - action - tf.math.softplus(-2*action)), axis=-1)
    action_squashed = tf.tanh(action)
    return action_squashed, lp

def training_step(batch_size=256):
    upper = replay_buffer_cap if replay_buffer_populated else replay_buffer_pos
    idx = np.arange(upper)
    np.random.shuffle(idx)
    batch_idx = idx[:batch_size]
    s = replay_buffer["states"][batch_idx]
    a = replay_buffer["actions"][batch_idx]
    r = replay_buffer["rewards"][batch_idx]
    s2 = replay_buffer["next_states"][batch_idx]
    d = replay_buffer["done"][batch_idx]

    a2, lp2 = sample_action(s2)
    i2 = tf.concat((s2, a2), axis=-1)
    q_target = r + gamma * (tf.minimum(c1t(i2)[:, 0], c2t(i2)[:, 0]) - alpha * lp2)

    i1 = tf.concat((s, a), axis=-1)
    with tf.GradientTape() as c_tape:
        critic_loss = MSE(c1(i1)[:, 0], q_target) + MSE(c2(i1)[:, 0], q_target)
    c1_grad, c2_grad = c_tape.gradient(critic_loss, [c1.trainable_weights, c2.trainable_weights])
    copt.apply_gradients(zip(c1_grad, c1.trainable_weights))
    copt.apply_gradients(zip(c2_grad, c2.trainable_weights))

    polyak(c1t, c1)
    polyak(c2t, c2)

    with tf.GradientTape() as a_tape:
        a1, lp1 = sample_action(s)
        i1 = tf.concat((s, a1), axis=-1)
        q = tf.minimum(c1(i1)[:, 0], c2(i1)[:, 0]) - alpha * lp1
        actor_loss = -q
    a_grad = a_tape.gradient(actor_loss, actor.trainable_weights)
    aopt.apply_gradients(zip(a_grad, actor.trainable_weights))

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/SACpy/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

env = gym.make("Pendulum-v0")
s = env.reset()
i = 0
episode = 0
episode_return = 0
render_interval = 5
while True:
    if i < warm_start_iterations:
        a = np.random.rand() * 2 - 1
    else:
        a, lp = sample_action(s.reshape(1, -1))
        a = a[:, 0]
    a = env.action_space.low + (a + 1) / 2 * (env.action_space.high - env.action_space.low)
    s2, r, d, _ = env.step(a)
    env.render() if render_interval is not None and episode % render_interval == 0 else None

    add_to_replay_buffer(s, a, r, s2, d)
    training_step()

    episode_return += r
    if not d:
        s = s2
    else:
        s = env.reset()
        with summary_writer.as_default():
            print(f"Episode: {episode} return: {episode_return} (total interactions: {i+1})")
            tf.summary.scalar('Return', episode_return, step=i)
        episode_return = 0
        episode += 1
    i += 1
