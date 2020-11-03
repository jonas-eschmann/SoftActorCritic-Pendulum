import datetime
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

warm_start_interactions = 1000
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

critic_1, critic_2 = [build_critic(state_dim + action_dim) for _ in range(2)]
critic_1_target, critic_2_target = [tf.keras.models.clone_model(c) for c in [critic_1, critic_2]]
critic_opt = Adam(learning_rate=lr)

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

def sample_action(state, deterministic=False):
    mean, log_std = actor(state)
    action_dist = tfp.distributions.Normal(mean, tf.exp(log_std))
    action = action_dist.sample() if not deterministic else mean
    action_log_prob = tf.reduce_sum(action_dist.log_prob(action), axis=-1)
    action_log_prob -= tf.reduce_sum(2 * (tf.math.log(2.0) - action - tf.math.softplus(-2*action)), axis=-1)
    action_squashed = tf.tanh(action)
    return action_squashed, action_log_prob

def training_step(batch_size=256, done_is_dead=False):
    upper = replay_buffer_cap if replay_buffer_populated else replay_buffer_pos
    idx = np.arange(upper)
    np.random.shuffle(idx)
    batch_idx = idx[:batch_size]
    states = replay_buffer["states"][batch_idx]
    actions = replay_buffer["actions"][batch_idx]
    rewards = replay_buffer["rewards"][batch_idx]
    next_states = replay_buffer["next_states"][batch_idx]
    done = replay_buffer["done"][batch_idx]

    next_actions, next_actions_log_prob = sample_action(next_states)
    next_critic_input = tf.concat((next_states, next_actions), axis=-1)
    next_min_q = tf.minimum(critic_1_target(next_critic_input)[:, 0], critic_2_target(next_critic_input)[:, 0])
    q_target = rewards + gamma * (next_min_q - alpha * next_actions_log_prob) * (1.0 - done if done_is_dead else 1.0)

    critic_input = tf.concat((states, actions), axis=-1)
    with tf.GradientTape() as c_tape:
        critic_loss = MSE(critic_1(critic_input)[:, 0], q_target) + MSE(critic_2(critic_input)[:, 0], q_target)
    c1_grad, c2_grad = c_tape.gradient(critic_loss, [critic_1.trainable_weights, critic_2.trainable_weights])
    critic_opt.apply_gradients(zip(c1_grad, critic_1.trainable_weights))
    critic_opt.apply_gradients(zip(c2_grad, critic_2.trainable_weights))

    polyak(critic_1_target, critic_1)
    polyak(critic_2_target, critic_2)

    with tf.GradientTape() as a_tape:
        new_actions, new_actions_log_prob = sample_action(states)
        new_critic_input = tf.concat((states, new_actions), axis=-1)
        new_min_q = tf.minimum(critic_1(new_critic_input)[:, 0], critic_2(new_critic_input)[:, 0])
        actor_loss = -(new_min_q - alpha * new_actions_log_prob)
    a_grad = a_tape.gradient(actor_loss, actor.trainable_weights)
    aopt.apply_gradients(zip(a_grad, actor.trainable_weights))

def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/SACpy/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    env = gym.make("Pendulum-v0")
    state = env.reset()
    interaction = 0
    episode = 0
    episode_return = 0
    render_interval = 1
    while True:
        if interaction < warm_start_interactions:
            action = np.random.rand() * 2 - 1
        else:
            action, _ = sample_action(state.reshape(1, -1))
            action = action[:, 0]
        action = env.action_space.low + (action + 1) / 2 * (env.action_space.high - env.action_space.low)
        next_state, reward, done, _ = env.step(action)

        add_to_replay_buffer(state, action, reward, next_state, done)
        training_step()

        episode_return += reward
        if not done:
            state = next_state
        else:
            state = env.reset()
            with summary_writer.as_default():
                print(f"Episode: {episode} return: {episode_return} (total interactions: {interaction+1})")
                tf.summary.scalar('Return', episode_return, step=interaction)
            episode_return = 0
            episode += 1
            if render_interval is not None and episode % render_interval == 0:
                done = False
                while not done:
                    state, reward, done, _ = env.step(sample_action(state.reshape(1, -1), deterministic=True)[0])
                    episode_return += reward
                    env.render()
                state = env.reset()
                print(f"Deterministic episode return: {episode_return}")
                episode_return = 0
        interaction += 1

main() if __name__ == "__main__" else None