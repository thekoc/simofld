import sys
sys.path.insert(0, 'src')

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from numpy import random
import numpy as np


from simofld.br import BRProfile
from simofld import masl
from simofld import br

def generate_profile():
    model = masl
    random.seed(2)
    until = 1
    step_interval = 1
    user_num = 30
    channel_num = 5
    # distances = 22.5 + random.random(user_num) * 5
    distances = 10 + random.random(user_num) * 40
    active_probabilities = 1 - random.random(user_num)
    channels = [model.RayleighChannel() for _ in range(channel_num)]
    users = [model.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
    cloud_server = model.CloudServer()
    profile = model.MASLProfile(users, 2)
    with model.create_env(users, cloud_server, profile, until=until, step_interval=step_interval) as env:
        env.run()
    return {
        'users': users,
        'profile': profile
    }


x_ds = np.linspace(1, 20e8, 50)
y_local_f = np.linspace(1, 20e8, 50)

X, Y = np.meshgrid(x_ds, y_local_f)

pairs = []
Z = np.empty_like(X)
for i in range(len(x_ds)):
    for j in range(len(y_local_f)):
        x = masl.SIMULATION_PARAMETERS['DATA_SIZE'] = X[j][i]
        y = Y[j][i]
        masl.SIMULATION_PARAMETERS['LOCAL_CPU_CAPABILITY'] = [y]
        result = generate_profile()
        users = result['users']
        profile: BRProfile = result['profile']
        local_cost = np.sum(u.local_cost() * u.active_probability for u in users)
        random_cost = profile._system_wide_cost_samples[0]
        diff = np.abs((local_cost/random_cost) - 1)
        if diff < 0.1:
            pairs.append([x, y, diff])
        Z[j][i] = diff
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')



ax.set_xlabel('data size')
ax.set_ylabel('frequency')
print(pairs)

fig, ax = plt.subplots()
min_Y = np.argmin(Z, axis=0)
ax.plot(x_ds, y_local_f[min_Y])
plt.show()

# for data_size in np.arange(1e8, 6e8, 1e8):
#     masl.SIMULATION_PARAMETERS['DATA_SIZE'] = data_size
#     result = generate_profile()
#     users = result['users']
#     profile: BRProfile = result['profile']
#     plt.plot(profile._system_wide_cost_samples, label=f'datasize={data_size}', linewidth=1)
#     # print(f'{data_size:e}')
#     # print(, )

# plt.legend()
# plt.show()
