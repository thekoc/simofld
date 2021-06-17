import sys
sys.path.insert(0, 'src')

from simofld.masl import CloudServer, MobileUser, RayleighChannel


from simofld.envs import create_env

if __name__ == '__main__':
    channels = [RayleighChannel() for _ in range()]
    nodes = [MobileUser(channels) for _ in range(2)]
    cloud_server = CloudServer()
    with create_env([node.main_loop() for node in nodes], until=10) as env:
        env.g.cloud_server = cloud_server
        env.run()