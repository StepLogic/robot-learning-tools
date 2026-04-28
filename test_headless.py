# test_scene.py
import habitat_sim
import numpy as np

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/gibson/Adairsville.glb"  # <-- update this
backend_cfg.enable_physics = False

sensor_spec = habitat_sim.CameraSensorSpec()
sensor_spec.uuid = "color_sensor"
sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
sensor_spec.resolution = [256, 256]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor_spec]

cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
print("Sim with scene: OK")

obs = sim.get_sensor_observations()
print("Observation shape:", obs["color_sensor"].shape)
sim.close()
print("Done.")