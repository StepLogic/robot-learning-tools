# test_headless.py
import habitat_sim

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = "NONE"  # No scene, just test the sim object
cfg.enable_physics = False

agent_cfg = habitat_sim.agent.AgentConfiguration()
sim_cfg = habitat_sim.Configuration(cfg, [agent_cfg])

sim = habitat_sim.Simulator(sim_cfg)
print("Simulator created OK")
sim.close()
print("Simulator closed OK")