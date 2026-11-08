# Panda notes

Model: `assets/mujoco_menagerie/franka_emika_panda/scene.xml`

## Joints

| id | name | type | range (rad or m) | qpos index |
|---|---|---|---|---|
| 0 | joint1 | hinge | -2.897 to 2.897 | 0 |
| 1 | joint2 | hinge | -1.763 to 1.763 | 1 |
| 2 | joint3 | hinge | -2.897 to 2.897 | 2 |
| 3 | joint4 | hinge | -3.072 to -0.070 | 3 |
| 4 | joint5 | hinge | -2.897 to 2.897 | 4 |
| 5 | joint6 | hinge | -0.018 to 3.752 | 5 |
| 6 | joint7 | hinge | -2.897 to 2.897 | 6 |
| 7 | finger_joint1 | slide | 0.000 to 0.040 | 7 |
| 8 | finger_joint2 | slide | 0.000 to 0.040 | 8 |

## Actuators (what data.ctrl controls)

| id | name | ctrl range |
|---|---|---|
| 0 | actuator1 | -2.897 to 2.897 |
| 1 | actuator2 | -1.763 to 1.763 |
| 2 | actuator3 | -2.897 to 2.897 |
| 3 | actuator4 | -3.072 to -0.070 |
| 4 | actuator5 | -2.897 to 2.897 |
| 5 | actuator6 | -0.018 to 3.752 |
| 6 | actuator7 | -2.897 to 2.897 |
| 7 | actuator8 | 0.000 to 255.000 |

## Tendons

- split

## Totals

- joints: 9
- actuators: 8
- bodies: 12
