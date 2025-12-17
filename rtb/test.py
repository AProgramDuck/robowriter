import roboticstoolbox as rtb
def test_robot_initialization():
    robot = rtb.models.DH.Puma560()
    assert robot.n == 6
    assert robot.name == "Puma 560"
    assert len(robot.links) == 6
    print("Robot initialized successfully with", robot.n, "joints.")
    for i, link in enumerate(robot.links):
        print(f"Link {i+1}: {link}")
    return robot
if __name__ == "__main__":
    test_robot_initialization()
    