#include <iostream>
#include <kdl/chain.hpp>
#include <kdl/chainfdsolver_recursive_newton_euler.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <vector>

using namespace KDL;
using namespace std;

void printFrame(Frame &T) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cout << T.M.data[i * 3 + j] << " ";
    }
    cout << T.p.data[i] << endl;
  }
  cout << "0 0 0 1" << endl;
}

int main(int argc, char **argv) {
  Chain JK;
  int joint_num = 2;
  JntArray q(joint_num);

  JK.addSegment(Segment(
      "link1", Joint(Vector(0, 0, 0.069), Vector(0, 0, 1), Joint::RotAxis),
      Frame(Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1), Vector(0, 0, 0.069)),
      RigidBodyInertia(
          4.27, Vector(-0.000038005, -0.0024889, 0.054452),
          RotationalInertia(0.0340068, 0.0340237, 0.00804672, -5.0973e-06,
                            2.9246e-05, 0.00154238))));

  JK.addSegment(Segment(
      "link2", Joint(Vector(0, 0, 0.073), Vector(0, -1, 0), Joint::RotAxis),
      Frame(Rotation(1, 0, 0, 0, 0, -1, 0, 1, 0), Vector(0, 0, 0.073)),
      RigidBodyInertia(10.1, Vector(0, 0.21252, 0.12053),
                       RotationalInertia(0.771684, 1.16388, 1.32438,
                                         -5.9634e-05, 0.258717, -0.258717))));

  ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(JK);

  // vector<float> qn = {0, 0, 90, 0, 0, 0};
  vector<float> qn = {0, 0};
  vector<int> q1 = {0, 70, -10, 0, 25, 0};

  for (int i = 0; i < joint_num; i++) {
    q(i) = qn[i] * PI / 180;
  }
  Frame T;
  fksolver.JntToCart(q, T);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cout << T.M.data[i * 3 + j] << " ";
    }
    cout << T.p.data[i] << endl;
  }
  cout << "0 0 0 1" << endl;

  JntArray Q(joint_num), Qd(joint_num), Qdd(joint_num);
  for (int i = 0; i < joint_num; i++) {
    Q(i) = qn[i] * PI / 180;
    Qd(i) = 0;
    Qdd(i) = 0;
  }
  JntArray torque(joint_num);
  vector<Wrench> wrenches;
  for (int i = 0; i < joint_num; i++) {
    wrenches.emplace_back(Wrench());
  }
  ChainIdSolver_RNE idsolver(JK, Vector(0, 0, -9.81));
  idsolver.CartToJnt(Q, Qd, Qdd, wrenches, torque);
  cout << "torque is :";
  for (int i = 0; i < joint_num; i++) {
    cout << torque(i) << " ";
  }
  cout << endl;

  return 0;
}
