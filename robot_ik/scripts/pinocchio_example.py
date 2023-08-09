import nlopt
import numpy as np
import pinocchio as pin
from typing import Tuple

class IKSolver:
    """Inverse kinematics solver."""

    def __init__(
        self,
        filepath: str,
        q0: np.ndarray,
        joint_index: int,
        p_frame: np.ndarray,
        R_frame: np.ndarray,
        p_star: np.ndarray,
        R_star: np.ndarray,
    ) -> None:
        """Initializes the IK solver.

        Specifically, solves an IK problem where the joints are constrained based on
        the values provided in a model URDF and a specific joint's frame is constrained
        by a desired pose specified by the user.

        Uses Lie theory to represent the rotations, so it's all mathematically kosher!

        Parameters
        ----------
        filepath : str
            The filepath to the URDF. Is a relative location.
        q0 : np.ndarray, shape=(n,)
            The initial guess for the solver (and the "home" configuration that the
            solver tries to stay close to).
        joint_index : int
            The joint index to compute the IK constraint for.
        p_frame : np.ndarray, shape=(3,)
            The translation of the target frame wrt the joint with specified index.
        R_frame : np.ndarray, shape=(3, 3)
            The rotation of the target frame wrt the joint with specified index.
        p_star : np.ndarray, shape=(3,)
            The desired translation of the joint frame wrt the robot base frame.
        R_star : np.ndarray, shape=(3, 3)
            The desired rotation of the joint frame wrt the robot base frame.
        """
        # properties of the model
        self.model = pin.buildModelFromUrdf(filepath)
        self.data = self.model.createData()
        self.n = self.model.nq  # number of states
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        
        # parameters to pass into the optimizer
        self.q0 = q0
        self.joint_index = joint_index
        self._frame_counter = 0
        self.frame = pin.Frame(
            "target_frame_0",
            self.joint_index,  # parent joint index
            0,  # index of parent frame (not 100% sure what this does)
            pin.SE3(R_frame, p_frame),  # pose of frame wrt joint frame
            pin.FrameType.OP_FRAME,  # "operational frame" defined by user
        )
        self.model.addFrame(self.frame)
        self.frame_idx = self.model.getFrameId("target_frame_0")
        self.T_star = pin.SE3(R_star, p_star)

        # optimization program
        self.opt = opt = nlopt.opt(nlopt.LD_SLSQP, self.n)
        self.opt.set_xtol_rel(1e-6)
        self.opt.set_xtol_abs(1e-6)
        self.hval = None  # the 6D error vector of the pose
        self.gval = None  # the inequality constraint values

    def update_q0(self, q0_new: np.ndarray) -> None:
        """Updates q0."""
        self.q0 = q0_new

    def update_IK_frame(
        self,
        joint_index_new: int,
        R_frame_new: np.ndarray,
        p_frame_new: np.ndarray,
    ) -> None:
        """Updates the frame defining the IK constraint."""
        self._frame_counter += 1
        self.joint_index = joint_index_new
        self.frame = pin.Frame(
            f"target_frame_{self._frame_counter}",
            self.joint_index,
            0,
            pin.SE3(R_frame_new, p_frame_new),
            pin.FrameType.OP_FRAME,
        )
        self.model.addFrame(self.frame)
        self.frame_idx = self.model.getFrameId(f"target_frame_{self._frame_counter}")

    def update_p_and_R_star(self, p_new: np.ndarray, R_new: np.ndarray) -> None:
        """Updates p_star and R_star."""
        self.T_star = pin.SE3(R_new, p_new)

    def _make_fgh(self) -> None:
        """Makes the cost and constraint functions for NLOPT."""
        A = np.concatenate((-np.eye(self.n), np.eye(self.n)), axis=0)
        b = np.concatenate((self.q_min, -self.q_max))

        def f(q, grad):
            """Cost function.

            f(q) = 0.5 * norm(q - q0) ^ 2
            """
            qmq0 = q - self.q0
            if grad.size > 0:
                grad[:] = qmq0
            return 0.5 * qmq0 @ qmq0

        def g(result, q, grad):
            """Inequality constraints. g(q) <= 0. Box constraints on q."""
            if grad.size > 0:
                grad[:] = A
            gval = A @ q + b
            result[:] = gval
            self.gval = gval

        def h(result, q, grad):
            """Equality constraints. Enforces forward kinematics."""
            # computing FK and EE Jacobian in world frame
            pin.forwardKinematics(self.model, self.data, q)
            T_frame = pin.updateFramePlacement(self.model, self.data, self.frame_idx)
            iMd = T_frame.actInv(self.T_star)
            err = pin.log(iMd).vector
            result[:] = err
            self.hval = err

            if grad.size > 0:
                # this gradient is used in the pinocchio example IK code
                # see: github.com/stack-of-tasks/pinocchio/blob/master/examples/inverse-kinematics.py
                _J = pin.computeFrameJacobian(
                    self.model, self.data, q, self.frame_idx
                )  # Jacobian in joint frame
                J = -np.dot(pin.Jlog6(iMd.inverse()), _J)  # chain rule
                grad[:] = J

        return f, g, h

    def solve(self) -> Tuple[np.ndarray, bool]:
        """Attempts to solve the IK problem.

        Returns
        -------
        q_star : np.ndarray, shape=(n,)
            The optimizer that satisfies (p_star, R_star) = FK(q_star).
        success : bool
            Whether the optimization was successful.
        """
        # setting up the optimization program
        f, g, h = self._make_fgh()
        self.opt.set_min_objective(f)
        self.opt.add_inequality_mconstraint(g, 1e-6 * np.ones(2 * self.n))
        self.opt.add_equality_mconstraint(h, 1e-6 * np.ones(6))

        # solving and checking result
        q_star = self.opt.optimize(self.q0)
        # result = self.opt.last_optimize_result()  # if you need return codes
        if np.linalg.norm(self.hval) <= 1e-6 and np.all(self.gval <= 1e-6):
            success = True
        else:
            success = False
        return q_star, success

if __name__ == "__main__":

    # ################## #
    # INVERSE KINEMATICS #
    # ################## #

    # example of using the solver
    filepath = "../models/fr3_algr.urdf"
    q0_fr3 = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])  # home config
    q0_algr = np.array(
        [
            0.0, 1.5, 1.0, 1.0,
            0.0, 1.5, 1.0, 1.0,
            0.0, 1.5, 1.0, 1.0,
            0.5, 0.5, 1.5, 1.0,
        ]
    )  # closes the hand like a fist
    q0 = np.concatenate((q0_fr3, q0_algr))
    joint_index = 7  # the EE joint, the world frame is the 0th joint in pinocchio

    # [NOTE] the x and y axes of the EE joint are rotated 45 degrees from how the hand
    # is aligned, which produces all these sqrt(2) / 2 terms. this transformation below
    # locates the Allegro origin wrt the weird frame attached to the FR3 end-effector.
    # The x-axis of the palm faces "out" and the z-axis goes towards the fingers
    p_algr_origin = np.array(
        [
            0.13398 * np.sqrt(2) / 2,
            -0.13398 * np.sqrt(2) / 2,
            0.0265 + 0.107,  # the joint frame is 0.107m recessed from the flange
        ]
    )
    R_algr_origin = np.array(
        [
            [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2],
            [0.0, -np.sqrt(2) / 2, -np.sqrt(2) / 2],
            [1.0, 0.0, 0.0],
        ]
    )

    # since the x-axis of the hand faces out, this is saying that we want the palm
    # facing towards the sky. We also want the fingers to be facing "forward".
    p_star = np.array([0.55, 0.3, 0.4])
    R_star = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    solver = IKSolver(
        filepath,
        q0,
        joint_index,
        p_algr_origin,
        R_algr_origin,
        p_star,
        R_star,
    )
    q_star, success = solver.solve()
    print(q_star)
    print(f"Optimization: {'success!' if success else 'failure!'}")

    # ################## #
    # FORWARD KINEMATICS #
    # ################## #

    # building a model from a filepath
    model = pin.buildModelFromUrdf(filepath)
    data = model.createData()

    # forwardKinematics calls FK which modifies the data struct in place
    pin.forwardKinematics(model, data, q0)

    # you can easily query the pose of a frame attached to a joint
    # M is how pinocchio refers to transformations (elements of SE(3))
    # o refers to the origin (base frame) and i refers to the joint indices
    joint_poses = data.oMi
    for i, pose in enumerate(joint_poses):
        print(f"joint {i}")
        print(f"translation: {pose.translation}")
        print(f"rotation: {pose.rotation}")
        print()

    # similarly, as in the IK solver class, you can define custom frames and update
    # their frame placements, then query the transformation of those frames wrt base

    breakpoint()
