# -*- coding: utf8 -*-

'''

Usage:

MeshcatVisualizer acts like a PyPlotVisualizer from the perspective of the
code: it is a System block that connects to the pose bundle output port of a
SceneGraph and visualizes the scene at every time step.

However, where PyPlotVisualizer opens its own visualizer window,
MeshcatVisualizer expects the command `meshcat-server`
to already be running in another terminal. The visualization
will be available at the web url provided by that server in real
time (either when the system is being simulated, or when the
animate() method is called).

'''

import argparse
import math
import os.path
import time

import numpy as np

from pydrake.all import (
    Context,
    DiagramBuilder,
    LeafSystem,
    PortDataType,
    Quaternion,
    RigidTransform,
    RotationMatrix
)

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (DispatchLoadMessage, SceneGraph)
from pydrake.lcm import DrakeMockLcm
from pydrake.multibody.multibody_tree import UniformGravityFieldElement
from pydrake.multibody.multibody_tree.multibody_plant import MultibodyPlant
from pydrake.multibody.multibody_tree.parsing import AddModelFromSdfFile
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator

from drake import lcmt_viewer_load_robot

from underactuated.utils import FindResource, Rgba2Hex

import meshcat
import meshcat.transformations as tf


class MeshcatVisualizer(LeafSystem):
    def __init__(self,
                 scene_graph,
                 draw_timestep=0.033333,
                 prefix="SceneGraph",
                 zmq_url="tcp://127.0.0.1:6000",
                 is_drawing_contact_force=False,
                 plant=None):
        LeafSystem.__init__(self)

        self.set_name('Meshcat')
        self.timestep = draw_timestep
        self._DeclarePeriodicPublish(draw_timestep, 0.0)

        # Pose bundle (from SceneGraph) input port.
        self._DeclareInputPort("lcm_visualization",
                               PortDataType.kAbstractValued, 0)

        # Set up meshcat.
        self.prefix = prefix
        self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        self.vis[self.prefix].delete()
        self._scene_graph = scene_graph

        # contact
        self.is_drawing_contact_force = is_drawing_contact_force
        if is_drawing_contact_force:
            assert not(plant is None)
            self.plant = plant
            # Contact results input port from MultiBodyPlant
            self._DeclareInputPort("contact_results",
                                   PortDataType.kAbstractValued, 0)
            self.contact_info_dict = dict()

            self.p_BC_dict = dict()
            # Each element of the contact_dict looks like:
            #  self.contact_info_dict["name"] is an instance of contact_info.
            self.contact_idx = 0
            self.t_previous = 0.

            self.state_input_port = self._DeclareInputPort(
                "state_input", PortDataType.kVectorValued,
                plant.num_positions() +
                plant.num_velocities())


    def load(self):
        """
        Loads `meshcat` visualization elements.
        @pre The `scene_graph` used to construct this object must be part of a
        fully constructed diagram (e.g. via `DiagramBuilder.Build()`).
        """
        # Intercept load message via mock LCM.
        mock_lcm = DrakeMockLcm()
        DispatchLoadMessage(self._scene_graph, mock_lcm)
        load_robot_msg = lcmt_viewer_load_robot.decode(
            mock_lcm.get_last_published_message("DRAKE_VIEWER_LOAD_ROBOT"))
        # Translate elements to `meshcat`.
        for i in range(load_robot_msg.num_links):
            link = load_robot_msg.link[i]
            [source_name, frame_name] = link.name.split("::")

            for j in range(link.num_geom):
                geom = link.geom[j]
                element_local_tf = RigidTransform(
                    RotationMatrix(Quaternion(geom.quaternion)),
                    geom.position).GetAsMatrix4()

                if geom.type == geom.BOX:
                    assert geom.num_float_data == 3
                    meshcat_geom = meshcat.geometry.Box(geom.float_data)
                elif geom.type == geom.SPHERE:
                    assert geom.num_float_data == 1
                    meshcat_geom = meshcat.geometry.Sphere(geom.float_data[0])
                elif geom.type == geom.CYLINDER:
                    assert geom.num_float_data == 2
                    meshcat_geom = meshcat.geometry.Cylinder(
                        geom.float_data[1],
                        geom.float_data[0])
                    # In Drake, cylinders are along +z
                    # In meshcat, cylinders are along +y
                    # Rotate to fix this misalignment
                    extra_rotation = tf.rotation_matrix(
                        math.pi/2., [1, 0, 0])
                    element_local_tf[0:3, 0:3] = \
                        element_local_tf[0:3, 0:3].dot(
                            extra_rotation[0:3, 0:3])
                elif geom.type == geom.MESH:
                    meshcat_geom = \
                        meshcat.geometry.ObjMeshGeometry.from_file(
                            geom.string_data[0:-3] + "obj")
                else:
                    print "UNSUPPORTED GEOMETRY TYPE ", \
                        geom.type, " IGNORED"
                    continue

                self.vis[self.prefix][source_name][str(link.robot_num)][
                    frame_name][str(j)]\
                    .set_object(meshcat_geom,
                                meshcat.geometry.MeshLambertMaterial(
                                    color=Rgba2Hex(geom.color)))
                self.vis[self.prefix][source_name][str(link.robot_num)][
                    frame_name][str(j)].set_transform(element_local_tf)

    def _DoPublish(self, context, event):
            self.draw(context)

    def _is_contact_existing(self, contact_info, dt, p_BC):
        '''
        contact_info: a contact to be checked if it describes the same contact as one
            one of the contacts in self.contact_info_dict
        p_BC: the coordinate of contact point in bodyB frame.
        returns true if contact_info is already in self.contact_dict_list
        '''
        is_contact_existing = False
        key_contact_dict = None
        for key, contact_info_i in self.contact_info_dict.iteritems():
            are_bodies_same1 = \
                contact_info_i.bodyA_index() == contact_info.bodyA_index() and \
                contact_info_i.bodyB_index() == contact_info.bodyB_index()
            are_bodies_same2 = \
                contact_info_i.bodyA_index() == contact_info.bodyB_index() and \
                contact_info_i.bodyB_index() == contact_info.bodyA_index()
            if are_bodies_same1 or are_bodies_same2:
                # Reaching here means that contact_info_i and contact_info
                # describe contact between the same pair of bodies.

                #TODO: support multiple contacts between the same pair of bodies, possibly by
                # checking if |contact_point_current - contact_point_previous| < timestep*slipping_speed
                v = np.sqrt(contact_info_i.separation_speed()**2 +
                            contact_info_i.slip_speed()**2)

                if np.linalg.norm(p_BC - self.p_BC_dict[key]) < 0.005:#v*dt*20:
                    is_contact_existing = True
                    key_contact_dict = key
                    break

        return is_contact_existing, key_contact_dict

    def get_visual_magnitude(self, magnitude):
        return magnitude / 20.

    def draw_contact_forces(self, context):
        contact_results = self.EvalAbstractInput(context, 1).get_value()
        x = self.EvalVectorInput(
            context, self.state_input_port.get_index()).get_value()
        t = context.get_time()

        context_plant = self.plant.CreateDefaultContext()
        tree = self.plant.tree()
        x_mutalbe = tree.get_mutable_multibody_state_vector(context_plant)
        x_mutalbe[:] = x
        world_frame = self.plant.world_frame()

        # First, set all existing contacts to be invalid
        is_contact_valid = dict()
        for key in self.contact_info_dict.keys():
            is_contact_valid[key] = False

        # check if i_contact is already in the current list of contact forces.
        # if yes, update the magnitude and location of the contact_info in self.contact_info_dict
        # if no, add the contact_info to self.contact_info_dict
        for i_contact in range(contact_results.num_contacts()):
            contact_info_i = contact_results.contact_info(i_contact)

            # contact ponit in frame B
            p_BC = tree.CalcPointsPositions(
                context=context_plant,
                frame_B=world_frame,
                p_BQi=contact_info_i.contact_point().reshape((3,1)),
                frame_A=tree.get_body(contact_info_i.bodyB_index()).body_frame()).flatten()

            dt = t - self.t_previous
            is_contact_existing, key = self._is_contact_existing(contact_info_i, dt, p_BC)
            if is_contact_existing:
                self.contact_info_dict[key] = contact_info_i
                is_contact_valid[key] = True
            else:
                # contact is new
                new_key = str(self.contact_idx)
                is_contact_valid[new_key] = True
                self.contact_info_dict[new_key] = contact_info_i
                self.vis[self.prefix]["contact_forces"][new_key].set_object(
                    meshcat.geometry.Cylinder(1, 0.01),
                    meshcat.geometry.MeshLambertMaterial(color=0xff0000))
                # Every new contact has its contact point in bodyB frame stored in
                # self.p_BC.dict
                self.p_BC_dict[new_key] = p_BC

                self.contact_idx += 1

        # delete invalid contact forces
        for key, is_valid in is_contact_valid.iteritems():
            if not is_valid:
                self.vis[self.prefix]["contact_forces"][key].delete()
                del self.contact_info_dict[key]

        # visualize all valid contact forces, and delete invalid contact forces
        for key, contact_info in self.contact_info_dict.iteritems():
            R = np.zeros((3,3))
            magnitude = np.linalg.norm(contact_info.contact_force())
            y = contact_info.contact_force()/magnitude
            R[:, 1] = y
            R[:, 0] = [0, -y[2], y[1]]
            R[:, 2] = np.cross(R[:, 0], y)

            # shift cylinder up by visual_magnitude/2 and scale by visual_magnitude
            visual_magnitude = self.get_visual_magnitude(magnitude)
            T0 = tf.translation_matrix([0, visual_magnitude/2, 0])
            T0[1,1] = visual_magnitude

            T1 = np.eye(4)
            T1[0:3, 0:3] = R
            T1[0:3, 3] = contact_info.contact_point()

            self.vis[self.prefix]["contact_forces"][key].set_transform(T1.dot(T0))

        # update t_previous
        self.t_previous = t


    def draw(self, context):
        assert(isinstance(context, Context))

        pose_bundle = self.EvalAbstractInput(context, 0).get_value()

        for frame_i in range(pose_bundle.get_num_poses()):
            # SceneGraph currently sets the name in PoseBundle as
            #    "get_source_name::frame_name".
            [source_name, frame_name] = pose_bundle.get_name(frame_i)\
                .split("::")
            model_id = pose_bundle.get_model_instance_id(frame_i)
            # The MBP parsers only register the plant as a nameless source.
            # TODO(russt): path should say a lot more about the MultiBodyTree.
            # TODO(russt): short term: add model instance id?
            self.vis[self.prefix][source_name][str(model_id)][frame_name]\
                .set_transform(pose_bundle.get_pose(frame_i).matrix())

        # visualize contact forces
        if self.is_drawing_contact_force:
            self.draw_contact_forces(context)


    def animate(self, log, resample=True):
        # TODO(russt): Finish this.
        print "MeshcatVisualizer: Animation is not (re-implemented) yet. " \
              "Coming soon."
        # Log would need to be a PoseBundle, instead of a vector of double
        # from SignalLogger?


# Cart-Pole with simple geometry.
def cartPoleTest(args):
    file_name = FindResourceOrThrow(
        "drake/examples/multibody/cart_pole/cart_pole.sdf")
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    cart_pole = builder.AddSystem(MultibodyPlant())
    AddModelFromSdfFile(
        file_name=file_name, plant=cart_pole, scene_graph=scene_graph)
    cart_pole.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))
    cart_pole.Finalize(scene_graph)
    assert cart_pole.geometry_source_is_registered()

    builder.Connect(
        cart_pole.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(cart_pole.get_source_id()))

    visualizer = builder.AddSystem(MeshcatVisualizer(scene_graph))
    builder.Connect(scene_graph.get_pose_bundle_output_port(),
                    visualizer.get_input_port(0))

    diagram = builder.Build()
    visualizer.load()

    diagram_context = diagram.CreateDefaultContext()
    cart_pole_context = diagram.GetMutableSubsystemContext(
        cart_pole, diagram_context)

    cart_pole_context.FixInputPort(
        cart_pole.get_actuation_input_port().get_index(), [0])

    cart_slider = cart_pole.GetJointByName("CartSlider")
    pole_pin = cart_pole.GetJointByName("PolePin")
    cart_slider.set_translation(context=cart_pole_context, translation=0.)
    pole_pin.set_angle(context=cart_pole_context, angle=2.)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.Initialize()
    simulator.StepTo(args.duration)


# Kuka IIWA with mesh geometry.
def kukaTest(args):
    file_name = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision"
        ".sdf")
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    kuka = builder.AddSystem(MultibodyPlant())
    AddModelFromSdfFile(
        file_name=file_name, plant=kuka, scene_graph=scene_graph)
    kuka.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))
    kuka.Finalize(scene_graph)
    assert kuka.geometry_source_is_registered()

    builder.Connect(
        kuka.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(kuka.get_source_id()))

    visualizer = builder.AddSystem(MeshcatVisualizer(scene_graph))
    builder.Connect(scene_graph.get_pose_bundle_output_port(),
                    visualizer.get_input_port(0))

    diagram = builder.Build()
    visualizer.load()

    diagram_context = diagram.CreateDefaultContext()
    kuka_context = diagram.GetMutableSubsystemContext(
        kuka, diagram_context)

    kuka_context.FixInputPort(
        kuka.get_actuation_input_port().get_index(), np.zeros(
            kuka.get_actuation_input_port().size()))

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.Initialize()
    simulator.StepTo(args.duration)


def main():
    # Usage demo: simulate and then animate a simple cartpole.

    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run sim.",
                        default=10.0)
    parser.add_argument("--test",
                        action="store_true",
                        help="Help out CI by launching a meshcat server for "
                             "the duration of the test.")
    args = parser.parse_args()

    meshcat_server_p = None
    if args.test:
        print "Spawning"
        import subprocess
        meshcat_server_p = subprocess.Popen(["meshcat-server"])
    else:
        print "Warning: if you have not yet run meshcat-server in another " \
              "terminal, this will hang."

    cartPoleTest(args)

    kukaTest(args)

    if meshcat_server_p is not None:
        meshcat_server_p.kill()
        meshcat_server_p.wait()


if __name__ == "__main__":
    main()
