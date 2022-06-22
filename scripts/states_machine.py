#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import smach
import smach_ros

vet = []
ranges = []
front = []
back = []
global started_laser
started_laser = False

# Ângulo mínimo entre dois possíveis caminhos para que seja considerada evidência de bifurcação
MIN_ANGLE = 25
# Distância mínima para detectar uma bifurcação
MIN_DIST = 2.5
# Distância para deteccção de saída (a missão termina se a distância medida for maior que MAX_DETECT_DIST)
MAX_DETECT_DIST = float('inf')
# Distância usada como referência para identificar dead end. Quanto menor, mais perto chega
RET_DIST = 0.9


def callback_laser(laser):
    """Callback do laser planar"""

    global ranges, front, back, started_laser

    # vetor com valores de distância obtidos pelo laser 2d Hokuyo em uma volta completa de 360º ao longo do plano xy
    # o primeiro valor do vetor deve corresponder ao laser atrás do robô (6 horas)
    ranges = list(laser.ranges)

    for i in range(len(ranges)):
        if ranges[i] == 0:
            # random value
            ranges[i] = 10
        if ranges[i] >= MAX_DETECT_DIST:
            ranges[i] = ranges[i - 1]

    # divide o vetor de ranges
    front = ranges[70:290]  # 180º a frente do robô
    back = ranges[250:] + ranges[:110]  # 180º atrás do robô
    started_laser = True


def bif_detect_full(laser_data):
    """Recebe o vetor de ranges para cada ângulo obtidos do laser planar para detecção de bifurcação."""
    angles = []
    for i in range(len(laser_data)):
        # para cada valor presente no vetor de ranges
        laser_data[i] = round(laser_data[i], 2)
        if laser_data[i] > MIN_DIST:
            # caso o valor seja maior que a distância parâmetro mínima, adiciona ângulo no novo vetor
            angles.append(i)

    # gera vetor de diferenças entre elementos vizinhos
    angles_diff = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]

    # verifica quantos dos valores do vetor anterior são maiores que um parâmetro mínimo para bifurcação
    bif_angles = [e for e in angles_diff if e > MIN_ANGLE]

    if len(bif_angles) >= 1:
        # bifurcação detectada
        return True

    # não há bifurcação detectada
    return False


def bif_detect_180(reverse=False):
    """Recebe o vetor de ranges para cada ângulo obtidos do laser planar para detecção de bifurcação.
    Desconsidera os primeiros e os últimos 45 valores do vetor, restringindo a abertura do laser"""

    if reverse:
        data_ranges = back
    else:
        data_ranges = front

    angles = []
    for i in range(len(data_ranges)):
        data_ranges[i] = round(data_ranges[i], 2)
        if data_ranges[i] > MIN_DIST:
            angles.append(i)

    angles_diff = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    bif_angles = [e for e in angles_diff if e > MIN_ANGLE]
    if len(bif_angles) >= 1:
        return True
    else:
        return False


def deadend_detected(laser_data):
    """Recebe o vetor de ranges para cada ângulo obtidos do laser planar para detecção de ponto de retorno"""
    data_avg = round(sum(laser_data[:]) / len(laser_data[:]), 2)
    if data_avg < RET_DIST:
        return True
    else:
        return False


def exit_detect(laser_data):
    """Recebe o vetor de ranges para cada ângulo obtidos do laser planar para detecção de saída"""
    data_avg = round(sum(laser_data[100:170]) / len(laser_data[100:170]), 2)

    if data_avg > MAX_DETECT_DIST:
        return True
    else:
        return False


"""ESTADOS"""


class StandardControl(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['bifurcation', 'dead end', 'gallery exit', 'mantain'])

    def execute(self, userdata):
        # rospy.loginfo('Executing state: StandardControl')
        pub.publish('StandardControl')
        rate.sleep()
        if bif_detect_180(reverse=False):
            return 'bifurcation'
        elif deadend_detected(front):
            return 'dead end'
        elif exit_detect(front):
            return 'gallery exit'
        else:
            return 'mantain'


class RevStandardControl(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['bifurcation', 'dead end', 'gallery exit', 'mantain'])

    def execute(self, userdata):
        # rospy.loginfo('Executing state: RevStandardControl')
        pub.publish('RevStandardControl')
        rate.sleep()
        if bif_detect_180(reverse=True):
            return 'bifurcation'
        elif deadend_detected(back):
            return 'dead end'
        elif exit_detect(back):
            return 'gallery exit'
        else:
            return 'mantain'


class BifurcationControl(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['gallery', 'mantain'])

    def execute(self, userdata):
        # rospy.loginfo('Executing state: BifurcationControl')
        pub.publish('BifurcationControl')
        rate.sleep()
        if bif_detect_full(front):
            return 'mantain'
        else:
            return 'gallery'


class RevBifurcationControl(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['gallery', 'mantain'])

    def execute(self, userdata):
        # rospy.loginfo('Executing state: RevBifurcationControl')
        pub.publish('RevBifurcationControl')
        rate.sleep()
        if bif_detect_full(back):
            return 'mantain'
        else:
            return 'gallery'


class Exit(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['end'])

    def execute(self, userdata):
        # rospy.loginfo('Executing state: Exit')
        pub.publish('Exit')
        rate.sleep()
        return 'end'


if __name__ == '__main__':

    rospy.init_node("maq_estados", anonymous=True)  # inicia nó do pacote
    rospy.Subscriber("/scan", LaserScan, callback_laser)  # subscreve no nó do laser
    pub = rospy.Publisher('state', String, queue_size=10)
    rate = rospy.Rate(3)

    state_machine = smach.StateMachine(outcomes=['End'])

    std_ctrl = 'Gallery Control'
    rev_std_ctrl = 'Reverse Gallery Control'
    bif_ctrl = 'Bifurcation Control'
    rev_bif_ctrl = 'Reverse Bifurcation Control'

    while not started_laser:
        continue

    with state_machine:  # construções da máquina de estados
        smach.StateMachine.add(std_ctrl, StandardControl(),
                               transitions={'bifurcation': bif_ctrl, 'dead end': rev_std_ctrl,
                                            'gallery exit': 'Mission End', 'mantain': std_ctrl})
        smach.StateMachine.add(rev_std_ctrl, RevStandardControl(),
                               transitions={'bifurcation': rev_bif_ctrl, 'dead end': std_ctrl,
                                            'gallery exit': 'Mission End', 'mantain': rev_std_ctrl})
        smach.StateMachine.add(bif_ctrl, BifurcationControl(), transitions={'gallery': std_ctrl, 'mantain': bif_ctrl})
        smach.StateMachine.add(rev_bif_ctrl, RevBifurcationControl(),
                               transitions={'gallery': rev_std_ctrl, 'mantain': rev_bif_ctrl})
        smach.StateMachine.add('Mission End', Exit(), transitions={'end': 'End'})

    sis = smach_ros.IntrospectionServer('state_machine_view', state_machine, '/Mission Start')
    sis.start()

    try:
        outcome = state_machine.execute()
    except rospy.ROSInterruptException:
        pass
