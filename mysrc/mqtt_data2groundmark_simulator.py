#!/usr/bin/env python3

"""
Description :
-------------
    Envoie des données aléatoires sur les topics nécessaires
    au script data2groundmark.py.
    CTRL+C peut être utilisé pour terminer le script.

Options :
---------
    * aucune

Utilisation :
-------------
    ./mqtt_data2groundmark_simulator.py

Auteur :
--------
    Olivier Henry (SAFIRE)

Version :
---------
    * 0.1.0 : première version du script.
    * 1.0.0 : version finale.

"""


__author__ = 'Olivier Henry <olivier.henry@safire.fr>'
__version__ = '1.0.0'
__creation_date__ = '12/05/2025'
__modification_date__ = '12/05/2025'


import time
import random
import struct
import paho.mqtt.client as mqtt


random.seed()


def random_data(topic):
    """

    :param str topic:
        topic sur lequel envoyer la donnée générée aléatoirement
    """

    if 'alt' in topic:
        data = 5000. + random.uniform(-500, 500)
    elif 'thead' in topic:
        data = 90. + random.uniform(-10, 10)
    elif 'pitch' in topic:
        data = 3. + random.uniform(-0.5, 1.5)
    elif 'roll' in topic:
        data = 0. + random.uniform(-3, 3)
    elif 'lat' in topic:
        data = 42. + random.uniform(-0.5, 0.5)
    else:
        data = 1.5 + random.uniform(-0.5, 0.5)
    return data


def on_publish(client, userdata, result):
    print('data published')
    pass


def mqtt_on_disconnect(client, userdata, rc):
    print('disconnection from mosquitto successfull !')


##################
#      MAIN      #
##################


# initialisation des variables
topics = ['aipov/altitude/alt_imu1_m/synchro',
          'aipov/attitude/thead_imu1_deg/synchro',
          'aipov/attitude/pitch_imu1_deg/synchro',
          'aipov/attitude/roll_imu1_deg/synchro',
          'aipov/position_horizontale/lat_imu1/synchro',
          'aipov/position_horizontale/lon_imu1/synchro']


# connexion au broker mqtt
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id='mqtt_data2groundmark_simulator')
mqtt_client.on_publish = on_publish
mqtt_client.connect('127.0.0.1', 1883)
start_time = time.time()
while True:
    try:

        # publication des messages
        for topic in topics:
            mqtt_client.publish(topic, struct.pack('f', random_data(topic)))

        # mise en pause du script pour respecter une boucle d'1s
        time.sleep(1.0 - ((time.time() - start_time) % 1.0))

    except KeyboardInterrupt:
        print('stopping now')
        break
