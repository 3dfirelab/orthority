#!/usr/bin/env python3

"""
Description :
-------------
    Récupère les paramètres suivants de la centrale par MQTT afin de réaliser
    les calculs de la marque au sol des caméras de l'ATR :
        - longitude
        - latitude
        - altitude
        - cap
        - roulis
        - tangage
    Le script se lance comme un script python standard, sans argument.
    CTRL+C doit être utilisé pour terminer le script, ou fermer le terminal.

Options :
---------
    * aucune

Utilisation :
-------------
    ./data2groundmark.py

Auteur :
--------
    Olivier Henry (SAFIRE)

Version :
---------
    * 0.1.0 : première version du script

"""


__author__ = 'Olivier Henry <olivier.henry@safire.fr>'
__version__ = '0.1.0'
__creation_date__ = '07/05/2025'
__modification_date__ = '07/05/2025'


import time
import struct
import logging
import pathlib
import tempfile
import paho.mqtt.client as mqtt

#ronan
import footprint
import os 
import warnings 
import shutil
roll, pitch, thead, altitude, longitude, latitude = None, None, None, None, None, None


def preparation_logging(log_file):
    """
    prépare le fichier de log : le format, sa localisation et le format des messages.

    :param pathlib.Path log_file:
        nom du fichier de log, au format pathlib.Path
    """

    logging.getLogger('').handlers = []
    logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w',
                        format='%(asctime)s : %(levelname)s : %(message)s')


def mqtt_on_connect(client, userdata, flags, rc):
    """
    callback de la fonction on_connect du client MQTT.
    grosso modo : ce que doit faire le script lors de la connexion au broker MQTT.

    :param client:
    :param userdata:
    :param flags:
    :param rc:
        result code : résultat de la connexion ; 0 = success, 1 = failed.
    """

    if rc == 0:
        print('connection to mosquitto successfull !')
        logging.info('connection to mosquitto successfull !')
    else:
        print('connection to mosquitto not successfull !')
        logging.warning('impossible to connect to mosquitto ; please check that mosquitto has been started or is '
                        'running')


def mqtt_on_disconnect(client, userdata, rc):
    """
    callback de la fonction on_disconnect du client MQTT.
    grosso modo : ce que doit faire le script lors de la déconnexion du broker MQTT.

    :param client:
    :param userdata:
    :param rc:
    """

    print('disconnection from mosquitto successfull !')
    logging.info('disconnection from mosquitto successfull !')


def mqtt_on_message(client, userdata, message):
    """
    callback de la fonction on_message du client MQTT.
    grosso modo : ce que doit faire le script lors de la réception d'un message

    :param client:
    :param userdata:
    :param message:
        message reçu par mqtt, normalement, c'est du binaire.
    """

    logging.info(f'message has been received from mosquitto : {[message.topic, message.payload]}')

    global roll, pitch, thead, altitude, longitude, latitude

    data = struct.unpack('f', message.payload)[0]
    if 'alt' in message.topic:
        altitude = data
    elif 'thead' in message.topic:
        thead = data
    elif 'pitch' in message.topic:
        pitch = data
    elif 'roll' in message.topic:
        roll = data
    elif 'lat' in message.topic:
        latitude = data
    elif 'lon' in message.topic:
        longitude = data


##################
#      MAIN      #
##################

# preparation du système de log
logfile = pathlib.Path(tempfile.gettempdir()).joinpath('data2groundmark.log')
preparation_logging(logfile)


# initialisation des variables
logging.info('initializing variables...')
loop_time = 10 # temps d'exécution du calcul en seconde
host = '127.0.0.1' # à changer lorsque le script tournera en environnement opérationnel
port = 1883
mqtt_name = 'data2groundmark'
topics = ['aipov/altitude/alt_imu1_m/synchro',
          'aipov/attitude/thead_imu1_deg/synchro',
          'aipov/attitude/pitch_imu1_deg/synchro',
          'aipov/attitude/roll_imu1_deg/synchro',
          'aipov/position_horizontale/lat_imu1/synchro',
          'aipov/position_horizontale/lon_imu1/synchro']
logging.info(f'variables initialized - mqtt_name: {mqtt_name} ; host: {host} ; topics: {topics}')


# connexion au broker MQTT
logging.info('tentative de connexion à mosquitto ...')
print('connecting to mosquitto...')
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=mqtt_name)
mqtt_client.on_connect = mqtt_on_connect
mqtt_client.on_disconnect = mqtt_on_disconnect
mqtt_client.on_message = mqtt_on_message
mqtt_client.connect(host, port)
mqtt_client.subscribe([(topic, 0) for topic in topics])
mqtt_client.loop_start()
start_time = time.time()

#initialization calcul footprint
crs_code = 32631

wkdir = '/tmp/paugam/footprint_wkdir/'
if os.path.isdir(wkdir): shutil.rmtree(wkdir)
os.makedirs(wkdir, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")
intparamFile = "{:s}/io/as240051_int_param.yaml".format(indir)
correction_xyz = [-4.54166272e-05,  1.46991992e-04,  3.92582905e-04]
correction_opk = [ -4.62240706e-01,2.50020186e+00,  1.76677744e-04]
    
crs_code=32631
crs = pyproj.CRS.from_epsg(crs_code)
# Define the coordinate systems
wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
# Initialize the transformer
transformer     = pyproj.Transformer.from_crs(wgs84, crs, always_xy=True)
transformer_inv = pyproj.Transformer.from_crs(crs, wgs84, always_xy=True)

#question:
##########
#time.time() is the time of the data

# boucle de calcul
while True:
    try:

        if (roll is not None and pitch is not None and thead is not None and altitude is not None
            and longitude is not None and latitude is not None):

            # insérer ici le calcul
            print(roll, pitch, thead, altitude, longitude, latitude)
        
            '''
            gdf_ = footprint.orthro(row_dummy_file, time.time(), 
                          latitude,longitude,altitude,roll,pitch,thead, 
                          correction_xyz,correction_opk)
            if gdf_footprint is None:
                gdf_footprint = gdf_
            else:
                gdf_footprint = pd.concat([gdf_,gdf_footprint])

            
            #transfert vers planete
            '''
            pass

        else:
            logging.error(f'impossible d\'effectuer le calcul, l\'une des variables est None : '
                          f'    roll {roll} | pitch {pitch} | thead {thead} | altitude {altitude} | longitude '
                          f'{longitude} | latitude {latitude} | ')

        # mise en pause du script pour respecter une boucle de XX secondes
        sleep = loop_time - ((time.time() - start_time) % loop_time)
        time.sleep(sleep)

    except KeyboardInterrupt:
        logging.info('interruption volontaire du script')
        print('disconnecting from mosquitto...')
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        break
