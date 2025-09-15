#!/usr/bin/python3.6

import paho.mqtt.client as mqtt
from struct import *
import time


ADDRESS="127.0.0.1"
PORT=1883
TOPIC="ice871/control/FAILICE_RS/+"
QOS=1
KEEPALIVE=60


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("\nConnected with result code "+str(rc))


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, message):

    if 'metadata' in message.topic :
        print ("\nReceive the metadata:")

        dico = dict(x.split('=') for x in message.payload.decode('ascii').split('\n'))
        global format
        format = dico['format']
        print ( dico )

    if 'mesure' in message.topic :
        print ("\nReceive the measure:")

        if len(message.payload) == 20:
            if "float" in format:
                (tv_sec, tv_nsec, val) = unpack( 'llf', message.payload )
                print( str(tv_sec)+"."+str(tv_nsec)+" "+str( val ) )
            else:
                (tv_sec, tv_nsec, val) = unpack( 'lli', message.payload )
                print( str(tv_sec)+"."+str(tv_nsec)+" "+str( val ) )
        else:
            if len( message.payload ) == 24:
                if "float" in format:
                    (tv_sec, tv_nsec, val) = unpack( "lld", message.payload )
                    print( str(tv_sec)+"."+str(tv_nsec)+" "+str( val ) )
                else:
                    (tv_sec, tv_nsec, val) = unpack( "lll", message.payload )
                    print( str(tv_sec)+"."+str(tv_nsec)+" "+str( val ) )
            else:
                print( '\tnot decoded ', len(message.payload) )



# Create the mqtt client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message=on_message


# Connect to mqtt client
client.connect(ADDRESS, PORT, KEEPALIVE);


# Start the loop
client.loop_start()


# Subscribe to topic
print("\nSubscribe to topic '"+TOPIC)
client.subscribe(TOPIC, QOS)


# Wait 20 sec
time.sleep(20)


# Stop the loop
client.loop_stop()


# Disconnect from mqtt client
client.disconnect()
