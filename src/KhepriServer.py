import sys
def warn(msg):
    print(msg, file=sys.stderr)
    sys.stderr.flush()

import math
import time
import os.path
#from math import *
from typing import List, Tuple, NewType
import struct
from functools import partial

Size = NewType('Size', int)
Id = Size
MatId = Size
Float3 = Tuple[float,float,float]
RGB = Tuple[float,float,float]
RGBA = Tuple[float,float,float,float]

#######################################
# Communication
#Python provides sendall but not recvall
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def dump_exception(ex, conn):
    warn('Dumping exception!!!')
    warn("".join(traceback.TracebackException.from_exception(ex).format()))
    w_str("".join(traceback.TracebackException.from_exception(ex).format()), conn)

def r_struct(s, conn):
    return s.unpack(recvall(conn, s.size))[0]

def w_struct(s, e, conn):
    conn.sendall(s.pack(e))

def e_struct(s, e, ex, conn):
    w_struct(s, e, conn)
    dump_exception(ex, conn)

def r_tuple_struct(s, conn):
    return s.unpack(recvall(conn, s.size))

def w_tuple_struct(s, e, conn):
    conn.sendall(s.pack(*e))

def e_tuple_struct(s, e, ex, conn):
    w_tuple_struct(s, e, conn)
    dump_exception(ex, conn)

def r_list_struct(s, conn):
    n = r_int(conn)
    es = []
    for i in range(n):
        es.append(r_struct(s, conn))
    return es

def w_list_struct(s, es, conn):
    w_int(len(es), conn)
    for e in es:
        w_struct(s, e, conn)

def e_list(ex, conn):
    w_int(-1, conn)
    dump_exception(ex, conn)

int_struct = struct.Struct('i')
float_struct = struct.Struct('d')
byte_struct = struct.Struct('1B')

def w_None(e, conn)->None:
    w_struct(byte_struct, 0, conn)

e_None = partial(e_struct, byte_struct, 127)

def r_bool(conn)->bool:
    return r_struct(byte_struct, conn) == 1
def w_bool(b:bool, conn)->None:
    w_struct(byte_struct, 1 if b else 0, conn)

e_bool = partial(e_struct, byte_struct, 127)

r_int = partial(r_struct, int_struct)
w_int = partial(w_struct, int_struct)
e_int = partial(e_struct, int_struct, -12345678)

r_float = partial(r_struct, float_struct)
w_float = partial(w_struct, float_struct)
e_float = partial(w_struct, float_struct, math.nan)

r_Size = partial(r_struct, int_struct)
w_Size = partial(w_struct, int_struct)
e_Size = partial(e_struct, int_struct, -1)

def r_str(conn)->str:
    size = 0
    shift = 0
    byte = 0x80
    while byte & 0x80:
        try:
            byte = ord(conn.recv(1))
        except TypeError:
            raise IOError('Buffer empty')
        size |= (byte & 0x7f) << shift
        shift += 7
    return recvall(conn, size).decode('utf-8')

def w_str(s:str, conn):
    size = len(s)
    array = bytearray()
    while True:
        byte = size & 0x7f
        size >>= 7
        if size:
            array.append(byte | 0x80)
        else:
            array.append(byte)
            break
    conn.send(array)
    conn.sendall(s.encode('utf-8'))

def e_str(ex, conn)->None:
    w_str("This an error!", conn)
    dump_exception(ex, conn)

float3_struct = struct.Struct('3d')
r_float3 = partial(r_tuple_struct, float3_struct)
w_float3 = partial(w_tuple_struct, float3_struct)
e_float3 = e_float

def r_List(f, conn):
    n = r_int(conn)
    es = []
    for i in range(n):
        es.append(f(conn))
    return es

def w_List(f, es, conn):
    w_int(len(es), conn)
    for e in es:
        f(e, conn)

e_List = e_list

r_List_int = partial(r_List, r_int)
w_List_int = partial(w_List, w_int)
e_List_int = e_List

r_List_Size = partial(r_List, r_Size)
w_List_Size = partial(w_List, w_Size)
e_List_Size = e_List

r_List_float = partial(r_List, r_float)
w_List_float = partial(w_List, w_float)
e_List_float = e_List

r_List_float3 = partial(r_List, r_float3)
w_List_float3 = partial(w_List, w_float3)
e_List_float3 = e_List

r_List_List_int = partial(r_List, r_List_int)
w_List_List_int = partial(w_List, w_List_int)
e_List_List_int = e_List

int_int_struct = struct.Struct('2i')
r_Tint_intT = partial(r_tuple_struct, int_int_struct)
w_Tint_intT = partial(w_tuple_struct, int_int_struct)

r_List_Tint_intT = partial(r_List, r_Tint_intT)
w_List_Tint_intT = partial(w_List, w_Tint_intT)
e_List_Tint_intT = e_List

##############################################################
# For automatic generation of serialize/deserialize code
import inspect

def is_tuple_type(t)->bool:
    return hasattr(t, '__origin__') and t.__origin__ is tuple

def tuple_elements_type(t):
    return t.__args__

def is_list_type(t)->bool:
    return hasattr(t, '__origin__') and t.__origin__ is list

def list_element_type(t):
    return t.__args__[0]

def method_name_from_type(t)->str:
    if is_list_type(t):
        return "List_" + method_name_from_type(list_element_type(t))
    elif is_tuple_type(t):
        return "T" + '_'.join([method_name_from_type(pt) for pt in tuple_elements_type(t)]) + 'T'
    elif t is None:
        return "None"
    else:
        return t.__name__

def deserialize_parameter(c:str, t)->str:
    if is_tuple_type(t):
        return '(' + ','.join([deserialize_parameter(c, pt) for pt in tuple_elements_type(t)]) + ',)'
    else:
        return f"r_{method_name_from_type(t)}({c})"

def serialize_return(c:str, t, e:str)->str:
    if is_tuple_type(t):
        return f"__r = {e}; " + '; '.join([serialize_return(c, pt, f"__r[{i}]")
                                           for i, pt in enumerate(tuple_elements_type(t))])
    else:
        return f"w_{method_name_from_type(t)}({e}, {c})"

def serialize_error(c:str, t, e:str)->str:
    if is_tuple_type(t):
        return serialize_error(c, tuple_elements_type(t)[0], e)
    else:
        return f"e_{method_name_from_type(t)}({e}, {c})"

def try_serialize(c:str, t, e:str)->str:
    return f"""
    try:
        {e}
    except Exception as __ex:
        warn('RMI Error!!!!')
        warn("".join(traceback.TracebackException.from_exception(__ex).format()))
        warn('End of RMI Error.  I will attempt to serialize it.')
        {serialize_error(c, t, "__ex")}"""

def generate_rmi(f):
    c = 'c'
    name = f.__name__
    sig = inspect.signature(f)
    rt = sig.return_annotation
    pts = [p.annotation for p in sig.parameters.values()]
    des_pts = ','.join([deserialize_parameter(c, p) for p in pts])
    body = try_serialize(c, rt, serialize_return(c, rt, f"{name}({des_pts})"))
    dict = globals()
    rmi_name = f"rmi_{name}"
    rmi_f = f"def {rmi_name}({c}):{body}"
    #warn(rmi_f)
    exec(rmi_f, dict)
    return dict[rmi_name]

##############################################################
# Socket server
import socket

backend_port = 12345
socket_server = None
connection = None
current_action = None
min_wait_time = 0.1
max_wait_time = 0.2
max_repeated = 1000

def set_backend_port(n:int)->int:
    global backend_port
    backend_port = n
    return backend_port


def set_max_repeated(n:int)->int:
    global max_repeated
    prev = max_repeated
    max_repeated = n
    return prev

def read_operation(conn):
    return r_int(conn)

def try_read_operation(conn):
    conn.settimeout(min_wait_time)
    try:
        return read_operation(conn)
    except socket.timeout:
        return -2
    finally:
        conn.settimeout(None)

def execute_read_and_repeat(op, conn):
    count = 0
    while True:
        if op == -1:
            return False
        execute(op, conn)
        count =+ 1
        if count > max_repeated:
            return False
        conn.settimeout(max_wait_time)
        try:
            op = read_operation(conn)
        except socket.timeout:
            break
        finally:
            conn.settimeout(None)
    return True;


operations = []

def provide_operation(name:str)->int:
    warn(f"Requested operation |{name}| -> {globals()[name]}")
    operations.append(generate_rmi(globals()[name]))
    return len(operations) - 1

# The first operation is the operation that makes operations available
operations.append(generate_rmi(provide_operation))


def execute(op, conn):
    operations[op](conn)

def wait_for_connection():
    global current_action
    warn('Waiting for connection...')
    current_action = accept_client

def start_server():
    global socket_server
    socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_server.bind(('localhost', backend_port))
    socket_server.settimeout(max_wait_time)
    socket_server.listen(5)
    wait_for_connection()

counter_accept_attempts = 1
def accept_client():
    global counter_accept_attempts
    global connection
    global current_action
    try:
        #warn(f"Is anybody out there? (attempt {counter_accept_attempts})")
        connection, client_address = socket_server.accept()
        warn('Connection established.')
        current_action = handle_client
    except socket.timeout:
        counter_accept_attempts += 1
        #warn('It does not seem that there is.')
        # keep trying
        pass
    except Exception as ex:
        warn('Something bad happened!')
        traceback.print_exc()
        #warn('Resetting socket server.')


def handle_client():
    conn = connection
    op = try_read_operation(conn)
    if op == -1:
        warn("Connection terminated.")
        wait_for_connection()
    elif op == -2:
        # timeout
        pass
    else:
        execute_read_and_repeat(op, conn)

import traceback
current_action = start_server

def execute_current_action():
    #warn(f"Execute {current_action}")
    try:
        current_action()
    except Exception as ex:
        traceback.print_exc()
        #warn('Resetting socket server.')
        warn('Killing socket server.')
        if connection:
            connection.close()
        wait_for_connection()
    finally:
        return max_wait_time # timer

################################################################################
# Now, the backend-specific part. First, import this file:
"""
from KhepriServer import set_backend_port, execute_current_action, Size, r_float3, w_float3
"""

# Then, define your own types, serializers/deserializers, and operations.
# E.g., for Blender, we might have:
"""
from mathutils import Vector, Matrix
Point3d = Vector
Vector3d = Vector
Id = Size
MatId = Size

def r_Vector(conn)->Vector:
    return Vector(r_float3(conn))

def w_Vector(e:Vector, conn)->None:
    w_float3((e.x, e.y, e.z))

e_Vector = e_float3

r_List_Vector = partial(r_List, r_Vector)
w_List_Vector = partial(w_List, w_Vector)
e_List_Vector = e_List

r_List_List_Vector = partial(r_List, r_List_Vector)
w_List_List_Vector = partial(w_List, w_List_Vector)
e_List_List_Vector = e_List
...

def circle(c:Point3d, v:Vector3d, r:float, mat:MatId)->Id:
    ...
"""

# Finally, to run the server, define the listening port:
"""
set_backend_port(12345)
"""
# and then either we run in batch mode:
"""
while True:
    execute_current_action()
"""
# or we use a timer:
"""
backend.timers.register(execute_current_action)
"""
# or an idle event handler
"""
backend.on_idle_event(execute_current_action)
"""
