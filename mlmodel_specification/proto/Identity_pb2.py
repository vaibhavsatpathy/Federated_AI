# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Identity.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Identity.proto',
  package='CoreML.Specification',
  syntax='proto3',
  serialized_options=_b('H\003'),
  serialized_pb=_b('\n\x0eIdentity.proto\x12\x14\x43oreML.Specification\"\n\n\x08IdentityB\x02H\x03\x62\x06proto3')
)




_IDENTITY = _descriptor.Descriptor(
  name='Identity',
  full_name='CoreML.Specification.Identity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=50,
)

DESCRIPTOR.message_types_by_name['Identity'] = _IDENTITY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Identity = _reflection.GeneratedProtocolMessageType('Identity', (_message.Message,), {
  'DESCRIPTOR' : _IDENTITY,
  '__module__' : 'Identity_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.Identity)
  })
_sym_db.RegisterMessage(Identity)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
