# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ArrayFeatureExtractor.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ArrayFeatureExtractor.proto',
  package='CoreML.Specification',
  syntax='proto3',
  serialized_options=_b('H\003'),
  serialized_pb=_b('\n\x1b\x41rrayFeatureExtractor.proto\x12\x14\x43oreML.Specification\"-\n\x15\x41rrayFeatureExtractor\x12\x14\n\x0c\x65xtractIndex\x18\x01 \x03(\x04\x42\x02H\x03\x62\x06proto3')
)




_ARRAYFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='ArrayFeatureExtractor',
  full_name='CoreML.Specification.ArrayFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='extractIndex', full_name='CoreML.Specification.ArrayFeatureExtractor.extractIndex', index=0,
      number=1, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=53,
  serialized_end=98,
)

DESCRIPTOR.message_types_by_name['ArrayFeatureExtractor'] = _ARRAYFEATUREEXTRACTOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ArrayFeatureExtractor = _reflection.GeneratedProtocolMessageType('ArrayFeatureExtractor', (_message.Message,), {
  'DESCRIPTOR' : _ARRAYFEATUREEXTRACTOR,
  '__module__' : 'ArrayFeatureExtractor_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.ArrayFeatureExtractor)
  })
_sym_db.RegisterMessage(ArrayFeatureExtractor)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
