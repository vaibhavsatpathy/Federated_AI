# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: TreeEnsemble.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import DataStructures_pb2 as DataStructures__pb2
try:
  FeatureTypes__pb2 = DataStructures__pb2.FeatureTypes__pb2
except AttributeError:
  FeatureTypes__pb2 = DataStructures__pb2.FeatureTypes_pb2

from DataStructures_pb2 import *

DESCRIPTOR = _descriptor.FileDescriptor(
  name='TreeEnsemble.proto',
  package='CoreML.Specification',
  syntax='proto3',
  serialized_options=_b('H\003'),
  serialized_pb=_b('\n\x12TreeEnsemble.proto\x12\x14\x43oreML.Specification\x1a\x14\x44\x61taStructures.proto\"\xc4\x06\n\x16TreeEnsembleParameters\x12\x44\n\x05nodes\x18\x01 \x03(\x0b\x32\x35.CoreML.Specification.TreeEnsembleParameters.TreeNode\x12\x1f\n\x17numPredictionDimensions\x18\x02 \x01(\x04\x12\x1b\n\x13\x62\x61sePredictionValue\x18\x03 \x03(\x01\x1a\xa5\x05\n\x08TreeNode\x12\x0e\n\x06treeId\x18\x01 \x01(\x04\x12\x0e\n\x06nodeId\x18\x02 \x01(\x04\x12\\\n\x0cnodeBehavior\x18\x03 \x01(\x0e\x32\x46.CoreML.Specification.TreeEnsembleParameters.TreeNode.TreeNodeBehavior\x12\x1a\n\x12\x62ranchFeatureIndex\x18\n \x01(\x04\x12\x1a\n\x12\x62ranchFeatureValue\x18\x0b \x01(\x01\x12\x17\n\x0ftrueChildNodeId\x18\x0c \x01(\x04\x12\x18\n\x10\x66\x61lseChildNodeId\x18\r \x01(\x04\x12#\n\x1bmissingValueTracksTrueChild\x18\x0e \x01(\x08\x12\\\n\x0e\x65valuationInfo\x18\x14 \x03(\x0b\x32\x44.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo\x12\x17\n\x0frelativeHitRate\x18\x1e \x01(\x01\x1a\x42\n\x0e\x45valuationInfo\x12\x17\n\x0f\x65valuationIndex\x18\x01 \x01(\x04\x12\x17\n\x0f\x65valuationValue\x18\x02 \x01(\x01\"\xcf\x01\n\x10TreeNodeBehavior\x12\x1e\n\x1a\x42ranchOnValueLessThanEqual\x10\x00\x12\x19\n\x15\x42ranchOnValueLessThan\x10\x01\x12!\n\x1d\x42ranchOnValueGreaterThanEqual\x10\x02\x12\x1c\n\x18\x42ranchOnValueGreaterThan\x10\x03\x12\x16\n\x12\x42ranchOnValueEqual\x10\x04\x12\x19\n\x15\x42ranchOnValueNotEqual\x10\x05\x12\x0c\n\x08LeafNode\x10\x06\"\xc7\x02\n\x16TreeEnsembleClassifier\x12\x42\n\x0ctreeEnsemble\x18\x01 \x01(\x0b\x32,.CoreML.Specification.TreeEnsembleParameters\x12Z\n\x17postEvaluationTransform\x18\x02 \x01(\x0e\x32\x39.CoreML.Specification.TreeEnsemblePostEvaluationTransform\x12?\n\x11stringClassLabels\x18\x64 \x01(\x0b\x32\".CoreML.Specification.StringVectorH\x00\x12=\n\x10int64ClassLabels\x18\x65 \x01(\x0b\x32!.CoreML.Specification.Int64VectorH\x00\x42\r\n\x0b\x43lassLabels\"\xb7\x01\n\x15TreeEnsembleRegressor\x12\x42\n\x0ctreeEnsemble\x18\x01 \x01(\x0b\x32,.CoreML.Specification.TreeEnsembleParameters\x12Z\n\x17postEvaluationTransform\x18\x02 \x01(\x0e\x32\x39.CoreML.Specification.TreeEnsemblePostEvaluationTransform*\x9d\x01\n#TreeEnsemblePostEvaluationTransform\x12\x0f\n\x0bNoTransform\x10\x00\x12\x1a\n\x16\x43lassification_SoftMax\x10\x01\x12\x17\n\x13Regression_Logistic\x10\x02\x12\x30\n,Classification_SoftMaxWithZeroClassReference\x10\x03\x42\x02H\x03P\x00\x62\x06proto3')
  ,
  dependencies=[DataStructures__pb2.DESCRIPTOR,],
  public_dependencies=[DataStructures__pb2.DESCRIPTOR,])

_TREEENSEMBLEPOSTEVALUATIONTRANSFORM = _descriptor.EnumDescriptor(
  name='TreeEnsemblePostEvaluationTransform',
  full_name='CoreML.Specification.TreeEnsemblePostEvaluationTransform',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NoTransform', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Classification_SoftMax', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Regression_Logistic', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Classification_SoftMaxWithZeroClassReference', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1422,
  serialized_end=1579,
)
_sym_db.RegisterEnumDescriptor(_TREEENSEMBLEPOSTEVALUATIONTRANSFORM)

TreeEnsemblePostEvaluationTransform = enum_type_wrapper.EnumTypeWrapper(_TREEENSEMBLEPOSTEVALUATIONTRANSFORM)
NoTransform = 0
Classification_SoftMax = 1
Regression_Logistic = 2
Classification_SoftMaxWithZeroClassReference = 3


_TREEENSEMBLEPARAMETERS_TREENODE_TREENODEBEHAVIOR = _descriptor.EnumDescriptor(
  name='TreeNodeBehavior',
  full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.TreeNodeBehavior',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BranchOnValueLessThanEqual', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BranchOnValueLessThan', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BranchOnValueGreaterThanEqual', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BranchOnValueGreaterThan', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BranchOnValueEqual', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BranchOnValueNotEqual', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LeafNode', index=6, number=6,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=696,
  serialized_end=903,
)
_sym_db.RegisterEnumDescriptor(_TREEENSEMBLEPARAMETERS_TREENODE_TREENODEBEHAVIOR)


_TREEENSEMBLEPARAMETERS_TREENODE_EVALUATIONINFO = _descriptor.Descriptor(
  name='EvaluationInfo',
  full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='evaluationIndex', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.evaluationIndex', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='evaluationValue', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.evaluationValue', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=627,
  serialized_end=693,
)

_TREEENSEMBLEPARAMETERS_TREENODE = _descriptor.Descriptor(
  name='TreeNode',
  full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='treeId', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.treeId', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nodeId', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.nodeId', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nodeBehavior', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.nodeBehavior', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='branchFeatureIndex', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.branchFeatureIndex', index=3,
      number=10, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='branchFeatureValue', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.branchFeatureValue', index=4,
      number=11, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='trueChildNodeId', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.trueChildNodeId', index=5,
      number=12, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='falseChildNodeId', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.falseChildNodeId', index=6,
      number=13, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='missingValueTracksTrueChild', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.missingValueTracksTrueChild', index=7,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='evaluationInfo', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.evaluationInfo', index=8,
      number=20, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relativeHitRate', full_name='CoreML.Specification.TreeEnsembleParameters.TreeNode.relativeHitRate', index=9,
      number=30, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TREEENSEMBLEPARAMETERS_TREENODE_EVALUATIONINFO, ],
  enum_types=[
    _TREEENSEMBLEPARAMETERS_TREENODE_TREENODEBEHAVIOR,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=226,
  serialized_end=903,
)

_TREEENSEMBLEPARAMETERS = _descriptor.Descriptor(
  name='TreeEnsembleParameters',
  full_name='CoreML.Specification.TreeEnsembleParameters',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nodes', full_name='CoreML.Specification.TreeEnsembleParameters.nodes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='numPredictionDimensions', full_name='CoreML.Specification.TreeEnsembleParameters.numPredictionDimensions', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basePredictionValue', full_name='CoreML.Specification.TreeEnsembleParameters.basePredictionValue', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TREEENSEMBLEPARAMETERS_TREENODE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=67,
  serialized_end=903,
)


_TREEENSEMBLECLASSIFIER = _descriptor.Descriptor(
  name='TreeEnsembleClassifier',
  full_name='CoreML.Specification.TreeEnsembleClassifier',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='treeEnsemble', full_name='CoreML.Specification.TreeEnsembleClassifier.treeEnsemble', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='postEvaluationTransform', full_name='CoreML.Specification.TreeEnsembleClassifier.postEvaluationTransform', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stringClassLabels', full_name='CoreML.Specification.TreeEnsembleClassifier.stringClassLabels', index=2,
      number=100, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int64ClassLabels', full_name='CoreML.Specification.TreeEnsembleClassifier.int64ClassLabels', index=3,
      number=101, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
    _descriptor.OneofDescriptor(
      name='ClassLabels', full_name='CoreML.Specification.TreeEnsembleClassifier.ClassLabels',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=906,
  serialized_end=1233,
)


_TREEENSEMBLEREGRESSOR = _descriptor.Descriptor(
  name='TreeEnsembleRegressor',
  full_name='CoreML.Specification.TreeEnsembleRegressor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='treeEnsemble', full_name='CoreML.Specification.TreeEnsembleRegressor.treeEnsemble', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='postEvaluationTransform', full_name='CoreML.Specification.TreeEnsembleRegressor.postEvaluationTransform', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=1236,
  serialized_end=1419,
)

_TREEENSEMBLEPARAMETERS_TREENODE_EVALUATIONINFO.containing_type = _TREEENSEMBLEPARAMETERS_TREENODE
_TREEENSEMBLEPARAMETERS_TREENODE.fields_by_name['nodeBehavior'].enum_type = _TREEENSEMBLEPARAMETERS_TREENODE_TREENODEBEHAVIOR
_TREEENSEMBLEPARAMETERS_TREENODE.fields_by_name['evaluationInfo'].message_type = _TREEENSEMBLEPARAMETERS_TREENODE_EVALUATIONINFO
_TREEENSEMBLEPARAMETERS_TREENODE.containing_type = _TREEENSEMBLEPARAMETERS
_TREEENSEMBLEPARAMETERS_TREENODE_TREENODEBEHAVIOR.containing_type = _TREEENSEMBLEPARAMETERS_TREENODE
_TREEENSEMBLEPARAMETERS.fields_by_name['nodes'].message_type = _TREEENSEMBLEPARAMETERS_TREENODE
_TREEENSEMBLECLASSIFIER.fields_by_name['treeEnsemble'].message_type = _TREEENSEMBLEPARAMETERS
_TREEENSEMBLECLASSIFIER.fields_by_name['postEvaluationTransform'].enum_type = _TREEENSEMBLEPOSTEVALUATIONTRANSFORM
_TREEENSEMBLECLASSIFIER.fields_by_name['stringClassLabels'].message_type = DataStructures__pb2._STRINGVECTOR
_TREEENSEMBLECLASSIFIER.fields_by_name['int64ClassLabels'].message_type = DataStructures__pb2._INT64VECTOR
_TREEENSEMBLECLASSIFIER.oneofs_by_name['ClassLabels'].fields.append(
  _TREEENSEMBLECLASSIFIER.fields_by_name['stringClassLabels'])
_TREEENSEMBLECLASSIFIER.fields_by_name['stringClassLabels'].containing_oneof = _TREEENSEMBLECLASSIFIER.oneofs_by_name['ClassLabels']
_TREEENSEMBLECLASSIFIER.oneofs_by_name['ClassLabels'].fields.append(
  _TREEENSEMBLECLASSIFIER.fields_by_name['int64ClassLabels'])
_TREEENSEMBLECLASSIFIER.fields_by_name['int64ClassLabels'].containing_oneof = _TREEENSEMBLECLASSIFIER.oneofs_by_name['ClassLabels']
_TREEENSEMBLEREGRESSOR.fields_by_name['treeEnsemble'].message_type = _TREEENSEMBLEPARAMETERS
_TREEENSEMBLEREGRESSOR.fields_by_name['postEvaluationTransform'].enum_type = _TREEENSEMBLEPOSTEVALUATIONTRANSFORM
DESCRIPTOR.message_types_by_name['TreeEnsembleParameters'] = _TREEENSEMBLEPARAMETERS
DESCRIPTOR.message_types_by_name['TreeEnsembleClassifier'] = _TREEENSEMBLECLASSIFIER
DESCRIPTOR.message_types_by_name['TreeEnsembleRegressor'] = _TREEENSEMBLEREGRESSOR
DESCRIPTOR.enum_types_by_name['TreeEnsemblePostEvaluationTransform'] = _TREEENSEMBLEPOSTEVALUATIONTRANSFORM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TreeEnsembleParameters = _reflection.GeneratedProtocolMessageType('TreeEnsembleParameters', (_message.Message,), {

  'TreeNode' : _reflection.GeneratedProtocolMessageType('TreeNode', (_message.Message,), {

    'EvaluationInfo' : _reflection.GeneratedProtocolMessageType('EvaluationInfo', (_message.Message,), {
      'DESCRIPTOR' : _TREEENSEMBLEPARAMETERS_TREENODE_EVALUATIONINFO,
      '__module__' : 'TreeEnsemble_pb2'
      # @@protoc_insertion_point(class_scope:CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo)
      })
    ,
    'DESCRIPTOR' : _TREEENSEMBLEPARAMETERS_TREENODE,
    '__module__' : 'TreeEnsemble_pb2'
    # @@protoc_insertion_point(class_scope:CoreML.Specification.TreeEnsembleParameters.TreeNode)
    })
  ,
  'DESCRIPTOR' : _TREEENSEMBLEPARAMETERS,
  '__module__' : 'TreeEnsemble_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.TreeEnsembleParameters)
  })
_sym_db.RegisterMessage(TreeEnsembleParameters)
_sym_db.RegisterMessage(TreeEnsembleParameters.TreeNode)
_sym_db.RegisterMessage(TreeEnsembleParameters.TreeNode.EvaluationInfo)

TreeEnsembleClassifier = _reflection.GeneratedProtocolMessageType('TreeEnsembleClassifier', (_message.Message,), {
  'DESCRIPTOR' : _TREEENSEMBLECLASSIFIER,
  '__module__' : 'TreeEnsemble_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.TreeEnsembleClassifier)
  })
_sym_db.RegisterMessage(TreeEnsembleClassifier)

TreeEnsembleRegressor = _reflection.GeneratedProtocolMessageType('TreeEnsembleRegressor', (_message.Message,), {
  'DESCRIPTOR' : _TREEENSEMBLEREGRESSOR,
  '__module__' : 'TreeEnsemble_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.TreeEnsembleRegressor)
  })
_sym_db.RegisterMessage(TreeEnsembleRegressor)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
